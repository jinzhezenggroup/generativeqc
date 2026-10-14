"""Typed semilocal point arithmetic; not complete molecular Fast qualification."""

from __future__ import annotations

import ctypes
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS
from generativeqc_compiler.xc.geometry_cuda import (
    _emit_mixed_point,
    _emit_stationary_point,
)
from generativeqc_compiler.xc.semilocal_codegen import (
    emit_polarized_semilocal,
    emit_r2scan_program,
    polarized_feature_count,
)
from generativeqc_compiler.xc.spec import functional
from generativeqc_compiler.xc.split_hybrid_codegen import emit_split_hybrid_device_body


@pytest.fixture(scope="module", params=("host", "cuda"))
def point_backend(request: pytest.FixtureRequest) -> str:
    """Keep real-device arithmetic checks within the scheduled CUDA contract."""
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
            pytest.skip("scheduled CUDA point qualification is opt-in")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("CUDA point qualification requires finite srun")
    return request.param


@pytest.fixture(
    scope="module", params=("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X", "WB97M-V")
)
def point_probe(
    request: pytest.FixtureRequest,
    native_cxx: Any,
    tmp_path_factory: Any,
    point_backend: str,
) -> tuple[str, Any]:
    """Compile genuine float parameters/results and reject implicit promotion."""
    method = request.param
    semilocal = resolve_method(method, spin="polarized").primitives[0].functional
    width = polarized_feature_count(semilocal)
    directory = tmp_path_factory.mktemp("semilocal-fp32")
    cuda = point_backend == "cuda"
    generated = ["#include <cmath>", "#include <type_traits>"]
    generated.append("#include <cuda_runtime.h>" if cuda else "#define __device__")
    for scalar, name in (("double", "Strict"), ("float", "Mixed")):
        generated.append(f"namespace {name} {{")
        if method == "M06-2X":
            body, value_type, _ = emit_split_hybrid_device_body(
                method, scalar_type=scalar
            )
            body += f"using {name}Point = {value_type};\n"
        elif method == "R2SCAN":
            body = emit_r2scan_program(
                value_type=f"{name}Point",
                function_name=f"evaluate{name}",
                identity_constant=f"{name}Identity",
                qualifier="__device__ inline" if cuda else "inline",
                include_equal_spin=True,
                scalar_type=scalar,
            )
        else:
            body = emit_polarized_semilocal(
                semilocal,
                value_type=f"{name}Point",
                function_name=f"evaluate{name}",
                identity_constant=f"{name}Identity",
                production=True,
                equal_spin_function_name=f"evaluate{name}_equal_spin",
                function_qualifier="__device__ inline" if cuda else "inline",
                scalar_type=scalar,
            )
        generated.append(body)
        generated.append(
            f"static_assert(std::is_same_v<decltype({name}Point{{}}.energy_density), {scalar}>);"
        )
        generated.append("}")
    calls = []
    for mode, (scalar, name) in enumerate(
        (("double", "Strict"), ("float", "Mixed"), ("float", "Mixed_equal_spin"))
    ):
        arguments = ", ".join(
            f"static_cast<{scalar}>(input[{index}])" for index in range(width)
        )
        function_name = (
            "m06_2x_device" + ("_equal_spin" if mode == 2 else "")
            if method == "M06-2X"
            else f"evaluate{name}"
        )
        calls.append(
            f"case {mode}: {{ const auto value = {name.split('_')[0]}::{function_name}({arguments}); "
            "output[0] = value.energy_density; "
            f"for (unsigned channel = 0; channel < {width}; ++channel) "
            "output[channel + 1] = value.feature_derivative[channel]; return 0; }"
        )
    generated.append(
        "__device__ int point_value(const double* input, double* output, int mode) {"
        " switch (mode) { " + " ".join(calls) + " default: return 1; }}"
    )
    return method, _compile_probe(
        generated,
        width,
        width + 1,
        directory,
        native_cxx,
        point_backend,
        strict_promotion=True,
    )


def _compile_probe(
    generated: list[str],
    width: int,
    output_width: int,
    directory: Path,
    native_cxx: Any,
    backend: str,
    *,
    strict_promotion: bool = False,
) -> Any:
    """Execute the same generated adapter on host and the assigned CUDA device."""
    cuda = backend == "cuda"
    source = directory / ("point.cu" if cuda else "point.cpp")
    if cuda:
        generated.append(
            r"""
__global__ void point_kernel(const double* input, double* output, int mode) {
  point_value(input, output, mode);
}
extern "C" int point(const double* input, double* output, int mode) {
  if (mode < 0 || mode > 2) return int(cudaErrorInvalidValue);
  double* device_input = nullptr;
  double* device_output = nullptr;
  auto status = cudaMalloc(&device_input, POINT_WIDTH * sizeof(double));
  if (status == cudaSuccess) status = cudaMalloc(&device_output, OUTPUT_WIDTH * sizeof(double));
  if (status == cudaSuccess)
    status = cudaMemcpy(device_input, input, POINT_WIDTH * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    point_kernel<<<1, 1>>>(device_input, device_output, mode);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(output, device_output, OUTPUT_WIDTH * sizeof(double), cudaMemcpyDeviceToHost);
  if (device_output) {
    const auto released = cudaFree(device_output);
    if (status == cudaSuccess) status = released;
  }
  if (device_input) {
    const auto released = cudaFree(device_input);
    if (status == cudaSuccess) status = released;
  }
  return int(status);
}
""".replace("POINT_WIDTH", str(width)).replace("OUTPUT_WIDTH", str(output_width))
        )
    else:
        generated.append(
            'extern "C" int point(const double* input, double* output, int mode) {'
            "return point_value(input, output, mode); }"
        )
    source.write_text("\n".join(generated))
    binary = directory / "point.so"
    flags = (
        "-std=c++17",
        "-O2",
        "-I",
        str(directory),
        "-I",
        str(Path(__file__).resolve().parents[2] / "src"),
    )
    if strict_promotion:
        flags += (
            (
                "-Xcompiler=-Werror=double-promotion"
                if cuda
                else "-Werror=double-promotion"
            ),
        )
    if cuda:
        compiler = shutil.which("nvcc")
        if compiler is None:
            pytest.fail("scheduled point qualification requires nvcc")
        native_cxx = replace(native_cxx, compiler=compiler)
        flags = (
            "-forward-unknown-to-host-compiler",
            *flags,
            "-arch=native",
            "--ftz=false",
        )
    native_cxx.build_shared(
        [source],
        binary,
        compile_args=flags,
    )
    library = ctypes.CDLL(str(binary))
    vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.point.argtypes = (vector, vector, ctypes.c_int)
    library.point.restype = ctypes.c_int

    def evaluate(features: np.ndarray, mode: int) -> np.ndarray:
        output = np.empty(output_width)
        assert library.point(np.ascontiguousarray(features[:width]), output, mode) == 0
        return output

    return evaluate


@pytest.fixture(
    scope="module",
    params=(
        ("PBE0", 1),
        ("B3LYP", 3),
        ("R2SCAN", 2),
        ("M06-2X", SPLIT_HYBRIDS["M06-2X"]["functional_code"]),
        ("WB97M-V", 4),
    ),
)
def guarded_probe(
    request: pytest.FixtureRequest,
    native_cxx: Any,
    tmp_path_factory: Any,
    point_backend: str,
) -> tuple[str, Any]:
    """Compile the production Cartesian adapter, including its actual fallback."""
    method, code = request.param
    semilocal = resolve_method(method, spin="polarized").primitives[0].functional
    generated = [
        "#include <cmath>",
        "#include <cuda_runtime.h>"
        if point_backend == "cuda"
        else "#define __device__",
        "using std::isfinite;",
        '#include "dft/xc_point_mixed.hpp"',
        _emit_stationary_point(code, semilocal=semilocal),
        _emit_mixed_point(code, semilocal),
        r"""
__device__ int point_value(const double* input, double* output, int mode) {
  const double rho[2]{input[0], input[1]};
  const double gradient[2][3]{{input[2], input[3], input[4]}, {input[5], input[6], input[7]}};
  const double tau[2]{input[8], input[9]};
  bool used = false;
  const auto value = mode == 0 ? stationary_evaluate_point(rho, gradient, tau)
                              : stationary_evaluate_point_fp32(rho, gradient, tau, &used);
  output[0] = value.energy;
  for (unsigned spin = 0; spin < 2; ++spin) {
    output[1 + spin] = value.rho[spin];
    for (unsigned axis = 0; axis < 3; ++axis)
      output[3 + 3 * spin + axis] = value.gradient[spin][axis];
    output[9 + spin] = value.kinetic[spin];
  }
  output[11] = value.valid;
  output[12] = used;
  return 0;
}
""",
    ]
    return method, _compile_probe(
        generated,
        10,
        13,
        tmp_path_factory.mktemp("guarded-fp32"),
        native_cxx,
        point_backend,
    )


@pytest.mark.parametrize("equal_spin", (False, True))
def test_guarded_cartesian_point_matches_libxc(
    guarded_probe: tuple[str, Any],
    equal_spin: bool,
) -> None:
    """Independent response channels must survive the Cartesian ABI conversion."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    method, evaluate = guarded_probe
    random = np.random.default_rng(4192)
    used = 0
    for _ in range(64):
        density = np.exp(random.uniform(-10, 4, size=2))
        gradient = random.normal(size=(2, 3)) * density[:, None] * 0.2
        kinetic = np.sum(gradient**2, axis=1) / (8 * density) + density ** (5 / 3)
        if equal_spin:
            density[1], gradient[1], kinetic[1] = density[0], gradient[0], kinetic[0]
        inputs = np.concatenate((density, gradient.ravel(), kinetic))
        meta_gga = method in ("R2SCAN", "M06-2X", "WB97M-V")
        oracle = np.zeros((2, 6 if meta_gga else 4, 1))
        oracle[:, 0, 0], oracle[:, 1:4, 0] = density, gradient
        if meta_gga:
            oracle[:, 5, 0] = kinetic
        energy, potential, *_ = libxc.eval_xc(method, oracle, spin=1, deriv=1)
        sigma = potential[1][0]
        gradient_response = np.array(
            [
                2 * sigma[0] * gradient[0] + sigma[1] * gradient[1],
                sigma[1] * gradient[0] + 2 * sigma[2] * gradient[1],
            ]
        )
        expected = np.concatenate(
            (
                [energy[0] * density.sum()],
                potential[0][0],
                gradient_response.ravel(),
                0.5 * potential[3][0] if meta_gga else [0, 0],
            )
        )
        strict, mixed = evaluate(inputs, 0), evaluate(inputs, 1)
        assert strict[11] == mixed[11] == 1
        np.testing.assert_allclose(strict[:11], expected, rtol=2e-9, atol=2e-11)
        np.testing.assert_allclose(mixed[:11], expected, rtol=3e-4, atol=2e-6)
        used += int(mixed[12])
    assert used > 0


@pytest.mark.parametrize("tail", (0.0, 1e-40, 1e-60))
def test_unqualified_spin_tail_uses_unchanged_strict_point(
    guarded_probe: tuple[str, Any],
    tail: float,
) -> None:
    """A guarded fallback must not clip spin tails or change the strict result."""
    _, evaluate = guarded_probe
    inputs = np.array([0.3, tail, 0.02, -0.01, 0.03, tail, -tail, tail, 0.2, tail])
    strict, mixed = evaluate(inputs, 0), evaluate(inputs, 1)
    assert strict[12] == mixed[12] == 0
    np.testing.assert_array_equal(mixed, strict)


def test_composed_high_polarization_uses_strict_response(
    guarded_probe: tuple[str, Any],
) -> None:
    """Finite FP32 values do not establish a safe minority-spin derivative."""
    method, evaluate = guarded_probe
    if method == "PBE0":
        pytest.skip("PBE uses its separate normalized point owner")
    for densities in ((0.32, 0.001), (0.001, 0.32)):
        inputs = np.array(
            [*densities, 0.001, -0.002, 0.001, 0.001, 0.002, -0.001, 0.2, 0.1]
        )
        strict, mixed = evaluate(inputs, 0), evaluate(inputs, 1)
        assert mixed[12] == 0
        np.testing.assert_array_equal(mixed, strict)


@pytest.mark.parametrize("equal_spin", (False, True))
def test_fp32_interior_points_match_independent_libxc(
    point_probe: tuple[str, Any], equal_spin: bool
) -> None:
    """Cover physical gradients/kinetic density without reusing native equations."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    method, evaluate = point_probe
    random = np.random.default_rng(714)
    for _ in range(32):
        density = np.exp(random.uniform(-2, 1, size=2))
        gradient = random.normal(size=(2, 3)) * density[:, None] * 0.2
        kinetic = np.sum(gradient**2, axis=1) / (8 * density) + random.uniform(
            0.3, 2, size=2
        ) * density ** (5 / 3)
        if equal_spin:
            density[1] = density[0]
            gradient[1] = gradient[0]
            kinetic[1] = kinetic[0]
        features = np.array(
            [
                *density,
                gradient[0] @ gradient[0],
                gradient[0] @ gradient[1],
                gradient[1] @ gradient[1],
                *kinetic,
            ]
        )
        meta_gga = method in ("R2SCAN", "M06-2X", "WB97M-V")
        oracle_input = np.zeros((2, 6 if meta_gga else 4, 1))
        oracle_input[:, 0, 0] = density
        oracle_input[:, 1:4, 0] = gradient
        if meta_gga:
            oracle_input[:, 5, 0] = kinetic
        energy, potential, *_ = libxc.eval_xc(method, oracle_input, spin=1, deriv=1)
        expected = np.concatenate(
            (
                [energy[0] * density.sum()],
                potential[0][0],
                potential[1][0],
                potential[3][0] if meta_gga else [],
            )
        )
        strict = evaluate(features, 0)
        mixed = evaluate(features, 2 if equal_spin else 1)
        np.testing.assert_allclose(strict, expected, rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(mixed, expected, rtol=3e-4, atol=2e-6)
        np.testing.assert_array_equal(mixed, mixed.astype(np.float32).astype(float))
        if equal_spin:
            np.testing.assert_allclose(
                mixed, evaluate(features, 1), rtol=3e-4, atol=2e-6
            )


@pytest.mark.parametrize("scalar_type", ("half", "long double", "", None))
def test_semilocal_rejects_unimplemented_arithmetic(scalar_type: str) -> None:
    with pytest.raises(ValueError, match="scalar type"):
        emit_polarized_semilocal(
            functional("PBE"),
            value_type="Point",
            function_name="evaluate",
            identity_constant="Identity",
            scalar_type=scalar_type,
        )

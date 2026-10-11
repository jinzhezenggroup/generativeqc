"""Independent E/vxc gates for the common native automatic point source.

CUDA is opt-in and must run inside a finite Slurm GPU allocation. These are
point-source checks, not public CUDA XC/KS/force admission or cache evidence.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.common.compiler_process import run_compiler
from generativeqc_compiler.common.cuda_resources import parse_resources
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.dft.ao_cuda import emit_native_xc_contraction_kernels
from generativeqc_compiler.xc.automatic_point import (
    AutomaticPointSource,
    emit_automatic_point,
)
from generativeqc_compiler.xc.libxc_work import automatic_work_policy

NAMES = (
    "LDA_C_VWN_4",
    "LDA_C_CHACHIYO",
    "LDA_C_PZ_MOD",
    "GGA_X_APBE",
    "GGA_X_PBE_SOL",
    "GGA_X_RPBE",
    "GGA_X_B86",
    "GGA_C_APBE",
    "MGGA_X_LTA",
    "MGGA_X_GVT4",
    "MGGA_C_TPSS",
)
ROOT = Path(__file__).resolve().parents[2]


def _native_source(point: AutomaticPointSource) -> str:
    """Wrap the emitted point in a test-only packed ten-input/twelve-output ABI."""
    cuda = point.backend == "cuda"
    qualifier = "__device__ " if cuda else ""
    value_type = "DevicePointValue" if cuda else "SemilocalPointValue"
    native_device_value = re.search(
        r"struct DevicePointValue \{.*?\n\};",
        emit_native_xc_contraction_kernels(),
        re.DOTALL,
    )
    assert native_device_value is not None
    prelude = (
        "#include <cuda_runtime.h>\n" + native_device_value.group() + "\n"
        if cuda
        else '#include "dft/xc.hpp"\nusing generativeqc::dft::SemilocalPointValue;\n'
    )
    body = f"""
{qualifier}void packed_point(const double* input, double* output) {{
  const double rho[2]{{input[0], input[1]}};
  const double gradient[2][3]{{
      {{input[2], input[3], input[4]}}, {{input[5], input[6], input[7]}}}};
  const double tau[2]{{input[8], input[9]}};
  {value_type} value{{}};
  bool valid = true;
  @BEGIN@
    value = {point.function_name}(rho, gradient, tau);
  @END@
  output[0] = value.energy;
  for (unsigned spin = 0; spin < 2; ++spin) {{
    output[1 + spin] = value.rho[spin];
    for (unsigned axis = 0; axis < 3; ++axis)
      output[3 + 3 * spin + axis] = value.gradient[spin][axis];
    output[9 + spin] = value.kinetic[spin];
  }}
  output[11] = valid ? 1.0 : 0.0;
}}
"""
    body = body.replace("@BEGIN@", "" if cuda else "try {").replace(
        "@END@",
        "valid = value.valid;"
        if cuda
        else "} catch (const std::domain_error&) { valid = false; }",
    )
    if cuda:
        body += """
__global__ void automatic_point_probe(const double* input, double* output, size_t count) {
  const size_t point = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point < count) packed_point(input + 10 * point, output + 12 * point);
}
extern "C" int evaluate(const double* input, double* output, size_t count) {
  if (!count) return 0;
  double* device_input = nullptr;
  double* device_output = nullptr;
  cudaError_t status = cudaMalloc(&device_input, 10 * count * sizeof(double));
  if (status == cudaSuccess) status = cudaMalloc(&device_output, 12 * count * sizeof(double));
  if (status == cudaSuccess)
    status = cudaMemcpy(device_input, input, 10 * count * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    automatic_point_probe<<<(count + 127) / 128, 128>>>(device_input, device_output, count);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(output, device_output, 12 * count * sizeof(double), cudaMemcpyDeviceToHost);
  if (device_output) cudaFree(device_output);
  if (device_input) cudaFree(device_input);
  return static_cast<int>(status);
}
extern "C" int device_info(int* major, int* minor, int* driver, char* name) {
  cudaDeviceProp properties{};
  cudaError_t status = cudaGetDeviceProperties(&properties, 0);
  if (status == cudaSuccess) status = cudaDriverGetVersion(driver);
  if (status != cudaSuccess) return static_cast<int>(status);
  *major = properties.major;
  *minor = properties.minor;
  for (unsigned index = 0; index < 256; ++index) name[index] = properties.name[index];
  return 0;
}
"""
    else:
        body += """
extern "C" int evaluate(const double* input, double* output, size_t count) {
  for (size_t point = 0; point < count; ++point)
    packed_point(input + 10 * point, output + 12 * point);
  return 0;
}
"""
    return "#include <cmath>\n#include <stdexcept>\n" + prelude + point.source + body


@pytest.fixture(scope="session", params=("cpu", "cuda"))
def backend(request: pytest.FixtureRequest) -> str:
    selected = str(request.param)
    if selected == "cuda":
        if os.environ.get("GENERATIVEQC_XC_CUDA_TEST") != "1":
            pytest.skip("opt-in Slurm CUDA point-source gate")
        assert os.environ.get("SLURM_JOB_ID"), "real GPU gate requires srun"
        assert os.environ.get("CUDA_VISIBLE_DEVICES"), "preserve Slurm GPU visibility"
    return selected


@pytest.fixture(scope="session", params=NAMES)
def native_point(
    request: pytest.FixtureRequest,
    backend: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[AutomaticPointSource, ctypes.CDLL, Path]:
    point = emit_automatic_point(str(request.param), backend=backend)  # type: ignore[arg-type]
    directory = tmp_path_factory.mktemp(f"{backend}-{point.name}")
    path = directory / ("point.cu" if backend == "cuda" else "point.cpp")
    path.write_text(_native_source(point), encoding="utf-8")
    compiler = (
        os.environ.get("GENERATIVEQC_NVCC", "nvcc") if backend == "cuda" else "c++"
    )
    executable = shutil.which(compiler)
    if executable is None:
        pytest.skip(f"compiler unavailable: {compiler}")
    launcher = resolve_compiler_cache()
    compiler_version = run_compiler(
        [executable, "--version"], 30, label="point compiler version"
    )
    assert compiler_version.returncode == 0 and not compiler_version.timed_out
    options = [
        "-std=c++17" if backend == "cuda" else "-std=c++20",
        "-O2",
        "-I",
        str(ROOT / "src"),
        "-I",
        str(ROOT / "include"),
    ]
    options += (
        ["-arch=sm_120", "--fmad=false", "-Xcompiler=-fPIC", "-Xptxas=-v"]
        if backend == "cuda"
        else ["-fPIC"]
    )
    compiled = run_cached_compiler(
        [executable, *options, "-c", str(path), "-o", str(directory / "point.o")],
        180,
        label="automatic Libxc point probe",
    )
    (directory / "compile.log").write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0 and not compiled.timed_out, compiled.stderr
    binary = directory / "point.so"
    linked = run_compiler(
        [executable, "-shared", str(directory / "point.o"), "-o", str(binary)],
        60,
        label="automatic Libxc point probe link",
    )
    assert linked.returncode == 0 and not linked.timed_out, linked.stderr
    resources = parse_resources(compiled.stderr) if backend == "cuda" else ()
    if backend == "cuda":
        assert len(resources) == 1
        assert 0 < resources[0].registers <= 255
        assert resources[0].stack_bytes <= 4096
        assert resources[0].spill_load_bytes + resources[0].spill_store_bytes <= 4096
    library = ctypes.CDLL(str(binary))
    device = None
    if backend == "cuda":
        major, minor, driver = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        device_name = ctypes.create_string_buffer(256)
        library.device_info.argtypes = (ctypes.POINTER(ctypes.c_int),) * 3 + (
            ctypes.POINTER(ctypes.c_char),
        )
        library.device_info.restype = ctypes.c_int
        assert (
            library.device_info(
                ctypes.byref(major),
                ctypes.byref(minor),
                ctypes.byref(driver),
                device_name,
            )
            == 0
        )
        device = {
            "name": device_name.value.decode(),
            "compute_capability": [major.value, minor.value],
            "driver_version": driver.value,
        }
    (directory / "source-evidence.json").write_text(
        json.dumps(
            {
                "stage": "point-source-only",
                "name": point.name,
                "backend": backend,
                "expression_identity": point.expression_identity,
                "source_identity": point.source_identity,
                "translation_unit_hash": file_hash(path),
                "binary_hash": file_hash(binary),
                "object_hash": file_hash(directory / "point.o"),
                "features": point.features,
                "outputs": point.outputs,
                "compile_seconds": compiled.duration_seconds,
                "compiler": executable,
                "compiler_version": compiler_version.stdout.strip(),
                "compiler_options": options,
                "compiler_cache": {
                    "name": launcher.name,
                    "path": str(launcher.path),
                    "version": launcher.version,
                },
                "resources": [asdict(row) for row in resources],
                "device": device,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
                if backend == "cuda"
                else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")
                if backend == "cuda"
                else None,
                "public_cuda_admitted": False,
            },
            indent=2,
        )
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = (pointer, pointer, ctypes.c_size_t)
    library.evaluate.restype = ctypes.c_int
    return point, library, directory


def _evaluate(library: ctypes.CDLL, inputs: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(inputs, dtype=np.float64)
    output = np.empty((len(packed), 12))
    pointer = ctypes.POINTER(ctypes.c_double)
    status = library.evaluate(
        packed.ctypes.data_as(pointer), output.ctypes.data_as(pointer), len(packed)
    )
    assert status == 0, f"native point launch failed: {status}"
    return output


def _inputs(name: str, spin: int) -> np.ndarray:
    """Build physical inputs, including vacuum, empty spin, floors, and FHC."""
    random = np.random.default_rng(11182026)
    density = np.exp(random.uniform(np.log(1e-4), np.log(1.5), (37, 2)))
    gradients = 0.1 * random.normal(size=(37, 2, 3)) * density[:, :, None] ** (4 / 3)
    tau = 0.4 * density ** (5 / 3) + np.sum(gradients**2, axis=2) / (8 * density)
    inputs = np.column_stack((density, gradients.reshape(37, 6), tau))
    threshold = automatic_work_policy(name).density_threshold
    inputs[0] = 0
    inputs[1, [1, 5, 6, 7, 9]] = 0
    inputs[2, [0, 2, 3, 4, 8]] = 0
    inputs[3, [1, 5, 6, 7, 9]] = 0
    inputs[3, 1] = threshold / 4
    inputs[4] = 0
    inputs[4, :2] = threshold / 4
    inputs[5, 2:8] = 0
    inputs[6, 2:8] = 2
    inputs[6, 8:] = 1e-4
    if spin == 0:
        inputs[:, 1] = inputs[:, 0]
        inputs[:, 5:8] = inputs[:, 2:5]
        inputs[:, 9] = inputs[:, 8]
    return inputs


def _oracle(name: str, inputs: np.ndarray, spin: int) -> np.ndarray:
    """Use pinned Libxc directly, not the imported Graph or work emitter."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0", "point oracle must be PySCF 2.14.0"
    assert libxc.__version__ == "7.0.0", "point oracle must be Libxc 7.0.0"
    density = inputs[:, :2].T
    gradient = inputs[:, 2:8].reshape(-1, 2, 3).transpose(1, 2, 0)
    tau = inputs[:, 8:].T
    family = automatic_work_policy(name).family
    rho = np.zeros((2, 6 if family == "mgga" else 4, len(inputs)))
    rho[:, 0] = density
    rho[:, 1:4] = gradient
    if family == "mgga":
        rho[:, 5] = tau
    if family == "lda":
        rho = density
    if spin == 0:
        rho = rho.sum(axis=0)
    energy, vxc, _, _ = libxc.eval_xc(name, rho, spin=spin, deriv=1)
    expected = np.zeros((len(inputs), 6 if spin == 0 else 11))
    expected[:, 0] = energy * density.sum(axis=0)
    if spin == 0:
        expected[:, 1] = vxc[0]
        if family != "lda":
            expected[:, 2:5] = (2 * vxc[1] * gradient.sum(axis=0)).T
        if family == "mgga":
            expected[:, 5] = 0.5 * vxc[3]
    else:
        expected[:, 1:3] = vxc[0]
        if family != "lda":
            expected[:, 3:6] = (
                2 * vxc[1][:, 0] * gradient[0] + vxc[1][:, 1] * gradient[1]
            ).T
            expected[:, 6:9] = (
                vxc[1][:, 1] * gradient[0] + 2 * vxc[1][:, 2] * gradient[1]
            ).T
        if family == "mgga":
            expected[:, 9:11] = 0.5 * vxc[3]
    return expected


@pytest.mark.parametrize("spin", (0, 1))
def test_point_source_matches_independent_libxc_and_directional_derivative(
    native_point: tuple[AutomaticPointSource, ctypes.CDLL, Path],
    spin: int,
) -> None:
    point, library, directory = native_point
    inputs = _inputs(point.name, spin)
    actual = _evaluate(library, inputs)
    assert np.all(actual[:, 11] == 1), (point.name, actual[:, 11])
    assert np.all(actual[[0, 4], :11] == 0)
    observed = (
        actual[:, :11]
        if spin
        else np.column_stack(
            (
                actual[:, 0],
                actual[:, 1:3].mean(axis=1),
                0.5 * (actual[:, 3:6] + actual[:, 6:9]),
                actual[:, 9:11].mean(axis=1),
            )
        )
    )
    expected = _oracle(point.name, inputs, spin)
    np.testing.assert_allclose(observed, expected, atol=2e-11, rtol=2e-9)
    interior = inputs[10:]
    directions = (
        np.random.default_rng(11212026).normal(size=interior.shape)
        * np.abs(interior)
        * 0.1
    )
    if spin == 0:
        directions[:, 1] = directions[:, 0]
        directions[:, 5:8] = directions[:, 2:5]
        directions[:, 9] = directions[:, 8]
    step = 1e-5
    finite_difference = (
        _oracle(point.name, interior + step * directions, spin)[:, 0]
        - _oracle(point.name, interior - step * directions, spin)[:, 0]
    ) / (2 * step)
    coefficients = np.column_stack((actual[10:, 1:9], 2 * actual[10:, 9:11]))
    derivative = np.sum(coefficients * directions, axis=1)
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-10, rtol=2e-7)
    (directory / f"oracle-spin-{spin}.json").write_text(
        json.dumps(
            {
                "oracle": "PySCF/Libxc-7.0.0",
                "spin": spin,
                "points": len(inputs),
                "max_absolute_error": float(np.max(np.abs(observed - expected))),
                "max_directional_error": float(
                    np.max(np.abs(derivative - finite_difference))
                ),
                "public_cuda_admitted": False,
            },
            indent=2,
        )
    )


def test_point_source_rejects_invalid_required_features(
    native_point: tuple[AutomaticPointSource, ctypes.CDLL, Path],
) -> None:
    point, library, _ = native_point
    valid = _inputs(point.name, 1)[12]
    invalid = np.repeat(valid[None, :], 8, axis=0)
    invalid[0, 0] = -1
    invalid[1, 1] = np.nan
    invalid[2, 0] = np.inf
    invalid[3, :2] = np.finfo(float).max
    invalid[4, 2] = np.nan
    invalid[5, 7] = np.inf
    invalid[6, 8] = -1
    invalid[7, 9] = np.nan
    actual = _evaluate(library, invalid)
    required_invalid = (
        4 if point.ingredient_mask == 1 else 6 if point.ingredient_mask == 7 else 8
    )
    assert np.all(actual[:required_invalid, 11] == 0)
    assert np.all(actual[required_invalid:, 11] == 1)

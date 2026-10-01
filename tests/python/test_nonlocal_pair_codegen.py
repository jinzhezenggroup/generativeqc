"""Qualify shared VV10 pair codegen, without claiming device execution.

The small frozen C++ oracles retain the pre-migration CPU and CUDA operation
orders. Both are host-compiled with contraction disabled: this establishes
FP64 scalar lowering equivalence, not CUDA compiler or GPU qualification.
"""

from __future__ import annotations

import ctypes as ct
import itertools
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools/generate_nonlocal_pair_native.py"
OUTPUTS = ("phi", "dphi_domega", "dphi_dkappa", "dphi_dr2")
MODES = tuple(
    itertools.product(("vv10", "rvv10"), (False, True), (False, True), (False, True))
)


def _generate(output: Path, *, hash_seed: int = 0) -> str:
    # Block imports explicitly, rather than merely assuming -S excludes an
    # accidentally installed runtime or an oracle living in this checkout.
    script = r"""
import importlib.abc
import runpy
import sys

class ForbidRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"numpy", "generativeqc", "pyscf", "torch", "cupy", "jax"} or "oracle" in fullname:
            raise AssertionError("standalone generation imported " + fullname)
        return None

sys.meta_path.insert(0, ForbidRuntimeImports())
script, output = sys.argv[1:]
sys.argv = [script, "--output", output]
runpy.run_path(script, run_name="__main__")
assert not any(name.split(".")[0] in {"numpy", "generativeqc", "pyscf", "torch", "cupy", "jax"}
               or "oracle" in name for name in sys.modules)
"""
    environment = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    # The generator itself must establish checkout imports; no site packages or
    # parent PYTHONPATH may accidentally make this check pass.
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-S", "-c", script, str(GENERATOR), str(output)],
        cwd=output.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return output.read_text(encoding="utf-8")


def test_nonlocal_pair_generation_is_standalone_and_deterministic(
    tmp_path: Path,
) -> None:
    from generativeqc_compiler.method.nonlocal_pair import (
        PAIR_INPUT_ORDER,
        PAIR_OUTPUT_ORDER,
        build_nonlocal_pair_program,
    )
    from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

    first = _generate(tmp_path / "first.hpp", hash_seed=1)
    second_directory = tmp_path / "other-cwd"
    second_directory.mkdir()
    second = _generate(second_directory / "second.hpp", hash_seed=81237)
    assert first == second
    assert str(ROOT) not in first
    assert str(tmp_path) not in first
    assert first.count("// TensorIR logical hash: ") == 12
    assert first.count("GENERATIVEQC_NONLOCAL_PAIR_HD inline bool pair_") == 12
    assert "#define GENERATIVEQC_NONLOCAL_PAIR_HD __host__ __device__" in first
    # The runtime owns numerical failure observation, including outputs that
    # are not consumed physically but remain part of raw CPU admission.
    assert "std::isfinite" not in first
    assert "return false" not in first
    assert "std::fma" not in first
    # Every generated specialization uses legacy root scheduling: energy,
    # omega feature, kappa feature, then radial. The generic emitter's default
    # remains canonical depth/hash order for its other consumers.
    for (variant, prepared), features, geometry in itertools.product(
        (("vv10", False), ("rvv10", False), ("rvv10", True)),
        (False, True),
        (False, True),
    ):
        program = build_nonlocal_pair_program(
            variant, preconditioned=prepared, features=features, geometry=geometry
        )
        inputs = {
            node.attrs["name"] for node in program.live_nodes if node.op == "input"
        }
        name = f"pair_{variant}_{'prepared' if prepared else 'raw'}_f{int(features)}_g{int(geometry)}"
        body = emit_scalar_cpp(
            program,
            function_name=name,
            input_order=tuple(name for name in PAIR_INPUT_ORDER if name in inputs),
            output_order=tuple(
                name for name in PAIR_OUTPUT_ORDER if name in program.outputs
            ),
            caller_owned_checks=True,
            ordered_native_sums=True,
            output_dependency_order=True,
        )
        assert (
            body.replace(
                "inline bool ", "GENERATIVEQC_NONLOCAL_PAIR_HD inline bool ", 1
            )
            in first
        )


@pytest.mark.parametrize("variant,preconditioned,features,geometry", MODES)
def test_nonlocal_pair_roots_are_live_and_demand_specific(
    variant: str, preconditioned: bool, features: bool, geometry: bool
) -> None:
    from generativeqc_compiler.method.nonlocal_pair import build_nonlocal_pair_program
    from generativeqc_compiler.tensor.program import node_hashes
    from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

    program = build_nonlocal_pair_program(
        variant, preconditioned=preconditioned, features=features, geometry=geometry
    )
    full = build_nonlocal_pair_program(variant, preconditioned=preconditioned)
    expected = {"phi"}
    if features:
        expected.update(("dphi_domega", "dphi_dkappa"))
    if geometry:
        expected.add("dphi_dr2")
    assert set(program.outputs) == expected
    assert set(program.nodes) == set(program.live_nodes)
    assert all(
        node.spec.shape == () and node.spec.dtype == "float64"
        for node in program.live_nodes
    )
    inputs = {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
    expected_inputs = {"r2", "wi", "wj", "ki", "kj"}
    if variant == "rvv10" and preconditioned and features:
        expected_inputs.add("row_inverse_kappa")
    assert inputs == expected_inputs

    # Demand selection must only prune closures, never rewrite a retained
    # output or change the ordering of its floating-point operations.
    hashes, full_hashes = node_hashes(program.live_nodes), node_hashes(full.live_nodes)
    for output in expected:
        assert hashes[program.outputs[output]] == full_hashes[full.outputs[output]]
    assert set(hashes.values()) <= set(full_hashes.values())
    if not features or not geometry:
        assert len(program.live_nodes) < len(full.live_nodes)
    source = emit_scalar_cpp(
        program,
        function_name="probe",
        caller_owned_checks=True,
        ordered_native_sums=True,
    )
    assert source.count("double& tensor_output_") == len(expected)
    assert source.count("double tensor_input_") == len(expected_inputs)
    assert ("std::pow" in source) == (variant == "rvv10" and not preconditioned)


@pytest.mark.parametrize("flag", ("preconditioned", "features", "geometry"))
@pytest.mark.parametrize("value", (None, 0, 1, "true"))
def test_nonlocal_pair_rejects_ambiguous_demand_flags(flag: str, value: object) -> None:
    from generativeqc_compiler.method.nonlocal_pair import build_nonlocal_pair_program

    with pytest.raises(TypeError, match="Boolean"):
        build_nonlocal_pair_program("vv10", **{flag: value})


def test_nonlocal_pair_rejects_unknown_variant() -> None:
    from generativeqc_compiler.common.nonlocal_correlation import (
        UnsupportedNonlocalCorrelation,
    )
    from generativeqc_compiler.method.nonlocal_pair import build_nonlocal_pair_program

    with pytest.raises(UnsupportedNonlocalCorrelation):
        build_nonlocal_pair_program("VV10")


# Frozen test-only arithmetic from vv10_runtime.cpp::pair_values and
# vv10_runtime_cuda.cu::pair_kernel_values before their migration to TensorIR.
# Keep raw CPU pow(ki*kj, 1.5) separate from CUDA's preconditioned ki*kj;
# those representations are mathematically equivalent, not bitwise equivalent.
_LEGACY_CPP = r"""
#include <cmath>
#include <cstddef>
namespace generativeqc::dft::nlc {
enum class Vv10Variant { vv10, rvv10 };
}
#include "generated_nonlocal_pair_native.hpp"
using generativeqc::dft::nlc::Vv10Variant;
namespace generated = generativeqc::dft::nlc::generated;

struct LegacyValues {
  double phi{}, dphi_domega{}, dphi_dkappa{}, dphi_dr2{};
};
LegacyValues legacy_cpu(double r2, double omega_i, double omega_j,
                        double kappa_i, double kappa_j, Vv10Variant variant) {
  LegacyValues result;
  if (variant == Vv10Variant::rvv10) {
    const auto ai = omega_i / kappa_i;
    const auto aj = omega_j / kappa_j;
    const auto zi = 1.0 + ai * r2;
    const auto zj = 1.0 + aj * r2;
    const auto kappa_product = kappa_i * kappa_j;
    const auto denominator = std::pow(kappa_product, 1.5) * zi * zj * (zi + zj);
    result.phi = -1.5 / denominator;
    const auto factor_z = 1.0 / zi + 1.0 / (zi + zj);
    result.dphi_domega = -result.phi * r2 / kappa_i * factor_z;
    result.dphi_dkappa = result.phi / kappa_i * (-1.5 + (zi - 1.0) * factor_z);
    const auto logarithmic = ai / zi + aj / zj + (ai + aj) / (zi + zj);
    result.dphi_dr2 = -result.phi * logarithmic;
  } else {
    const auto gi = omega_i * r2 + kappa_i;
    const auto gj = omega_j * r2 + kappa_j;
    result.phi = -1.5 / (gi * gj * (gi + gj));
    const auto dphi_dgi = -result.phi * (1.0 / gi + 1.0 / (gi + gj));
    result.dphi_domega = dphi_dgi * r2;
    result.dphi_dkappa = dphi_dgi;
    const auto logarithmic = omega_i / gi + omega_j / gj + (omega_i + omega_j) / (gi + gj);
    result.dphi_dr2 = -result.phi * logarithmic;
  }
  return result;
}

template <Vv10Variant Variant, bool Features, bool Geometry>
LegacyValues legacy_cuda(double r2, double wi, double wj, double ki,
                         double kj, double row_inverse_kappa) {
  LegacyValues result{};
  if constexpr (Variant == Vv10Variant::rvv10) {
    const double zi = wi * r2 + 1.0;
    const double zj = wj * r2 + 1.0;
    result.phi = -1.5 / (ki * kj * zi * zj * (zi + zj));
    if constexpr (Features) {
      const double factor_z = 1.0 / zi + 1.0 / (zi + zj);
      result.dphi_domega = -result.phi * r2 * row_inverse_kappa * factor_z;
      result.dphi_dkappa = result.phi * row_inverse_kappa * (-1.5 + (zi - 1.0) * factor_z);
    }
    if constexpr (Geometry) {
      const double logarithmic = wi / zi + wj / zj + (wi + wj) / (zi + zj);
      result.dphi_dr2 = -result.phi * logarithmic;
    }
  } else {
    const double gi = wi * r2 + ki;
    const double gj = wj * r2 + kj;
    result.phi = -1.5 / (gi * gj * (gi + gj));
    if constexpr (Features) {
      const double dphi_dgi = -result.phi * (1.0 / gi + 1.0 / (gi + gj));
      result.dphi_domega = dphi_dgi * r2;
      result.dphi_dkappa = dphi_dgi;
    }
    if constexpr (Geometry) {
      const double logarithmic = wi / gi + wj / gj + (wi + wj) / (gi + gj);
      result.dphi_dr2 = -result.phi * logarithmic;
    }
  }
  return result;
}

template <Vv10Variant Variant, bool Features, bool Geometry, bool Preconditioned>
void evaluate(const double* x, double* actual, double* expected) {
  const auto a = generated::pair_values<Variant, Features, Geometry, Preconditioned>(
      x[0], x[1], x[2], x[3], x[4], x[5]);
  LegacyValues b;
  if constexpr (Preconditioned) {
    b = legacy_cuda<Variant, Features, Geometry>(x[0], x[1], x[2], x[3], x[4], x[5]);
  } else {
    b = legacy_cpu(x[0], x[1], x[2], x[3], x[4], Variant);
    if constexpr (!Features) b.dphi_domega = b.dphi_dkappa = 0.0;
    if constexpr (!Geometry) b.dphi_dr2 = 0.0;
  }
  actual[0] = a.phi; actual[1] = a.dphi_domega;
  actual[2] = a.dphi_dkappa; actual[3] = a.dphi_dr2;
  expected[0] = b.phi; expected[1] = b.dphi_domega;
  expected[2] = b.dphi_dkappa; expected[3] = b.dphi_dr2;
}
extern "C" void pair_probe(int mode, const double* x, double* actual, double* expected) {
  switch (mode) {
"""


@pytest.fixture(scope="module")
def native_pair_probe(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("nonlocal-pair-codegen")
    _generate(directory / "generated_nonlocal_pair_native.hpp")
    branches = []
    for index, (variant, prepared, features, geometry) in enumerate(MODES):
        flags = ", ".join(str(flag).lower() for flag in (features, geometry, prepared))
        branches.append(
            f"case {index}: evaluate<Vv10Variant::{variant}, {flags}>(x, actual, expected); break;"
        )
    source, library = directory / "pair_probe.cpp", directory / "pair_probe.so"
    source.write_text(
        _LEGACY_CPP + "\n".join(branches) + "\n  }\n}\n", encoding="utf-8"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    native = ct.CDLL(str(library))
    native.pair_probe.argtypes = [ct.c_int, *(ct.POINTER(ct.c_double),) * 3]
    native.pair_probe.restype = None
    return native


def _evaluate(
    native: ct.CDLL, mode: tuple, values: tuple[float, ...]
) -> tuple[list[float], list[float]]:
    inputs = (ct.c_double * 6)(*values)
    actual, expected = (ct.c_double * 4)(), (ct.c_double * 4)()
    native.pair_probe(MODES.index(mode), inputs, actual, expected)
    return list(actual), list(expected)


def _same_fp64(actual: float, expected: float) -> bool:
    if math.isnan(expected):
        return math.isnan(actual)
    # This also distinguishes infinity signs, signed zero, and every finite ULP.
    return struct.pack("=d", actual) == struct.pack("=d", expected)


def _pair_cases() -> list[tuple[float, ...]]:
    random_source = random.Random(0x56563130)
    cases = [
        tuple(10.0 ** random_source.uniform(-3.0, 3.0) for _ in range(6))
        for _ in range(256)
    ]
    # Broad independently varying exponents exercise intermediate overflow and
    # underflow; negative inputs deliberately probe failure-publication behavior.
    for _ in range(512):
        cases.append(
            tuple(
                math.ldexp(
                    random_source.uniform(0.5, 1.0), random_source.randint(-1073, 1023)
                )
                * random_source.choice((-1.0, 1.0))
                for _ in range(6)
            )
        )
    edges = (
        0.0,
        -0.0,
        math.ulp(0.0),
        -math.ulp(0.0),
        sys.float_info.min,
        -sys.float_info.min,
        1e-300,
        -1e-300,
        1e-150,
        1.0,
        -1.0,
        1e150,
        1e300,
        sys.float_info.max,
        -sys.float_info.max,
        math.inf,
        -math.inf,
        math.nan,
    )
    baseline = (0.75, 0.9, 1.1, 1.2, 1.4, 1.0 / 1.2)
    for index in range(6):
        for value in edges:
            sample = list(baseline)
            sample[index] = value
            cases.append(tuple(sample))
    cases.extend(tuple(value for _ in range(6)) for value in edges)
    # gi/gj cancellation, opposite signed zeros, and radial roots that become
    # nonfinite although energy is finite must retain legacy classifications.
    cases.extend(
        [
            (1.0, 1.0, 1.0, -1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
            (-0.0, 1.0, 1.0, -0.0, -0.0, 0.0),
            (0.0, -0.0, -0.0, -0.0, -0.0, -0.0),
            (0.0, sys.float_info.max, sys.float_info.max, 1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 1e-200, 1e200, 1e200),
            (1.0, 1.0, 1.0, 1e200, 1e200, 1e-200),
            (math.ulp(0.0), 1e300, 1e300, 1.0, 1.0, 1.0),
        ]
    )
    # Exhaust signed-zero combinations independently of the other edge cases.
    cases.extend(itertools.product((0.0, -0.0), repeat=6))
    return cases


@pytest.mark.parametrize("variant,preconditioned,features,geometry", MODES)
def test_generated_pair_matches_legacy_fp64_and_exception_classification(
    native_pair_probe: ct.CDLL,
    variant: str,
    preconditioned: bool,
    features: bool,
    geometry: bool,
) -> None:
    mode = (variant, preconditioned, features, geometry)
    for values in _pair_cases():
        actual, expected = _evaluate(native_pair_probe, mode, values)
        for name, a, b in zip(OUTPUTS, actual, expected, strict=True):
            assert _same_fp64(a, b), (mode, name, values, a.hex(), b.hex())


def _reference_phi(variant: str, values: list[Decimal]) -> Decimal:
    r2, wi, wj, ki, kj = values
    if variant == "rvv10":
        zi, zj = 1 + wi * r2 / ki, 1 + wj * r2 / kj
        product = ki * kj
        denominator = product * product.sqrt() * zi * zj * (zi + zj)
    else:
        gi, gj = wi * r2 + ki, wj * r2 + kj
        denominator = gi * gj * (gi + gj)
    return Decimal("-1.5") / denominator


@pytest.mark.parametrize(
    "variant,preconditioned", tuple(itertools.product(("vv10", "rvv10"), (False, True)))
)
@pytest.mark.parametrize("r2", (0.0, 0.2, 1.0, 9.0))
def test_generated_pair_matches_independent_high_precision_finite_differences(
    native_pair_probe: ct.CDLL, variant: str, preconditioned: bool, r2: float
) -> None:
    raw = (r2, 0.7, 1.3, 1.9, 2.3)
    values = (*raw, 1.0 / raw[3])
    if variant == "rvv10" and preconditioned:
        values = (
            r2,
            raw[1] / raw[3],
            raw[2] / raw[4],
            raw[3] * math.sqrt(raw[3]),
            raw[4] * math.sqrt(raw[4]),
            1.0 / raw[3],
        )
    actual, _ = _evaluate(
        native_pair_probe, (variant, preconditioned, True, True), values
    )
    # Different arithmetic precision and numerical differentiation provide an
    # oracle independent of both the TensorIR derivatives and frozen native code.
    with localcontext() as context:
        context.prec = 80
        precise = [Decimal.from_float(value) for value in raw]
        expected = [float(_reference_phi(variant, precise))]
        for index in (1, 3, 0):
            step = Decimal("1e-25")
            plus, minus = precise.copy(), precise.copy()
            plus[index] += step
            minus[index] -= step
            expected.append(
                float(
                    (_reference_phi(variant, plus) - _reference_phi(variant, minus))
                    / (2 * step)
                )
            )
    assert actual == pytest.approx(expected, rel=8e-14, abs=1e-18)


def test_pair_output_demands_do_not_change_observed_failure_domain(
    native_pair_probe: ct.CDLL,
) -> None:
    # At zero separation omega_i+omega_j can overflow in the radial derivative
    # while energy and feature outputs remain finite. Raw CPU's all-output
    # checking must keep seeing the failure; demand-specific CUDA can omit it.
    values = (0.0, sys.float_info.max, sys.float_info.max, 1.0, 1.0, 1.0)
    full, _ = _evaluate(native_pair_probe, ("vv10", False, True, True), values)
    energy, _ = _evaluate(native_pair_probe, ("vv10", False, False, False), values)
    assert math.isfinite(full[0])
    assert not math.isfinite(full[3])
    assert _same_fp64(energy[0], full[0])
    assert energy[1:] == [0.0, 0.0, 0.0]


def test_cpu_and_cuda_pair_consumers_include_shared_generated_header() -> None:
    generated = _generate(Path(pytest.ensuretemp("nonlocal-local-scale")) / "generated.hpp")
    assert "local_scales_cpu(" in generated
    assert "local_scales_cuda(" in generated
    for filename in ("vv10_runtime.cpp", "vv10_runtime_cuda.cu"):
        source = (ROOT / "src/dft/nonlocal_correlation" / filename).read_text(
            encoding="utf-8"
        )
        assert '#include "generated_nonlocal_pair_native.hpp"' in source
        assert re.search(r"generated::pair_values\s*<", source)
    cpu = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime.cpp").read_text()
    cuda = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    assert "generated::local_scales_cpu<" in cpu
    assert "generated::local_scales_cuda<" in cuda
    assert "const auto omega2 = c * sigma2 / rho4" not in cpu
    assert "const double ratio = sigma / (rho * rho)" not in cuda

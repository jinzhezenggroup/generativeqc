"""Compile typed AD arithmetic, including casts before products and overflow."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.precision import weighted_force_precision_schedule
from generativeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from generativeqc_compiler.integral.weighted_eri_cuda import (
    emit_weighted_eri_function,
    emit_weighted_eri_header,
)

from tools.generativeqc_validation.weighted_eri import primitive_variables

CLASSES = ((0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 1, 0), (2, 1, 1, 0), (2, 2, 1, 0))


class Geometry(ctypes.Structure):
    """Stable compiler geometry transport; every input deliberately remains double."""

    _fields_ = [
        ("inverse_two_p", ctypes.c_double),
        ("inverse_two_q", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("difference", ctypes.c_double * 3),
        ("shifts", (ctypes.c_double * 3) * 4),
        ("product_scales", ctypes.c_double * 4),
        ("decay", (ctypes.c_double * 3) * 4),
        ("prefactor", ctypes.c_double),
        ("boys", ctypes.c_double * 14),
    ]


def geometry(exponents: np.ndarray, centers: np.ndarray) -> Geometry:
    """Use independent Gauss-Legendre Boys moments, not the native evaluator."""
    variables = primitive_variables(exponents, centers, 9)
    result = Geometry()
    for name in ("inverse_two_p", "inverse_two_q", "rho", "prefactor"):
        setattr(result, name, variables[name])
    for axis, label in enumerate("xyz"):
        result.difference[axis] = variables[f"difference_{label}"]
        for center, prefix in enumerate(("pa", "pb", "qc", "qd")):
            result.shifts[center][axis] = variables[f"{prefix}_{label}"]
        for center, prefix in enumerate(("first", "second", "third", "fourth")):
            result.decay[center][axis] = variables[f"decay_{prefix}_{label}"]
            result.product_scales[center] = variables[f"{prefix}_product_scale"]
    for order in range(10):
        result.boys[order] = variables[f"boys_{order}"]
    return result


@pytest.fixture(scope="module")
def compiled(native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Exercise emitted C++ on the host; no native library or device is needed."""
    folder = tmp_path_factory.mktemp("weighted-force-precision")
    kernels = [
        build_weighted_eri_kernel(build_weighted_eri_ir(angular), (0,))
        for angular in CLASSES
    ]
    source = emit_weighted_eri_header(
        tuple((kernel, f"strict_{index}") for index, kernel in enumerate(kernels)),
        backend="cpu",
    )
    source += "\nnamespace generativeqc::scf::generated_weighted_eri {\n"
    for index, kernel in enumerate(kernels):
        source += emit_weighted_eri_function(
            kernel, f"mixed_{index}", backend="cpu", scalar_type="float"
        )
    source += "}\nusing namespace generativeqc::scf::generated_weighted_eri;\n"
    for index in range(len(kernels)):
        source += f"""
extern "C" void probe_{index}(const Geometry* input, const double* weights, double* output, bool mixed) {{
  auto result = mixed ? mixed_{index}(*input, weights) : strict_{index}(*input, weights);
  output[0] = result.value;
  for (unsigned center = 0; center < 4; ++center)
    for (unsigned axis = 0; axis < 3; ++axis)
      output[1 + 3 * center + axis] = result.center[center][axis];
}}
"""
    path = folder / "probe.cpp"
    path.write_text(source)
    library = folder / "probe.so"
    native_cxx.build_shared(
        [path], library, compile_args=("-std=c++20", "-O2", "-ffp-contract=off")
    )
    loaded = ctypes.CDLL(str(library))
    functions = []
    for index in range(len(kernels)):
        function = getattr(loaded, f"probe_{index}")
        function.argtypes = (
            ctypes.POINTER(Geometry),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ctypes.c_bool,
        )
        function.restype = None
        functions.append(function)
    return functions


@pytest.mark.parametrize("index", range(len(CLASSES)))
@pytest.mark.parametrize("scale", (0.1, 1.0, 10.0))
def test_typed_derivatives_match_strict_and_are_really_float(
    compiled: Any, index: int, scale: float
) -> None:
    generator = np.random.default_rng(451)
    centers = generator.normal(size=(4, 3)) / np.sqrt(scale)
    inputs = geometry(np.array([0.6, 0.8, 1.1, 0.9]) * scale, centers)
    weights = np.array([0.73123456789])
    strict, mixed = np.empty(13), np.empty(13)
    compiled[index](ctypes.byref(inputs), weights, strict, False)
    compiled[index](ctypes.byref(inputs), weights, mixed, True)
    np.testing.assert_allclose(mixed, strict, rtol=3e-5, atol=2e-6 * max(abs(strict)))
    np.testing.assert_array_equal(mixed, mixed.astype(np.float32).astype(np.float64))
    assert np.any(mixed != strict)


def test_overflow_is_visible_to_native_retry(compiled: Any) -> None:
    inputs = geometry(np.ones(4), np.arange(12).reshape(4, 3) * 0.02)
    weights = np.array([1e40])
    strict, mixed = np.empty(13), np.empty(13)
    compiled[0](ctypes.byref(inputs), weights, strict, False)
    compiled[0](ctypes.byref(inputs), weights, mixed, True)
    assert np.isfinite(strict).all()
    assert not np.isfinite(mixed).all()


def test_force_precision_contract_requires_explicit_qualification() -> None:
    assert weighted_force_precision_schedule().is_strict_fp64
    with pytest.raises(ValueError, match="qualification"):
        weighted_force_precision_schedule(mixed_recurrence=True)
    mixed = weighted_force_precision_schedule(
        mixed_recurrence=True, qualification="test-only"
    )
    regions = dict(mixed.regions)
    assert regions["weighted_derivative"].compute_dtype == "float32"
    assert regions["primitive_reduction"].accumulation_dtype == "float32"
    assert regions["translation_atom_accumulation"].accumulation_dtype == "float64"

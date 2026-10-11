"""Registry-wide optimization coverage, independent of functional spelling."""

from __future__ import annotations

import ctypes
import inspect
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import _stationary_cuda as runtime
from generativeqc_compiler.dft import ao_cuda
from generativeqc_compiler.dft._generated_native_semilocal import SEMILOCAL_FAMILIES
from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS
from generativeqc_compiler.xc.geometry_cuda import (
    _emit_restricted_point_capability,
    _emit_stationary_point,
)
from generativeqc_compiler.xc.semilocal_codegen import emit_polarized_semilocal
from generativeqc_compiler.xc.spec import FunctionalSpec, functional

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "gradient,tau,jets",
    ((False, False, 1), (True, False, 4), (False, True, 4), (True, True, 4)),
)
def test_stationary_ao_derivatives_follow_ingredients_not_codes(
    monkeypatch: pytest.MonkeyPatch, gradient: bool, tau: bool, jets: int
) -> None:
    """Opaque rho-only and tau-only consumers must not inherit a code-range policy."""
    record = {"requires_gradient": gradient, "requires_tau": tau}
    monkeypatch.setattr(runtime, "SEMILOCAL_FAMILY_BY_CODE", {917: record})
    assert runtime._stationary_density_jet_count(917) == jets
    assert runtime._stationary_density_jet_count(0) == 4
    source = inspect.getsource(runtime.PreparedStationaryCudaExecution.ensure)
    assert "needs_first = _stationary_density_jet_count(functional) > 1" in source
    assert "needs_first = functional !=" not in source


STATIONARY_CONSUMERS = (
    tuple(
        (
            record["code"],
            FunctionalSpec(
                "opaque-composition",
                tuple(
                    (name, Fraction(weight)) for name, weight in record["components"]
                ),
                spin="unpolarized",
                range_omega=Fraction(record["range_omega"]),
            ),
        )
        for record in SEMILOCAL_FAMILIES
        if record["cuda_ks"]
    )
    + tuple(
        (
            record["functional_code"],
            FunctionalSpec(
                "opaque-composition",
                tuple(
                    (name, Fraction(weight)) for name, weight in record["components"]
                ),
                spin="unpolarized",
            ),
        )
        for record in SPLIT_HYBRIDS.values()
    )
    + tuple(
        (
            1,
            FunctionalSpec(
                "opaque-scaled-gga",
                (("GGA_X_PBE", weight), ("GGA_C_PBE", Fraction(1))),
                spin="unpolarized",
            ),
        )
        for weight in (Fraction(3, 4), Fraction(1, 2))
    )
)


@pytest.mark.parametrize("component", ("LDA_X", "GGA_X_PBE", "MGGA_X_SCAN"))
def test_cuda_scalar_lowering_automatically_emits_equal_spin(component: str) -> None:
    """A new scalar consumer needs no named generator opt-in or whitelist."""
    spec = functional(component, spin="polarized")
    emitted = emit_polarized_semilocal(
        spec,
        value_type="OpaqueValue",
        function_name="opaque_point",
        identity_constant="kOpaqueIdentity",
        function_qualifier="__device__ inline",
    )
    assert "opaque_point_equal_spin(" in emitted
    assert "kOpaqueIdentityEqualSpin" in emitted


def test_new_registered_shapes_automatically_join_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission and ingredient traits, not names/codes, select every variant."""
    records = []
    for index, width in enumerate((1, 4, 5)):
        record = dict(SEMILOCAL_FAMILIES[0])
        record.update(
            code=900 + index,
            symbol=f"Opaque{width}",
            name=f"unknown-{width}",
            requires_gradient=width >= 4,
            requires_tau=width == 5,
        )
        records.append(record)
    excluded = {
        **records[0],
        "code": 999,
        "symbol": "Excluded",
        "cuda_ks": False,
        "cuda_fast_paths": {**records[0]["cuda_fast_paths"], "response": "unavailable"},
    }
    monkeypatch.setattr(ao_cuda, "SEMILOCAL_FAMILIES", (*records, excluded))
    monkeypatch.setattr(
        ao_cuda,
        "SEMILOCAL_FAMILY_BY_CODE",
        {record["code"]: record for record in records},
    )
    source = ao_cuda.emit_native_xc_contraction_kernels()
    for width in (1, 4, 5):
        assert f"case semilocal_family_code(SemilocalFamily::Opaque{width}):" in source
        threads = 32 if width == 4 else 128
        assert f"return &launch_point_batches<{width}, {threads}>;" in source
    assert "SemilocalFamily::Excluded" not in source
    assert "functional <=" not in source
    assert "functional > static_cast<I>(SemilocalFamily::Pbe)" not in source


@pytest.fixture(scope="module")
def stationary_points(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> ctypes.CDLL:
    """Compile every actual stationary CUDA point body as strict host C++."""
    directory = tmp_path_factory.mktemp("all-functional-stationary-points")
    lines = [
        "#define __device__",
        "#include <cmath>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <stdexcept>",
        '#include "dft/xc_point.hpp"',
        "using std::isfinite;",
    ]
    for index, (code, spec) in enumerate(STATIONARY_CONSUMERS):
        lines.extend(
            (
                f"namespace consumer_{index} {{",
                _emit_stationary_point(code, semilocal=spec),
                _emit_restricted_point_capability(code, spec),
                "}",
            )
        )
    lines.extend(
        (
            "template <class Value> int store(Value value, double* output) {",
            "  output[0] = value.energy;",
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            "    output[1 + spin] = value.rho[spin];",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      output[3 + 3 * spin + axis] = value.gradient[spin][axis];",
            "    output[9 + spin] = value.kinetic[spin];",
            "  }",
            "  return value.valid;",
            "}",
            'extern "C" int evaluate(unsigned consumer, bool restricted, const double* input, double* output) {',
            "  double rho[2]{input[0],input[1]}, gradient[2][3], tau[2]{input[8],input[9]};",
            "  for (unsigned spin = 0; spin < 2; ++spin)",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      gradient[spin][axis] = input[2 + 3 * spin + axis];",
            "  switch (consumer) {",
        )
    )
    for index in range(len(STATIONARY_CONSUMERS)):
        lines.append(
            f"    case {index}: return store(restricted ? consumer_{index}::stationary_evaluate_restricted_point(rho, gradient, tau) : consumer_{index}::stationary_evaluate_point(rho, gradient, tau), output);"
        )
    lines.extend(("    default: return 0;", "  }", "}"))
    lines.extend(
        (
            'extern "C" int selected(unsigned consumer, bool requested) {',
            "  try {",
            "    switch (consumer) {",
            *(
                f"      case {index}: return consumer_{index}::stationary_restricted_point_selected(requested);"
                for index in range(len(STATIONARY_CONSUMERS))
            ),
            "      default: return 0;",
            "    }",
            "  } catch (const std::invalid_argument&) { return 2; }",
            "}",
        )
    )
    source = directory / "points.cpp"
    source.write_text("\n".join(lines), encoding="utf-8")
    library = ctypes.CDLL(
        str(
            native_cxx.build_shared(
                [source],
                directory / "points.so",
                compile_args=(
                    "-std=c++17",
                    "-O2",
                    "-ffp-contract=off",
                    f"-I{ROOT / 'src'}",
                ),
            )
        )
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = (ctypes.c_uint, ctypes.c_bool, pointer, pointer)
    library.evaluate.restype = ctypes.c_int
    library.selected.argtypes = (ctypes.c_uint, ctypes.c_bool)
    library.selected.restype = ctypes.c_int
    return library


@pytest.mark.parametrize("consumer", range(len(STATIONARY_CONSUMERS)))
def test_stationary_default_on_with_explicit_opt_out(
    stationary_points: ctypes.CDLL,
    consumer: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the generated selector, including invalid and disabled requests."""
    variable = "GENERATIVEQC_CUDA_XC_EQUAL_SPIN"
    monkeypatch.delenv(variable, raising=False)
    assert stationary_points.selected(consumer, True) == 1
    assert stationary_points.selected(consumer, False) == 0
    monkeypatch.setenv(variable, "0")
    assert stationary_points.selected(consumer, True) == 0
    monkeypatch.setenv(variable, "1")
    assert stationary_points.selected(consumer, True) == 1
    assert stationary_points.selected(consumer, False) == 0
    monkeypatch.setenv(variable, "invalid")
    assert stationary_points.selected(consumer, True) == 2
    assert stationary_points.selected(consumer, False) == 0


def _physical_points() -> np.ndarray:
    """Equal raw spin inputs, with unconstrained Cartesian gradient directions."""
    rng = np.random.default_rng(20261011)
    density = 10.0 ** rng.uniform(-2, 0, 16)
    gradient = density[:, None] ** (4 / 3) * rng.uniform(-0.2, 0.2, (16, 3))
    kinetic = density ** (5 / 3) * rng.uniform(0.5, 2.0, 16)
    return np.column_stack((density, density, gradient, gradient, kinetic, kinetic))


def _evaluate(
    library: ctypes.CDLL, consumer: int, restricted: bool, inputs: np.ndarray
) -> np.ndarray:
    values = np.empty((len(inputs), 11))
    pointer = ctypes.POINTER(ctypes.c_double)
    for input_row, output_row in zip(inputs, values, strict=True):
        assert library.evaluate(
            consumer,
            restricted,
            input_row.ctypes.data_as(pointer),
            output_row.ctypes.data_as(pointer),
        )
    assert np.all(np.isfinite(values))
    return values


@pytest.mark.parametrize("consumer", range(len(STATIONARY_CONSUMERS)))
def test_stationary_equal_spin_preserves_all_registered_partial_outputs(
    stationary_points: ctypes.CDLL,
    consumer: int,
) -> None:
    inputs = _physical_points()
    bound = _evaluate(stationary_points, consumer, True, inputs)
    general = _evaluate(stationary_points, consumer, False, inputs)
    np.testing.assert_allclose(bound, general, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("consumer", range(len(STATIONARY_CONSUMERS)))
def test_stationary_equal_spin_matches_independent_libxc(
    stationary_points: ctypes.CDLL,
    consumer: int,
) -> None:
    pytest.importorskip("pyscf")
    from pyscf.dft import libxc

    _, spec = STATIONARY_CONSUMERS[consumer]
    inputs = _physical_points()
    density = np.zeros((2, 6, len(inputs)))
    density[:, 0] = inputs[:, 0]
    density[:, 1:4] = inputs[:, 2:5].T
    density[:, 5] = inputs[:, 8]
    code = "+".join(f"{float(weight)}*{name}" for name, weight in spec.components)
    if any(name not in libxc.XC_CODES for name, _ in spec.components):
        record = next(
            record
            for record in SEMILOCAL_FAMILIES
            if dict(spec.components)
            == {name: Fraction(weight) for name, weight in record["components"]}
            and spec.range_omega == Fraction(record["range_omega"])
        )
        code = record["name"].replace("-", "_")
    if spec.ingredients == ("rho",):
        density = density[:, 0]
    elif spec.ingredients == ("rho", "sigma"):
        density = density[:, :4]
    exc, derivatives, _, _ = libxc.eval_xc(
        code, density, spin=1, deriv=1, omega=float(spec.range_omega)
    )
    reference = np.zeros((len(inputs), 11))
    reference[:, 0] = 2 * inputs[:, 0] * exc
    reference[:, 1:3] = derivatives[0]
    if "sigma" in spec.ingredients:
        for spin in range(2):
            reference[:, 3 + 3 * spin : 6 + 3 * spin] = (
                2 * derivatives[1][:, 2 * spin] + derivatives[1][:, 1]
            )[:, None] * inputs[:, 2:5]
    if "tau" in spec.ingredients:
        reference[:, 9:11] = 0.5 * derivatives[3]
    candidate = _evaluate(stationary_points, consumer, True, inputs)
    np.testing.assert_allclose(candidate[:, 0], reference[:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        candidate[:, 1:], reference[:, 1:], rtol=2e-10, atol=2e-10
    )


@pytest.mark.parametrize(("code", "spec"), STATIONARY_CONSUMERS)
def test_stationary_capability_depends_on_spin_not_a_named_method(
    code: int, spec: FunctionalSpec
) -> None:
    emitted = _emit_restricted_point_capability(code, spec)
    assert "stationary_restricted_point_capable = true" in emitted
    assert emitted == _emit_restricted_point_capability(
        code, replace(spec, identifier="renamed")
    )
    polarized = _emit_restricted_point_capability(code, replace(spec, spin="polarized"))
    assert "stationary_restricted_point_capable = false" in polarized

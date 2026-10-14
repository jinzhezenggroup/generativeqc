"""Whole-model budget selection and consumer-owned fitted precision evidence."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import GridSpec, TargetErrorBudget, _native
from generativeqc._df_precision import (
    fitted_gradient_admission,
    fitted_gradient_precision,
)
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc_compiler.method import ExactExchangePrimitive, resolve_method


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject only live provider proof; real state/lease gates have CUDA coverage."""
    calculator = SimpleNamespace(
        _application_precision_budget=TargetErrorBudget(1e-5, 1e-4, 3e-5),
        _precision_mode=_native.PRECISION_FP64,
        _density_fitting_mode=_native.DENSITY_FITTING_CUDA,
        _dispersion_method_ir=None,
        _energy_tolerance=1e-10,
        _density_tolerance=1e-8,
        _screening_tolerance=1e-12,
    )
    source = object.__new__(NativeKsSnapshot)
    source._handle = 0
    source.backend = "cuda"
    source.hamiltonian = "all-electron"
    source.grid_spec = GridSpec(24, 8, 16)
    source.method_ir = resolve_method("PBE0")
    source.coefficients = (0.75, 1.0, 0.25)
    provider = SimpleNamespace(value=("density-fitted", "density-fitted", 1e-10))

    def proof(owner: Any) -> tuple:
        if isinstance(provider.value, Exception):
            raise provider.value
        return provider.value

    monkeypatch.setattr(NativeKsSnapshot, "fock_provider_proof", proof)
    state = SimpleNamespace(
        _source=source, identity=SimpleNamespace(spin="unpolarized")
    )
    basis = SimpleNamespace(
        nao=24, natom=3, shells=(SimpleNamespace(angular_momentum=2),)
    )
    return calculator, state, basis, provider


@pytest.mark.parametrize("name", ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X"))
def test_complete_fitted_models_and_scientific_changes(model: Any, name: str) -> None:
    calculator, state, basis, provider = model
    source = state._source
    method = resolve_method(name)
    object.__setattr__(source, "method_ir", method)
    if name in ("PBE", "R2SCAN"):
        object.__setattr__(source, "coefficients", (1.0, 1.0, 0.0))
        provider.value = ("density-fitted", None, 1e-10)
    expected = f"{name.lower()}-df-rks-budget-fp32-v1"
    assert fitted_gradient_admission(calculator, state, basis) == expected
    object.__setattr__(source, "grid_spec", GridSpec(48, 16, 32))
    assert fitted_gradient_admission(calculator, state, basis) == expected
    primitives = tuple(
        primitive
        for primitive in method.primitives
        if not isinstance(primitive, ExactExchangePrimitive)
    )
    altered = replace(
        method, primitives=(*primitives, ExactExchangePrimitive(Fraction(1, 3)))
    )
    object.__setattr__(source, "method_ir", altered)
    assert fitted_gradient_admission(calculator, state, basis) is None


@pytest.mark.parametrize(
    "owner,name,value",
    (
        ("calculator", "_application_precision_budget", TargetErrorBudget(1e-7, 1e-4)),
        ("calculator", "_application_precision_budget", TargetErrorBudget(1e-5, 1e-6)),
        (
            "calculator",
            "_application_precision_budget",
            TargetErrorBudget(1e-5, 1e-4, 1e-7),
        ),
        ("calculator", "_precision_mode", _native.PRECISION_AUTO),
        ("calculator", "_density_fitting_mode", _native.DENSITY_FITTING_NONE),
        ("calculator", "_dispersion_method_ir", resolve_method("R2SCAN-3c")),
        ("calculator", "_density_tolerance", 1e-5),
        ("source", "backend", "cpu"),
        ("source", "hamiltonian", "scalar-semilocal-ecp"),
        ("source", "grid_spec", GridSpec(16, 8, 16)),
        ("source", "method_ir", resolve_method("WB97M-V")),
        ("source", "method_ir", resolve_method("PBE0", spin="polarized")),
        ("source", "method_ir", resolve_method("R2SCAN-3c")),
        ("basis", "nao", 97),
        ("basis", "natom", 13),
        ("basis", "shells", (SimpleNamespace(angular_momentum=3),)),
        ("provider", "value", ("exact", "density-fitted", 1e-10)),
        ("provider", "value", ("density-fitted", "exact", 1e-10)),
        ("provider", "value", ("density-fitted", "seminumerical-cosx", 1e-10)),
        ("provider", "value", ("density-fitted", "density-fitted", 1e-8)),
        ("provider", "value", NotImplementedError("old provider ABI")),
    ),
)
def test_unqualified_models_keep_strict_execution(
    model: Any, owner: str, name: str, value: Any
) -> None:
    calculator, state, basis, provider = model
    selected = {
        "calculator": calculator,
        "source": state._source,
        "basis": basis,
        "provider": provider,
    }
    object.__setattr__(selected[owner], name, value)
    assert fitted_gradient_admission(calculator, state, basis) is None


def test_stale_provider_is_not_optional_capability(model: Any) -> None:
    calculator, state, basis, provider = model
    provider.value = ValueError("stale snapshot")
    with pytest.raises(ValueError, match="stale"):
        fitted_gradient_admission(calculator, state, basis)


def test_canonical_3c_requires_both_electronic_and_correction_models(
    model: Any,
) -> None:
    """The constructor binds its basis; corrected admission checks both graphs."""
    calculator, state, basis, provider = model
    calculator._dispersion_method_ir = resolve_method("R2SCAN-3c")
    assert fitted_gradient_admission(calculator, state, basis) is None
    object.__setattr__(state._source, "method_ir", resolve_method("R2SCAN"))
    object.__setattr__(state._source, "coefficients", (1.0, 1.0, 0.0))
    provider.value = ("density-fitted", None, 1e-10)
    assert (
        fitted_gradient_admission(calculator, state, basis)
        == "r2scan-df-rks-budget-fp32-v1"
    )
    calculator._dispersion_method_ir = resolve_method("PBE0")
    assert fitted_gradient_admission(calculator, state, basis) is None


@pytest.mark.parametrize("retry", (False, True))
def test_precision_record_owns_actual_work_not_scf_inventory(retry: bool) -> None:
    result = SimpleNamespace(
        work={
            "fp32_geometry_points": 0 if retry else 120,
            "fp32_geometry_batches": 0 if retry else 2,
            "rejected_semilocal_work": {
                "fp32_geometry_points": 120,
                "fp32_geometry_batches": 2,
            }
            if retry
            else {},
            "fp64_geometry_retries": int(retry),
            "xc_points": 120,
            "grid_density_source": "exact-final-scf-device-binding",
            "native_integral_resources": {"density_fitted_provider": 1},
        },
        state_identity=SimpleNamespace(to_payload=lambda: {"generation": 12}),
        plan_identity="plan",
    )
    report = fitted_gradient_precision("pbe0-df-rks-budget-fp32-v1", result)
    assert report["attempted_fp32_geometry_points"] == 120
    assert report["attempted_fp32_grid_points"] == 0
    assert report["returned_mode"] == ("fp64" if retry else "mixed")
    assert report["attempted_fp32_geometry_batches"] == 2
    assert "operator_inventory_complete" not in report
    result.work["grid_density_source"] = "stale-density"
    with pytest.raises(RuntimeError, match="complete producer work"):
        fitted_gradient_precision("pbe0-df-rks-budget-fp32-v1", result)

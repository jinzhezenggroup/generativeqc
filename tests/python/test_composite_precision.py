"""Require the complete RSH/VV10 model and execution-owned mixed provenance."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import GridSpec, TargetErrorBudget, _native
from generativeqc._composite_precision import (
    COMPOSITE_DF_GRADIENT_QUALIFICATION,
    COMPOSITE_GRADIENT_QUALIFICATION,
    composite_gradient_admission,
    composite_gradient_precision,
)
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_POLICY
from generativeqc_compiler.method import RangeSeparatedExchangePrimitive, resolve_method


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Use the real type and canonical MethodIR, injecting only the provider port."""
    calculator = SimpleNamespace(
        _application_precision_budget=TargetErrorBudget(1e-5, 1e-4, 3e-5),
        _precision_mode=_native.PRECISION_FP64,
        _density_fitting_mode=_native.DENSITY_FITTING_NONE,
        _dispersion_method_ir=None,
        _energy_tolerance=1e-10,
        _density_tolerance=1e-8,
        _screening_tolerance=1e-12,
    )
    source = object.__new__(NativeKsSnapshot)
    source._handle = 0
    source.backend = "cuda"
    source.hamiltonian = "all-electron"
    source.grid_spec = GridSpec(48, 16, 32)
    source.nonlocal_density_policy = MOLECULAR_VV10_DENSITY_POLICY
    source.method_ir = resolve_method("WB97M-V")
    provider = SimpleNamespace(value=("exact", "exact", 0.0))

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


def test_complete_range_composition_not_point_code(model: Any) -> None:
    calculator, state, basis, _ = model
    assert (
        composite_gradient_admission(calculator, state, basis)
        == COMPOSITE_GRADIENT_QUALIFICATION
    )
    method = state._source.method_ir
    changed = replace(
        method,
        primitives=tuple(
            replace(primitive, omega=Fraction(1, 4))
            if isinstance(primitive, RangeSeparatedExchangePrimitive)
            else primitive
            for primitive in method.primitives
        ),
    )
    object.__setattr__(state._source, "method_ir", changed)
    assert composite_gradient_admission(calculator, state, basis) is None


@pytest.mark.parametrize(
    "owner,name,value",
    (
        ("calculator", "_application_precision_budget", None),
        ("calculator", "_application_precision_budget", TargetErrorBudget(1e-7, 1e-4)),
        ("calculator", "_application_precision_budget", TargetErrorBudget(1e-5, 1e-6)),
        (
            "calculator",
            "_application_precision_budget",
            TargetErrorBudget(1e-5, 1e-4, 1e-7),
        ),
        ("calculator", "_precision_mode", _native.PRECISION_AUTO),
        ("calculator", "_density_fitting_mode", _native.DENSITY_FITTING_CUDA),
        ("calculator", "_dispersion_method_ir", resolve_method("R2SCAN-3c")),
        ("calculator", "_energy_tolerance", 1e-8),
        ("calculator", "_density_tolerance", 1e-5),
        ("calculator", "_screening_tolerance", 1e-9),
        ("source", "backend", "cpu"),
        ("source", "hamiltonian", "scalar-semilocal-ecp"),
        ("source", "grid_spec", GridSpec(16, 6, 12)),
        ("source", "nonlocal_density_policy", "modified-cutoff"),
        ("source", "method_ir", resolve_method("PBE0")),
        ("source", "method_ir", resolve_method("WB97M-V", spin="polarized")),
        ("basis", "nao", 97),
        ("basis", "natom", 13),
        ("basis", "shells", (SimpleNamespace(angular_momentum=3),)),
        ("provider", "value", ("density-fitted", "exact", 1e-10)),
        ("provider", "value", ("exact", "seminumerical-cosx", 0.0)),
        ("provider", "value", NotImplementedError("optional proof absent")),
    ),
)
def test_unsupported_domains_keep_strict(
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
    assert composite_gradient_admission(calculator, state, basis) is None


def test_stale_provider_is_not_a_capability_miss(model: Any) -> None:
    calculator, state, basis, provider = model
    provider.value = ValueError("stale state")
    with pytest.raises(ValueError, match="stale"):
        composite_gradient_admission(calculator, state, basis)


def test_fitted_full_range_provider_requires_matching_precision_record(
    model: Any,
) -> None:
    """A provider/mode mismatch cannot turn direct evidence into fitted evidence."""
    calculator, state, basis, provider = model
    calculator._density_fitting_mode = _native.DENSITY_FITTING_CUDA
    provider.value = ("density-fitted", "density-fitted", 1e-10)
    assert (
        composite_gradient_admission(calculator, state, basis)
        == COMPOSITE_DF_GRADIENT_QUALIFICATION
    )
    for invalid in (
        ("density-fitted", "exact", 1e-10),
        ("density-fitted", "density-fitted", 1e-8),
    ):
        provider.value = invalid
        assert composite_gradient_admission(calculator, state, basis) is None


@pytest.mark.parametrize("retry", (False, True))
@pytest.mark.parametrize("fitted", (False, True))
def test_separate_semilocal_nonlocal_and_rejected_work(
    retry: bool, fitted: bool
) -> None:
    identity = SimpleNamespace(to_payload=lambda: {"generation": 12})
    semilocal = {
        "fp32_geometry_points": 0 if retry else 120,
        "fp32_geometry_batches": 0 if retry else 2,
    }
    work = {
        "execution": "cuda-complete-composite",
        "plan_identity": "authenticated-plan",
        "grid_density_source": "exact-final-scf-device-binding",
        "grid_points": 120,
        "geometry_point_visits": 240,
        "nonlocal_execution": "resident-full-grid-device-seeds",
        "native_integral_resources": {"mixed_rsh_df_provider": int(fitted)},
        "stationary_source_work": {
            "semilocal": semilocal,
            "nonlocal": {"fp32_geometry_points": 0},
        },
        "fp64_geometry_retries": int(retry),
        "rejected_semilocal_work": {
            "fp32_geometry_points": 120,
            "fp32_geometry_batches": 2,
        }
        if retry
        else {},
    }
    qualification = (
        COMPOSITE_DF_GRADIENT_QUALIFICATION
        if fitted
        else COMPOSITE_GRADIENT_QUALIFICATION
    )
    report = composite_gradient_precision(qualification, identity, work)
    assert report["returned_mode"] == ("fp64" if retry else "mixed")
    assert report["attempted_fp32_geometry_points"] == 120
    assert report["attempted_fp32_grid_points"] == 0
    assert "operator_inventory_complete" not in report
    work["stationary_source_work"]["nonlocal"]["fp32_geometry_points"] = 120
    with pytest.raises(RuntimeError, match="complete producer work"):
        composite_gradient_precision(qualification, identity, work)
    work["stationary_source_work"]["nonlocal"]["fp32_geometry_points"] = 0
    if fitted:
        work["native_integral_resources"]["mixed_rsh_df_provider"] = 0
        with pytest.raises(RuntimeError, match="complete producer work"):
            composite_gradient_precision(qualification, identity, work)

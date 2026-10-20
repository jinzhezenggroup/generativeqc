"""Native transport admission and independent CBS Fast observable gates."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    CBSBasis,
    CCSDTCBSCalculator,
    TargetErrorBudget,
    _native,
    load_basis,
)

REFERENCES = Path(__file__).resolve().parents[1] / "reference_data/cbs"
BUDGET = TargetErrorBudget(1e-5, 1e-4, 3e-5)
ATOMS = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))


def test_private_precision_query_rejects_invalid_buffers() -> None:
    """The private copied-counter ABI cannot write a null or undersized buffer."""
    library = _native.load_library(device="cpu", device_id=0)
    query = library.generativeqc_calculation_get_cc_precision_v1
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
    query.restype = ctypes.c_int
    counters = (ctypes.c_uint64 * 6)(*[99] * 6)
    assert query(None, counters, 6) == _native.STATUS_INVALID_ARGUMENT
    # A non-null sentinel is safe here only because invalid buffers/counts
    # must be rejected before dereferencing the handle.
    assert query(1, None, 6) == _native.STATUS_INVALID_ARGUMENT
    assert query(1, counters, 5) == _native.STATUS_INVALID_ARGUMENT
    assert list(counters) == [99] * 6


@pytest.mark.parametrize("version", (None, 0, 2))
def test_missing_native_convergence_capability_fails_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    version: int | None,
) -> None:
    library = _native.load_library(device="cpu", device_id=0)
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    monkeypatch.setattr(
        library,
        "generativeqc_correlation_convergence_version_v1",
        None if version is None else lambda: version,
    )
    with pytest.raises(NotImplementedError, match="convergence transport"):
        Calculator(
            method="ccsd(t)", basis="sto-3g", _correlation_convergence_budget=BUDGET
        )


def test_native_budget_descriptor_and_batch_boundary() -> None:
    """The legacy ABI stays strict and a batch cannot silently lose allowances."""
    owner = Calculator(
        method="ccsd(t)",
        basis="sto-3g",
        _correlation_convergence_budget=BUDGET,
        ccsd_residual_tolerance=1e-6,
        ccsd_energy_tolerance=1e-8,
    )
    try:
        descriptor = owner._method_descriptor()
        assert descriptor.struct_size == ctypes.sizeof(
            _native.ApplicationPrecisionMethodDescriptor
        )
        result = owner.singlepoint(ATOMS, properties=("energy",))
        assert result.converged and np.isfinite(result.energy)
        with pytest.raises(RuntimeError, match="single-point groups"):
            owner.prepare_batch([ATOMS], warm_start=False)
    finally:
        owner.clear_cache()
    with pytest.raises(ValueError, match="ccsd_residual_tolerance"):
        Calculator(method="ccsd(t)", basis="sto-3g", ccsd_residual_tolerance=1e-6)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_fast_composite_and_components_against_independent_water_oracle(
    device: str,
) -> None:
    """Gate nonzero triples and all force components against pinned PySCF data."""
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_CBS_CUDA_TEST") != "1":
            pytest.skip("requires an explicitly allocated CUDA library")
        assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require srun"
    reference = json.loads((REFERENCES / "water-algebra.json").read_text())
    workflow = CCSDTCBSCalculator(
        tuple(
            CBSBasis(load_basis(REFERENCES / reference[key]), cardinal)
            for key, cardinal in (("low_basis", 2), ("high_basis", 3))
        ),
        device=device,
        policy="fast",
    )
    result = workflow.singlepoint(reference["atoms"])
    oracle = reference["reference"]
    assert result.converged and result.convergence["selected"] == "relaxed"
    assert result.convergence["arithmetic"] == "fp64"
    assert abs(result.energy - oracle["energy"]) <= BUDGET.energy_abs
    error = result.forces - oracle["forces"]
    assert np.max(np.abs(error)) <= BUDGET.force_max_abs
    assert np.sqrt(np.mean(error**2)) <= BUDGET.force_rms
    allowance = result.convergence["component_budget"]
    for component in result.components:
        expected = oracle["components"][component.roles[0]]
        assert (
            abs(component.result.energy - expected["energy"]) <= allowance["energy_abs"]
        )
        delta = component.result.forces - expected["forces"]
        assert np.max(np.abs(delta)) <= allowance["force_max_abs"]
        assert np.sqrt(np.mean(delta**2)) <= allowance["force_rms"]
    cc = next(c.result.correlation for c in result.components if c.method == "ccsd(t)")
    assert (
        max(cc.ccsd_replay_singles_residual_max, cc.ccsd_replay_doubles_residual_max)
        <= result.convergence["limits"]["ccsd_residual_tolerance"]
    )
    assert cc.force_provenance_flags & 7 == 7
    energy_only = workflow.singlepoint(reference["atoms"], properties=("energy",))
    assert energy_only.forces is None
    assert abs(energy_only.energy - result.energy) < 1e-10


@pytest.mark.parametrize(
    "field,value",
    (
        ("application_energy_budget", float("nan")),
        ("application_energy_budget", -1.0),
        ("application_max_force_budget", 0.0),
        ("application_rms_force_budget", float("inf")),
        ("ccsd_residual_tolerance", 1e-3),
    ),
)
def test_native_rejects_malformed_correlated_allowances(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
) -> None:
    """Exercise native admission independently of Python's validated budget type."""
    owner = Calculator(
        method="ccsd(t)", basis="sto-3g", _correlation_convergence_budget=BUDGET
    )
    original = owner._method_descriptor

    def descriptor(*args: object, **kwargs: object) -> _native.MethodDescriptor:
        result = original(*args, **kwargs)
        setattr(result._extension_owner, field, value)
        return result

    monkeypatch.setattr(owner, "_method_descriptor", descriptor)
    try:
        with pytest.raises(RuntimeError):
            owner.singlepoint(ATOMS, properties=("energy",))
    finally:
        owner.clear_cache()

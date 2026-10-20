"""Real prepared Fast cold/frozen/moved fleets with independent observables."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    ObservableTarget,
    PrecisionCalculator,
    PrecisionPolicy,
    TargetAccuracy,
    TargetErrorBudget,
)
from generativeqc._precision_batch import _SeedLease
from test_df_fast_public_cuda import auxiliary_bases
from test_df_fast_public_cuda import options as fitted_options
from test_hf_fast_cuda import oracle as hf_oracle
from test_hf_fast_cuda import settings as hf_settings
from test_r2scan3c_fast_cuda import oracle as composite_oracle
from test_r2scan3c_fast_cuda import settings as composite_settings
from test_readme_fast_cuda import GRID, options

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
CASE = "water-def2-svp-spherical"
METHODS = [
    (name, fitted)
    for name in ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X")
    for fitted in (False, True)
] + [("HF", False), ("R2SCAN-3C", False)]


def test_hf_audit_replay_preserves_independent_frozen_history() -> None:
    """Warm both solvers separately and still execute a measured audit each time."""
    case_name = "water-tetramer-def2-svp-spherical"
    atoms = benchmark_cases()[case_name].atoms
    energy, forces = hf_oracle(case_name, atoms)
    target = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", absolute=1e-7),)
    )
    calculator = PrecisionCalculator(
        policy=PrecisionPolicy("fp64", target=target), **hf_settings(case_name)
    )
    try:
        with calculator.prepare_batch([atoms]) as prepared:
            for replay in range(3):
                result = prepared.execute(strict=True, properties=("energy", "forces"))
                observed = result.items[0]
                assert observed.energy == pytest.approx(energy, abs=1e-7)
                np.testing.assert_allclose(observed.forces, forces, atol=1e-6, rtol=0)
                assert observed.precision_validation.solve_count == 2
                attempt = result.precision_validation["audit_calls"][0]
                assert attempt["completed"] and attempt["iterations"] > 0
                # Some HF owners expose no Fock census; keep it unknown.
                assert attempt["fock_builds"] is None or attempt["fock_builds"] > 0
                assert attempt["warm_start_used"] == bool(replay)
                assert result.precision_validation["audit_seed_bytes_retained"] > 0
                if replay == 0:
                    prepared.set_warm_start_updates(False)
                    candidate = [array.copy() for array in prepared._seeds.storage]
                    auditor = [
                        array.copy() for array in prepared._audit_seeds[0].storage
                    ]
                else:
                    for actual, frozen in zip(
                        prepared._seeds.storage, candidate, strict=True
                    ):
                        np.testing.assert_array_equal(actual, frozen)
                    for actual, frozen in zip(
                        prepared._audit_seeds[0].storage, auditor, strict=True
                    ):
                        np.testing.assert_array_equal(actual, frozen)
    finally:
        calculator.clear_cache()


def inputs(functional: str, fitted: bool) -> dict:
    """Keep model, grid and strict stopping controls identical between modes."""
    if functional == "HF":
        return hf_settings(CASE)
    if functional == "R2SCAN-3C":
        return composite_settings()
    return fitted_options(CASE, functional) if fitted else options(CASE, functional)


def oracle(functional: str, fitted: bool, atoms: Any) -> dict:
    """Use separately owned integrals, grid response and composite corrections."""
    if functional == "HF":
        energy, forces = hf_oracle(CASE, atoms)
        return {"energy": energy, "forces": forces}
    if functional == "R2SCAN-3C":
        return composite_oracle(CASE, atoms)
    auxiliary = auxiliary_bases(benchmark_cases()[CASE])[1] if fitted else None
    return reference(
        CASE, GRID, 4, functional=functional, auxiliary_basis=auxiliary, atoms=atoms
    )


def geometries() -> tuple:
    """Move a bond, not the whole molecule or just an invariant frame."""
    atoms = benchmark_cases()[CASE].atoms
    moved = tuple(
        (
            element,
            tuple(
                np.asarray(position)
                + ([0.003, -0.002, 0.001] if index == 1 else [0, 0, 0])
            ),
        )
        for index, (element, position) in enumerate(atoms)
    )
    return atoms, moved


def check(observed: Any, expected: dict, *, fast: bool) -> None:
    """Qualify complete forces more tightly than the default Fast allowances."""
    assert observed.succeeded and observed.converged
    assert observed.executed_backend == "cuda"
    assert abs(observed.energy - expected["energy"]) < 1e-7
    difference = observed.forces - np.asarray(expected["forces"])
    assert np.max(np.abs(difference)) < (1e-5 if fast else 1e-6)
    assert np.sqrt(np.mean(difference**2)) < 3e-6
    # Older HF owners expose only the precision census, not the KS per-item
    # total. Unknown counts must stay unknown, never estimated from iterations.
    if observed.ks_diagnostic is not None:
        assert observed.fock_builds is not None
    if observed.fock_builds is not None:
        assert observed.fock_builds > 0
    if fast:
        validation = observed.precision_validation
        assert validation.status == "qualified_mixed_gradient"
        assert validation.solve_count == 1
        assert validation.solves[0].seconds is None
        gradient = observed.precision["gradient"]
        assert gradient["returned_mode"] == "mixed"
        assert (
            gradient.get("attempted_fp32_geometry_points", 0) > 0
            or gradient.get("attempted_fp32_weighted_tasks", 0) > 0
        )


@pytest.mark.parametrize("functional,fitted", METHODS)
def test_prepared_fast_frozen_and_moved_independent_endpoints(
    functional: str, fitted: bool, monkeypatch: Any
) -> None:
    """One fleet per mode; every replay retains its own post-cold/move density."""
    atoms, moved = geometries()
    expected = [oracle(functional, fitted, geometry) for geometry in (atoms, moved)]

    def no_scalar_replay(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "prepared Fast must execute a native fleet, not scalar replay"
        )

    monkeypatch.setattr(Calculator, "singlepoint", no_scalar_replay)
    settings = inputs(functional, fitted)
    for fast in (False, True):
        calculator = (
            PrecisionCalculator(policy="fast", **settings)
            if fast
            else Calculator(precision="fp64", **settings)
        )
        try:
            with calculator.prepare_batch([atoms]) as prepared:
                native_owner = prepared._owner if fast else prepared
                for geometry_index, geometry in enumerate((atoms, moved)):
                    prepared.set_warm_start_updates(True)
                    coordinates = (
                        None
                        if geometry_index == 0
                        else [np.asarray([position for _, position in geometry])]
                    )
                    for replay in range(3):
                        result = prepared.execute(
                            coordinates, strict=True, properties=("energy", "forces")
                        )
                        observed = result.items[0]
                        check(observed, expected[geometry_index], fast=fast)
                        if replay:
                            assert observed.warm_start_used
                        if fast:
                            assert prepared._owner is native_owner
                            assert result.precision_validation["audit_calls"] == []
                            assert (
                                result.precision_validation["seed_bytes_retained"] == 0
                            )
                        if replay == 0:
                            prepared.set_warm_start_updates(False)
        finally:
            calculator.clear_cache()


def test_fitted_energy_audit_restores_only_candidate_frozen_density() -> None:
    """The historical HF seed ABI really restores KS, including source geometry."""
    atoms, moved = geometries()
    expected = oracle("PBE0", True, moved)
    calculator = PrecisionCalculator(policy="fast", **inputs("PBE0", True))
    try:
        with calculator.prepare_batch([atoms]) as prepared:
            prepared.execute(strict=True)
            prepared.set_warm_start_updates(False)
            before = _SeedLease(prepared._owner, 1 << 20)
            energy = prepared.execute(properties=("energy",), strict=True)
            assert energy.items[0].forces is None
            assert (energy.items[0].precision or {}).get("gradient") is None
            assert energy.items[0].precision_validation.status == "observed_met"
            assert len(energy.precision_validation["audit_calls"]) == 1
            assert prepared._owner is None
            assert energy.precision_validation["seed_bytes_retained"] > 0
            for left, right in zip(
                before.storage, prepared._seeds.storage, strict=True
            ):
                np.testing.assert_array_equal(left, right)
            result = prepared.execute(
                [np.asarray([position for _, position in moved])], strict=True
            )
            check(result.items[0], expected, fast=True)
            assert result.items[0].warm_start_used
            after = _SeedLease(prepared._owner, 1 << 20)
            for left, right in zip(before.storage, after.storage, strict=True):
                np.testing.assert_array_equal(left, right)
    finally:
        calculator.clear_cache()


def test_tight_budget_audits_survive_real_failed_neighbor() -> None:
    atoms, moved = geometries()
    policy = PrecisionPolicy("fast", budget=TargetErrorBudget(1e-8, 1e-7, 3e-8))
    calculator = PrecisionCalculator(policy=policy, **inputs("PBE0", True))
    try:
        with calculator.prepare_batch([atoms, atoms]) as prepared:
            result = prepared.execute(
                [np.zeros(1), np.asarray([position for _, position in moved])]
            )
            assert result.failure_indices == (0,)
            assert result.items[0].precision_validation is None
            assert result.items[0].forces is None
            assert result.items[1].precision_validation.status == "observed_met"
            assert result.items[1].precision_validation.solve_count == 2
            assert result.precision_validation["audit_calls"][0]["index"] == 1
            assert prepared._owner is None
            prepared.clear_warm_starts()
            with pytest.raises(RuntimeError, match="batched item failures: 0"):
                prepared.execute([np.zeros(1), None], strict=True)
    finally:
        calculator.clear_cache()


@pytest.mark.parametrize("functional", ("PBE", "PBE0"))
@pytest.mark.parametrize("case_name", (CASE, "water-tetramer-def2-svp-spherical"))
def test_application_fast_does_not_perturb_converged_frozen_seeds(
    functional: str, case_name: str
) -> None:
    """Retain cold FP32 work without manufacturing a new warm SCF trajectory."""
    case = benchmark_cases()[case_name]
    settings = options(case_name, functional)
    settings.update(energy_tolerance=1e-12, density_tolerance=1e-10)
    expected = reference(case_name, GRID, 4, functional=functional)
    calculator = PrecisionCalculator(policy="fast", **settings)
    try:
        with calculator.prepare_batch([case.atoms]) as prepared:
            cold = prepared.execute(strict=True).items[0]
            check(cold, expected, fast=True)
            assert cold.precision["mixed_stage_fock_builds"] > 0
            prepared.set_warm_start_updates(False)
            for _ in range(2):
                result = prepared.execute(strict=True).items[0]
                check(result, expected, fast=True)
                assert result.warm_start_used and not result.warm_start_fallback
                assert result.precision["mixed_stage_fock_builds"] == 0
                assert result.precision["strict_stage_fock_builds"] > 0
                assert not result.precision["strict_refinement_applied"]
                assert result.fock_builds <= 2
                assert result.physical_residual_rms < 1e-9
                assert (
                    result.precision["gradient"]["attempted_fp32_geometry_points"] > 0
                )
    finally:
        calculator.clear_cache()

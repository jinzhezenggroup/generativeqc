"""Complete public local-RKS Fast candidates against independent moving grids.

These gates cover the currently bounded direct domain, not DF, RSH/VV10,
composites, HF or the full README molecule/timing matrix.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    PrecisionCalculator,
    ResourceBudget,
    TargetErrorBudget,
    _native,
)

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
GRID = GridSpec(48, 16, 32)
BUDGET = TargetErrorBudget(1e-5, 1e-4, 3e-5)


def options(case_name: str, method: str) -> dict[str, Any]:
    """Keep basis, quadrature and convergence fixed across arithmetic choices."""
    case = benchmark_cases()[case_name]
    return {
        "method": f"{method.lower()}-rks",
        "device": "cuda",
        "initial_guess": None,
        "basis": case.generativeqc_basis,
        "basis_representation": case.basis_representation,
        "ks_options": KsOptions(grid=GRID),
        "max_iterations": 180,
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }


@pytest.mark.parametrize("functional", ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X"))
@pytest.mark.parametrize(
    "case_name",
    (
        "water-def2-svp-spherical",
        "water-tetramer-def2-svp-spherical",
        "ammonia-def2-svp-spherical",
        "water-def2-svp",
    ),
)
def test_public_fast_complete_force_qualification(
    functional: str, case_name: str
) -> None:
    pytest.importorskip("pyscf")
    case = benchmark_cases()[case_name]
    oracle = reference(case_name, GRID, 4, functional=functional)
    settings = options(case_name, functional)
    fast = PrecisionCalculator(policy="fast", **settings)
    strict = Calculator(precision="fp64", **settings)
    try:
        baseline = strict.singlepoint(case.atoms)
        strict.clear_cache()
        candidate = fast.singlepoint(case.atoms)
        report = candidate.precision_validation
        assert report.status == "qualified_mixed_gradient"
        assert report.solve_count == 1
        assert report.gradient_admission == f"{functional.lower()}-rks-budget-fp32-v1"
        assert abs(candidate.energy - oracle["energy"]) < 1e-7
        np.testing.assert_allclose(baseline.forces, oracle["forces"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            candidate.forces, oracle["forces"], atol=1e-5, rtol=0
        )
        assert np.sqrt(np.mean((candidate.forces - baseline.forces) ** 2)) < 3e-6
        np.testing.assert_allclose(candidate.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
        assert candidate.precision["gradient"]["attempted_fp32_geometry_points"] > 0
        assert candidate.precision["gradient"]["attempted_fp32_grid_points"] > 0
        assert candidate.precision["gradient"]["fp64_gradient_retries"] == 0
        replay = fast.singlepoint(case.atoms, properties=("energy",))
        assert replay.precision.get("gradient") is None
    finally:
        fast.clear_cache()
        strict.clear_cache()


def test_m06_compatibility_owner_preserves_candidate_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain a scientific comparator for the alternative strict schedule."""
    case_name = "water-def2-svp-spherical"
    case = benchmark_cases()[case_name]
    settings = options(case_name, "M06-2X")
    strict = Calculator(precision="fp64", **settings)
    mixed = Calculator(
        precision="fp64", _application_precision_budget=BUDGET, **settings
    )
    try:
        with strict.prepare_batch([case.atoms]) as batch:
            monkeypatch.setattr(
                batch, "_native_dft_force_eligible", lambda index=None: False
            )
            assert not batch._native_dft_force_eligible(0)
            baseline = batch.execute(strict=True).items[0]
        with mixed.prepare_batch([case.atoms]) as batch:
            assert batch._native_dft_force_eligible(0)
            candidate = batch.execute(strict=True).items[0]
        assert candidate.precision["gradient"]["returned_mode"] == "mixed"
        assert candidate.precision["gradient"]["attempted_fp32_geometry_points"] > 0
        np.testing.assert_allclose(candidate.forces, baseline.forces, atol=1e-5, rtol=0)
    finally:
        strict.clear_cache()
        mixed.clear_cache()


def test_v2_strict_scf_retains_mixed_gradient_and_immutable_prepared_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation owns its copy, and a budget never silently enables AUTO XC."""
    case_name = "water-def2-svp-spherical"
    case = benchmark_cases()[case_name]
    calculator = Calculator(
        precision="fp64",
        _application_precision_budget=BUDGET,
        **options(case_name, "PBE0"),
    )
    descriptors = []
    marshal = calculator._method_descriptor

    def capture(*args: Any, **kwargs: Any) -> Any:
        descriptor = marshal(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(calculator, "_method_descriptor", capture)
    try:
        with calculator.prepare_batch([case.atoms]) as batch:
            assert descriptors
            for descriptor in descriptors:
                descriptor._extension_owner.application_energy_budget = 1e-8
                descriptor._extension_owner.application_max_force_budget = 1e-7
                descriptor._extension_owner.application_rms_force_budget = 3e-8
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            assert result.precision["mixed_stage_fock_builds"] == 0
            assert result.precision["gradient"]["returned_mode"] == "mixed"
            assert (
                result.precision["gradient"]["qualification"]
                == "pbe0-rks-budget-fp32-v1"
            )
            assert result.precision["gradient"]["attempted_fp32_grid_points"] > 0
    finally:
        calculator.clear_cache()


@pytest.mark.parametrize(
    "malformation,detail",
    (
        ("truncated", "incomplete method precision"),
        ("nonfinite", "finite and nonnegative"),
        ("missing_energy", "requires an energy allowance"),
    ),
)
def test_v2_native_rejects_malformed_request(
    malformation: str, detail: str, monkeypatch: Any
) -> None:
    case_name = "water-def2-svp-spherical"
    calculator = Calculator(
        precision="fp64",
        _application_precision_budget=BUDGET,
        **options(case_name, "PBE0"),
    )
    marshal = calculator._method_descriptor

    def malformed(*args: Any, **kwargs: Any) -> Any:
        descriptor = marshal(*args, **kwargs)
        if malformation == "truncated":
            descriptor.struct_size = ctypes.sizeof(_native.MethodDescriptor) + 8
        else:
            descriptor._extension_owner.application_energy_budget = (
                float("nan") if malformation == "nonfinite" else 0.0
            )
        return descriptor

    monkeypatch.setattr(calculator, "_method_descriptor", malformed)
    try:
        with pytest.raises(RuntimeError, match=detail):
            calculator.singlepoint(benchmark_cases()[case_name].atoms)
    finally:
        calculator.clear_cache()


def test_budget_resource_plan_reserves_and_binds_actual_mixed_grid_storage() -> None:
    """The dry-run descriptor and prepared owner must agree under a hard ledger."""
    case_name = "water-def2-svp-spherical"
    case = benchmark_cases()[case_name]
    settings = options(case_name, "PBE0")
    strict = Calculator(precision="fp64", **settings)
    mixed = Calculator(
        precision="fp64",
        _application_precision_budget=BUDGET,
        resource_budget=ResourceBudget(),
        **settings,
    )
    try:
        baseline = strict.estimate_resources(
            [case.atoms], properties=("energy", "forces")
        )
        planned = mixed.estimate_resources(
            [case.atoms], properties=("energy", "forces")
        )
        assert planned.peak_bytes["device"] > baseline.peak_bytes["device"]
        with mixed.prepare_batch([case.atoms]) as batch:
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            assert result.precision["mixed_stage_fock_builds"] == 0
            assert result.precision["gradient"]["attempted_fp32_grid_points"] > 0
            assert result.precision["gradient"]["fp64_gradient_retries"] == 0
    finally:
        strict.clear_cache()
        mixed.clear_cache()


@pytest.mark.parametrize("functional", ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X"))
def test_direct_rks_prepared_geometry_replay_and_failed_peer(functional: str) -> None:
    """A frozen warm seed cannot substitute for fresh moved-geometry evidence."""
    pytest.importorskip("pyscf")
    name = "water-def2-svp-spherical"
    atoms = benchmark_cases()[name].atoms
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in atoms]
    moved[1][1][0] += 0.025
    positions = np.array([position for _, position in moved])
    original = reference(name, GRID, 4, functional=functional)
    changed = reference(name, GRID, 4, functional=functional, atoms=moved)
    calculator = Calculator(
        precision="fp64",
        _application_precision_budget=BUDGET,
        **options(name, functional),
    )

    def check(result: Any, expected: dict[str, Any]) -> None:
        assert result.converged
        assert abs(result.energy - expected["energy"]) < 1e-7
        difference = result.forces - np.asarray(expected["forces"])
        assert np.max(np.abs(difference)) < 1e-5
        assert np.sqrt(np.mean(difference**2)) < 3e-6
        work = result.precision["gradient"]
        assert work["qualification"] == f"{functional.lower()}-rks-budget-fp32-v1"
        assert work["attempted_fp32_geometry_points"] > 0
        assert work["attempted_fp32_grid_points"] > 0
        assert work["fp64_gradient_retries"] == 0
        assert result.precision["mixed_stage_fock_builds"] == 0

    try:
        with calculator.prepare_batch([atoms, atoms]) as batch:
            for item in batch.execute(
                properties=("energy", "forces"), strict=True
            ).items:
                check(item, original)
            batch.set_warm_start_updates(False)
            for item in batch.execute(
                properties=("energy", "forces"), strict=True
            ).items:
                check(item, original)
            batch.set_warm_start_updates(True)
            result = batch.execute(
                [positions, None], properties=("energy", "forces"), strict=True
            )
            check(result.items[0], changed)
            check(result.items[1], original)
            batch.set_warm_start_updates(False)
            result = batch.execute(
                [positions, None], properties=("energy", "forces"), strict=True
            )
            check(result.items[0], changed)
            check(result.items[1], original)
            result = batch.execute(
                [positions, np.zeros((1, 3))], properties=("energy", "forces")
            )
            assert result.failure_indices == (1,)
            assert (result.items[1].precision or {}).get("gradient") is None
            check(result.items[0], changed)
    finally:
        calculator.clear_cache()

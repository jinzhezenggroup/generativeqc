"""Budget-selected electronic derivatives with unchanged canonical 3c corrections.

The independent oracle combines PySCF moving-grid r2SCAN with upstream dftd4
and the separate CPU gCP reference. Production never calls those oracles.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    PrecisionCalculator,
    TargetErrorBudget,
    load_r2scan3c_basis,
)

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference
from benchmarks.readme_omol25 import reference_correction

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
GRID = GridSpec(48, 16, 32)
BUDGET = TargetErrorBudget(1e-5, 1e-4, 3e-5)


def settings() -> dict[str, Any]:
    """The public composite selector owns its exact defining basis and model."""
    return {
        "method": "r2scan-3c-rks",
        "device": "cuda",
        "initial_guess": None,
        "ks_options": KsOptions(grid=GRID),
        "max_iterations": 180,
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }


def oracle(name: str, atoms: Any = None) -> dict[str, Any]:
    """Share only basis/geometry inputs across independently owned mathematics."""
    pytest.importorskip("pyscf")
    pytest.importorskip("dftd4")
    if atoms is None:
        atoms = benchmark_cases()[name].atoms
    electronic = reference(
        name, GRID, 4, functional="R2SCAN", atoms=atoms, basis=load_r2scan3c_basis()
    )
    correction_energy, correction_gradient = reference_correction(atoms)()
    return {
        "energy": electronic["energy"] + correction_energy,
        "forces": np.asarray(electronic["forces"]) - correction_gradient,
        "correction_energy": correction_energy,
        "correction_gradient": correction_gradient,
    }


def check(result: Any, expected: dict[str, Any], *, mixed: bool = True) -> None:
    """Check the complete endpoint as well as each unchanged correction source."""
    assert result.converged
    assert abs(result.energy - expected["energy"]) < 1e-7
    difference = result.forces - expected["forces"]
    assert np.max(np.abs(difference)) < (1e-5 if mixed else 1e-6)
    assert np.sqrt(np.mean(difference**2)) < 3e-6
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
    assert result.dispersion.ok and result.dispersion.d4.ok and result.dispersion.gcp.ok
    assert result.dispersion.backend == "cuda+cpu"
    assert abs(result.dispersion.energy - expected["correction_energy"]) < 1e-10
    np.testing.assert_allclose(
        result.dispersion.gradient, expected["correction_gradient"], atol=1e-9, rtol=0
    )
    if mixed:
        work = result.precision["gradient"]
        assert work["qualification"] == "r2scan-rks-budget-fp32-v1"
        assert work["attempted_fp32_geometry_points"] > 0
        assert work["attempted_fp32_grid_points"] > 0
        assert work["fp64_gradient_retries"] == 0
    else:
        assert result.precision.get("gradient") is None


@pytest.mark.parametrize(
    "name",
    (
        "water-def2-svp-spherical",
        "ammonia-def2-svp-spherical",
        "water-dimer-def2-tzvp-spherical",
    ),
)
def test_r2scan3c_public_fast_complete_independent_endpoint(name: str) -> None:
    atoms = benchmark_cases()[name].atoms
    expected = oracle(name)
    fast = PrecisionCalculator(policy="fast", **settings())
    strict = Calculator(precision="fp64", **settings())
    try:
        baseline = strict.singlepoint(atoms)
        check(baseline, expected, mixed=False)
        strict.clear_cache()
        candidate = fast.singlepoint(atoms)
        check(candidate, expected)
        assert candidate.precision_validation.status == "qualified_mixed_gradient"
        assert candidate.precision_validation.solve_count == 1
        assert (
            candidate.precision_validation.gradient_admission
            == "r2scan-3c-rks-budget-fp32-v1"
        )
        assert candidate.dispersion.energy == baseline.dispersion.energy
        np.testing.assert_allclose(
            candidate.dispersion.gradient,
            baseline.dispersion.gradient,
            atol=1e-13,
            rtol=0,
        )
    finally:
        fast.clear_cache()
        strict.clear_cache()


def test_r2scan3c_frozen_seed_and_moved_corrections_share_current_geometry() -> None:
    name = "water-def2-svp-spherical"
    atoms = benchmark_cases()[name].atoms
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in atoms]
    moved[1][1][0] += 0.025
    positions = np.asarray([position for _, position in moved])
    original, changed = oracle(name), oracle(name, moved)
    calculator = Calculator(
        precision="fp64", _application_precision_budget=BUDGET, **settings()
    )
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
            replay = batch.execute(
                [positions, None], properties=("energy", "forces"), strict=True
            )
            check(replay.items[0], changed)
            check(replay.items[1], original)
            invalid = batch.execute(
                [positions, np.zeros((1, 3))], properties=("energy", "forces")
            )
            assert invalid.failure_indices == (1,)
            assert invalid.items[1].dispersion is None
            check(invalid.items[0], changed)
    finally:
        calculator.clear_cache()

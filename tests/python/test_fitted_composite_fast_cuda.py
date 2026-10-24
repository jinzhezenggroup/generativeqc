"""Complete fitted Fast models with unchanged range exchange or 3c corrections.

The independent WB oracle fits full-range J/K, but not long-range exchange.
The 3c oracle includes separate upstream D4 and gCP, not native corrections.
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
    PrecisionPolicy,
    TargetErrorBudget,
    load_r2scan3c_basis,
)
from generativeqc import _stationary_cuda as runtime
from test_df_fast_public_cuda import auxiliary_bases
from test_dft_complete_cuda import no_cpu_derivatives

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference
from benchmarks.readme_omol25 import reference_correction

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires finite Slurm GPU allocation",
)
GRID = GridSpec(48, 16, 32)
CASE = "water-def2-svp-spherical"


def options(method: str, case_name: str, grid: GridSpec = GRID) -> dict[str, Any]:
    """Keep the defining orbital basis and independently matched auxiliary set."""
    case = benchmark_cases()[case_name]
    auxiliary, _ = auxiliary_bases(case)
    return {
        "method": f"{method}-rks",
        **(
            {
                "basis": case.generativeqc_basis,
                "basis_representation": case.basis_representation,
            }
            if method == "wb97m-v"
            else {}
        ),
        "auxiliary_basis": auxiliary,
        "density_fitting": "cuda",
        "device": "cuda",
        "ks_options": KsOptions(grid=grid),
        "max_iterations": 180,
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }


def oracle(
    method: str, case_name: str, atoms: Any, grid: GridSpec = GRID
) -> dict[str, Any]:
    """No production density, grid weights or observables enter this reference."""
    pytest.importorskip("gpu4pyscf")
    if method == "r2scan-3c":
        pytest.importorskip("dftd4")
    _, auxiliary = auxiliary_bases(benchmark_cases()[case_name])
    expected = reference(
        case_name,
        grid,
        4,
        atoms=atoms,
        auxiliary_basis=auxiliary,
        functional="WB97M_V" if method == "wb97m-v" else "R2SCAN",
        basis=load_r2scan3c_basis() if method == "r2scan-3c" else None,
        backend="gpu4pyscf",
    )
    if method == "r2scan-3c":
        energy, gradient = reference_correction(atoms)()
        expected["energy"] += energy
        expected["forces"] = np.asarray(expected["forces"]) - gradient
        expected["correction_energy"] = energy
        expected["correction_gradient"] = gradient
    return expected


def check(
    result: Any, expected: dict, method: str, *, fast: bool, retry: bool = False
) -> None:
    """Check every component without interpreting requested precision as work."""
    assert result.converged and result.executed_backend == "cuda"
    assert abs(result.energy - expected["energy"]) < 1e-7
    difference = result.forces - np.asarray(expected["forces"])
    assert np.max(np.abs(difference)) < (1e-5 if fast and not retry else 1e-6)
    assert np.sqrt(np.mean(difference**2)) < 3e-6
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
    if method == "r2scan-3c":
        assert (
            result.dispersion.ok
            and result.dispersion.d4.ok
            and result.dispersion.gcp.ok
        )
        assert result.dispersion.backend == "cuda+cpu"
        assert abs(result.dispersion.energy - expected["correction_energy"]) < 1e-10
        np.testing.assert_allclose(
            result.dispersion.gradient,
            expected["correction_gradient"],
            atol=1e-9,
            rtol=0,
        )
    if fast:
        report = result.precision_validation
        assert report.status == (
            "qualified_strict_final" if retry else "qualified_mixed_gradient"
        )
        assert report.solve_count == 1
        assert report.solves[0].requested_precision == "fp64"
        assert report.gradient_admission == f"{method}-df-rks-budget-fp32-v1"
        work = result.precision["gradient"]
        assert work["schema_version"] == 4
        assert work["owner"] == (
            "python-composite-stationary-cuda"
            if method == "wb97m-v"
            else "python-stationary-cuda"
        )
        assert work["attempted_fp32_geometry_points"] > 0
        assert work["returned_fp32_geometry_points"] == (
            0 if retry else expected["grid_points"]
        )
        assert work["attempted_fp32_grid_points"] == 0
        assert work["fp64_gradient_retries"] == int(retry)


@pytest.mark.parametrize(
    "method,case_name",
    (
        ("wb97m-v", CASE),
        ("wb97m-v", "ammonia-def2-svp-spherical"),
        ("wb97m-v", "water-tetramer-def2-svp-spherical"),
        ("wb97m-v", "water-def2-svp"),
        ("r2scan-3c", CASE),
        ("r2scan-3c", "ammonia-def2-svp-spherical"),
        ("r2scan-3c", "water-dimer-def2-tzvp-spherical"),
    ),
)
def test_complete_fitted_composite_scalar_frozen_and_moved(
    method: str, case_name: str
) -> None:
    """Public admission must execute real mixed work at both geometries."""
    atoms = benchmark_cases()[case_name].atoms
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in atoms]
    moved[1][1][0] += 0.025
    expected = [oracle(method, case_name, geometry) for geometry in (atoms, moved)]
    calculators = [
        Calculator(precision="fp64", **options(method, case_name)),
        PrecisionCalculator(policy="fast", **options(method, case_name)),
    ]
    try:
        for fast, calculator in zip((False, True), calculators, strict=True):
            with no_cpu_derivatives():
                check(calculator.singlepoint(atoms), expected[0], method, fast=fast)
                calculator.clear_cache()
                with calculator.prepare_batch([atoms]) as batch:
                    for positions, reference_value in zip(
                        (None, [np.asarray([position for _, position in moved])]),
                        expected,
                        strict=True,
                    ):
                        batch.set_warm_start_updates(True)
                        check(
                            batch.execute(positions, strict=True).items[0],
                            reference_value,
                            method,
                            fast=fast,
                        )
                        batch.set_warm_start_updates(False)
                        for _ in range(2):
                            result = batch.execute(positions, strict=True).items[0]
                            check(result, reference_value, method, fast=fast)
                            assert result.warm_start_used
    finally:
        for calculator in calculators:
            calculator.clear_cache()


@pytest.mark.parametrize("method", ("wb97m-v", "r2scan-3c"))
def test_fitted_composite_pending_retry_and_evidence_expiry(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete strict retry includes fitted response and every correction."""
    atoms = benchmark_cases()[CASE].atoms
    expected = oracle(method, CASE, atoms)
    original = runtime._CudaSources._call
    injected = False

    def call(owner: Any, symbol: str, *args: Any) -> Any:
        nonlocal injected
        if (
            owner.fp32_geometry_requested
            and symbol in ("stationary_finish", "stationary_finish_span")
            and not injected
        ):
            assert owner.metrics()["fp32_geometry_points"] > 0
            injected = True
            raise ArithmeticError("injected pending fitted composite geometry")
        return original(owner, symbol, *args)

    monkeypatch.setattr(runtime._CudaSources, "_call", call)
    calculator = PrecisionCalculator(policy="fast", **options(method, CASE))
    try:
        with calculator.prepare_batch([atoms, atoms]) as batch:
            failed = batch.execute([[0.0], None], strict=False)
            assert failed.failure_indices == (0,) and injected
            assert not (failed.items[0].precision or {}).get("gradient")
            check(failed.items[1], expected, method, fast=True, retry=True)
            result = batch.execute(strict=True)
            for item in result.items:
                check(item, expected, method, fast=True)
            energies = batch.execute(properties=("energy",), strict=True)
            for item in energies.items:
                assert item.forces is None
                assert not (item.precision or {}).get("gradient")
                # An energy-only call may use a qualified strict endpoint or
                # an audited fallback. Neither may inherit the preceding
                # force work/evidence; the number of complete solves is not
                # the scientific or lifetime contract.
                assert item.converged and abs(item.energy - expected["energy"]) < 1e-7
                report = item.precision_validation
                assert report.solves
                assert all(solve.force_evaluations == 0 for solve in report.solves)
            assert result.items[1].precision["gradient"]["returned_mode"] == "mixed"
    finally:
        calculator.clear_cache()


def test_fitted_3c_coarse_grid_keeps_independent_corrections() -> None:
    """The second admitted DF grid needs evidence with the defining 3c basis."""
    grid = GridSpec(24, 8, 16)
    atoms = benchmark_cases()[CASE].atoms
    expected = oracle("r2scan-3c", CASE, atoms, grid)
    for fast in (False, True):
        calculator = (
            PrecisionCalculator(policy="fast", **options("r2scan-3c", CASE, grid))
            if fast
            else Calculator(precision="fp64", **options("r2scan-3c", CASE, grid))
        )
        try:
            with calculator.prepare_batch([atoms]) as batch:
                check(
                    batch.execute(strict=True).items[0],
                    expected,
                    "r2scan-3c",
                    fast=fast,
                )
                batch.set_warm_start_updates(False)
                check(
                    batch.execute(strict=True).items[0],
                    expected,
                    "r2scan-3c",
                    fast=fast,
                )
        finally:
            calculator.clear_cache()


@pytest.mark.parametrize("method", ("wb97m-v", "r2scan-3c"))
def test_fitted_composite_tight_budget_retains_strict_audit(method: str) -> None:
    """A complete-model label cannot relax the user-supplied force allowance."""
    calculator = PrecisionCalculator(
        policy=PrecisionPolicy("fast", budget=TargetErrorBudget(1e-7, 1e-6, 3e-7)),
        **options(method, CASE),
    )
    try:
        result = calculator.singlepoint(benchmark_cases()[CASE].atoms)
        report = result.precision_validation
        assert report.status == "observed_met" and report.solve_count == 2
        assert report.errors
        assert (
            not (result.precision or {})
            .get("gradient", {})
            .get("attempted_fp32_geometry_points")
        )
    finally:
        calculator.clear_cache()

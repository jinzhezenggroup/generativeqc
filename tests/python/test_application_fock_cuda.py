"""Shared Fock arithmetic composes with each native XC ingredient family."""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import numpy as np
import pytest
from generativeqc import GridSpec, KsOptions, PrecisionCalculator

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicitly Slurm-allocated GPU",
)

CASE = "water-def2-svp-spherical"
GRID = GridSpec(48, 16, 32)


@cache
def _oracle(functional: str) -> dict[str, Any]:
    """Evaluate identical moving quadrature with an independent SCF/force owner."""
    pytest.importorskip("pyscf")
    pytest.importorskip("gpu4pyscf")
    return reference(CASE, GRID, 4, functional=functional, backend="gpu4pyscf")


@pytest.mark.parametrize(
    ("method", "functional"),
    (
        ("lda-rks", "LDA_X,LDA_C_PW"),
        ("pbe-rks", "PBE"),
        ("pbe0-rks", "PBE0"),
        ("b3lyp-rks", "B3LYP"),
        ("r2scan-rks", "R2SCAN"),
        ("m06-2x-rks", "M06-2X"),
        ("mn15-rks", "MN15"),
        ("wb97m-v-rks", "WB97M-V"),
    ),
)
def test_shared_fast_fock_complete_endpoint(
    method: str, functional: str, record_property: Any
) -> None:
    """Changing XC composition must not select another J/K precision policy."""
    case = benchmark_cases()[CASE]
    calculator = PrecisionCalculator(
        method=method,
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        initial_guess=None,
        max_iterations=180,
        policy="fast",
    )
    try:
        observed = calculator.singlepoint(case.atoms, properties=("energy", "forces"))
        report = observed.precision_validation
        assert observed.converged and observed.executed_backend == "cuda"
        assert report.status == "qualified_budget_fock"
        assert report.solve_count == 1 and report.errors == ()
        assert report.reference == "direct-fock-budget-v2/shared-module-qualification"
        assert calculator._candidate._density_tolerance == 1e-4
        work = observed.precision
        assert work["mixed_stage_fock_builds"] > 0
        assert work["strict_stage_fock_builds"] >= 2
        assert work["execution_retries"] == 0
        record_property("mixed_fock_builds", work["mixed_stage_fock_builds"])
        record_property("strict_fock_builds", work["strict_stage_fock_builds"])
        record_property("solve_seconds", report.solves[0].seconds)
        exchange = calculator._candidate.ks_options.execution_plan.exchange
        expected_k = calculator._candidate.ks_options.coefficients[2] != 0
        assert (
            any(
                row["name"] == "exchange_k"
                and row["compute"] == "fp32"
                and row["count"] > 0
                for row in work["operators"]
            )
            == expected_k
        )
        if any(term.operator != "full-range" for term in exchange):
            assert any(
                row["name"] == "exchange_k"
                and row["compute"] == "fp64"
                and row["count"] > 0
                for row in work["operators"]
            )
        oracle = _oracle(functional)
        energy_error = abs(observed.energy - oracle["energy"])
        difference = np.asarray(observed.forces) - oracle["forces"]
        force_error = float(np.max(np.abs(difference)))
        force_rms = float(np.sqrt(np.mean(difference**2)))
        record_property("energy_error_Eh", energy_error)
        record_property("maximum_force_error_Eh_per_bohr", force_error)
        record_property("force_rms_Eh_per_bohr", force_rms)
        assert energy_error <= report.budget.energy_abs
        assert force_error <= report.budget.force_max_abs
        assert force_rms <= report.budget.force_rms
    finally:
        calculator.clear_cache()

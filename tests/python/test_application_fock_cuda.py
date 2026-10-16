"""Shared Fock arithmetic composes with each native XC ingredient family."""

from __future__ import annotations

import os
from dataclasses import replace
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
def _oracle(functional: str, case_name: str = CASE) -> dict[str, Any]:
    """Evaluate identical moving quadrature with an independent SCF/force owner."""
    pytest.importorskip("pyscf")
    pytest.importorskip("gpu4pyscf")
    return reference(case_name, GRID, 4, functional=functional, backend="gpu4pyscf")


@pytest.mark.parametrize("selection", ("auto", "full", "partial", "refused"))
def test_fast_range_source_composes_with_strict_refinement(
    monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    """Reuse, partial coverage and capacity fallback retain independent gates.

    The range operator is selected by the execution plan, not the functional's
    name. FP32 cache applications must not masquerade as new recurrences or
    authorize strict refinement to consume the approximate source.
    """
    case = benchmark_cases()[CASE]
    monkeypatch.setenv(
        "GENERATIVEQC_KS_RANGE_RESIDENT_VALUES", "auto" if selection == "auto" else "1"
    )
    monkeypatch.setenv(
        "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
        "0x1555" if selection == "partial" else "0xfffffff",
    )
    monkeypatch.setenv(
        "GENERATIVEQC_KS_RESIDENT_MAX_BYTES",
        "1" if selection == "refused" else str(1 << 30),
    )
    calculator = PrecisionCalculator(
        method="wb97m-v-rks",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        initial_guess=None,
        max_iterations=180,
        policy="fast",
    )
    try:
        result = calculator.singlepoint(case.atoms, properties=("energy", "forces"))
    finally:
        calculator.clear_cache()

    assert (
        result.converged
        and result.precision_validation.status == "qualified_budget_fock"
    )
    assert result.precision_validation.solve_count == 1
    work = result.precision
    assert work["execution_retries"] == 0 and work["strict_stage_fock_builds"] >= 2
    assert any(
        row["name"] == "exchange_k"
        and row["compute"] == "fp32"
        and row["storage"] == "fp32"
        and row["count"] > 0
        for row in work["operators"]
    ) == (selection != "refused")
    assert any(
        row["name"] == "exchange_k"
        and row["compute"] == "fp64"
        and row["storage"] == "fp64"
        and row["count"] > 0
        for row in work["operators"]
    )
    oracle = _oracle("WB97M-V")
    delta = np.asarray(result.forces) - np.asarray(oracle["forces"])
    assert abs(result.energy - oracle["energy"]) <= 1e-5
    assert np.max(np.abs(delta)) <= 1e-4
    assert np.sqrt(np.mean(delta * delta)) <= 3e-5


@pytest.mark.parametrize("case_name", (CASE, "water-tetramer-def2-svp-spherical"))
@pytest.mark.parametrize("strict_tolerances", (False, True))
def test_fast_nonlocal_refinement_refreshes_then_reuses_strict_anchor(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    strict_tolerances: bool,
    record_property: Any,
) -> None:
    """Reuse a fresh strict anchor without publishing a provisional inner state.

    The production mixed budget must retain the shared outer solver for both
    molecular sizes. Budget proposals finish their frozen inner stage in FP32;
    explicitly strict tolerances retain the strict inner solver. The first
    strict Fock always replaces the mixed anchor. Independent moving-grid
    energy/force gates still apply to the complete returned endpoint.
    """
    monkeypatch.setenv("GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED", "1")
    case = benchmark_cases()[case_name]
    calculator = PrecisionCalculator(
        method="wb97m-v-rks",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        initial_guess=None,
        max_iterations=180,
        policy="fast",
        **(
            {
                "energy_tolerance": 1e-10,
                "density_tolerance": 1e-8,
                "screening_tolerance": 1e-12,
            }
            if strict_tolerances
            else {}
        ),
    )
    try:
        result = calculator.singlepoint(case.atoms, properties=("energy", "forces"))
    finally:
        calculator.clear_cache()
    assert result.converged
    validation = result.precision_validation
    assert validation.status == "qualified_budget_fock" and validation.solve_count == 1
    assert result.precision["execution_retries"] == 0
    assert result.precision["fallback_count"] == 0
    history = result.ks_diagnostic.to_payload()["history"]
    scopes = {row["iteration"]: row["operator_scope"] for row in history}
    strict_iterations = [
        event["iteration"]
        for event in result.precision["scf_fock_timeline"]
        if event["kind"] == "strict_fock"
    ]
    assert strict_iterations and scopes[strict_iterations[0]] == "full-target"
    if strict_tolerances:
        assert any(
            scopes[iteration] == "provisional-frozen-vv10"
            for iteration in strict_iterations
        )
    else:
        assert any(
            event["kind"] == "mixed_fock"
            and scopes[event["iteration"]] == "provisional-frozen-vv10"
            for event in result.precision["scf_fock_timeline"]
        )
    assert all(row["operator_scope"] == "full-target" for row in history[-2:])
    for name, count in result.ks_diagnostic.nonlocal_work.items():
        record_property(name, count)
    oracle = _oracle("WB97M-V", case_name)
    difference = np.asarray(result.forces) - np.asarray(oracle["forces"])
    assert abs(result.energy - oracle["energy"]) <= validation.budget.energy_abs
    assert np.max(np.abs(difference)) <= validation.budget.force_max_abs
    assert np.sqrt(np.mean(difference * difference)) <= validation.budget.force_rms


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
        expected_k = bool(exchange)
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
            mixed_k = sum(
                row["count"]
                for row in work["operators"]
                if row["name"] == "exchange_k" and row["compute"] == "fp32"
            )
            record_property("mixed_k_applications", mixed_k)
            assert mixed_k > work["mixed_stage_fock_builds"]
            assert any(
                row["name"] == "exchange_k"
                and row["compute"] == "fp64"
                and row["count"] > 0
                for row in work["operators"]
            )
        if (
            calculator._candidate.ks_options.execution_plan.nonlocal_correlation
            is not None
        ):
            assert any(
                row["name"] == "nonlocal_correlation"
                and row["compute"] == "fp32"
                and row["accumulation"] == "fp64"
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


@pytest.mark.parametrize("case_name", (CASE, "water-tetramer-def2-svp-spherical"))
@pytest.mark.parametrize(
    ("method", "functional"),
    (
        ("pbe-rks", "PBE"),
        ("pbe0-rks", "PBE0"),
        ("b3lyp-rks", "B3LYP"),
        ("r2scan-rks", "R2SCAN"),
        ("m06-2x-rks", "M06-2X"),
    ),
)
def test_budget_fock_keeps_native_fp64_force_owner(
    case_name: str, method: str, functional: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relaxed Focks must not lose an already qualified native FP64 force body.

    The native prepared capability remains the authority. Forbid compatibility
    orchestration only for these independently gated, admitted complete graphs;
    unsupported graphs must retain their bounded compatibility consumer.
    """
    from generativeqc.batch import PreparedBatch

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("budget Fock lost the prepared native FP64 force consumer")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cuda_force", forbidden)
    case = benchmark_cases()[case_name]
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
        moved = replace(
            case,
            atoms=tuple(
                (element, (position[0] + (0.01 if index == 1 else 0.0), *position[1:]))
                for index, (element, position) in enumerate(case.atoms)
            ),
        )
        for phase, current in (("cold", case), ("warm", case), ("moved", moved)):
            result = calculator.singlepoint(
                current.atoms, properties=("energy", "forces")
            )
            assert result.precision_validation.status == "qualified_budget_fock", phase
            assert result.precision_validation.solve_count == 1
            work = result.precision
            assert work["mixed_stage_fock_builds"] > 0
            assert (
                work["strict_stage_fock_builds"] >= 2 and work["execution_retries"] == 0
            )
            gradient = work.get("gradient") or {}
            assert not gradient.get("attempted_fp32_geometry_points", 0)
            assert not gradient.get("attempted_fp32_primitive_sources", 0)
            oracle = (
                reference(
                    case_name,
                    GRID,
                    4,
                    functional=functional,
                    backend="gpu4pyscf",
                    case_snapshot=current,
                )
                if phase == "moved"
                else _oracle(functional, case_name)
            )
            delta = np.asarray(result.forces) - np.asarray(oracle["forces"])
            assert abs(result.energy - oracle["energy"]) <= 1e-5
            assert np.max(np.abs(delta)) <= 1e-4
            assert np.sqrt(np.mean(delta * delta)) <= 3e-5
    finally:
        calculator.clear_cache()

"""Public dense-grid WB97M-V Fast, including its strict RSH and VV10 sources."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    ObservableTarget,
    PrecisionCalculator,
    PrecisionPolicy,
    TargetAccuracy,
    TargetErrorBudget,
)
from generativeqc import _stationary_cuda as runtime
from test_dft_complete_cuda import no_cpu_derivatives

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires finite Slurm GPU allocation",
)
GRID = GridSpec(48, 16, 32)
CASE = "water-def2-svp-spherical"


def independent_reference(case_name: str, atoms: Any = None) -> dict:
    """Choose an explicit independent engine, never a native density or grid."""
    return reference(
        case_name,
        GRID,
        4,
        functional="WB97M_V",
        atoms=atoms,
        backend=os.environ.get("GENERATIVEQC_WB97MV_REFERENCE", "pyscf"),
    )


def options(case_name: str = CASE) -> dict:
    """Use the ordinary strict model, grid, seeds and stopping criteria."""
    case = benchmark_cases()[case_name]
    return {
        "method": "wb97m-v-rks",
        "basis": case.generativeqc_basis,
        "basis_representation": case.basis_representation,
        "device": "cuda",
        "ks_options": KsOptions(grid=GRID),
        "max_iterations": 180,
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }


def check(result: Any, oracle: dict, *, fast: bool) -> None:
    """Independent complete E/F gates, stricter than the default application budget."""
    assert result.converged and result.executed_backend == "cuda"
    assert abs(result.energy - oracle["energy"]) < 1e-7
    delta = result.forces - np.asarray(oracle["forces"])
    assert np.max(np.abs(delta)) < (1e-5 if fast else 1e-6)
    assert np.sqrt(np.mean(delta**2)) < 3e-6
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
    if fast:
        report = result.precision_validation
        assert report.status == "qualified_mixed_gradient"
        assert (
            report.solve_count == 1 and report.solves[0].requested_precision == "fp64"
        )
        assert report.gradient_admission == "wb97m-v-rks-budget-fp32-v1"
        work = result.precision["gradient"]
        assert work["owner"] == "python-composite-stationary-cuda"
        assert work["returned_fp32_geometry_points"] == oracle["grid_points"]
        assert work["attempted_fp32_geometry_points"] == oracle["grid_points"]
        assert work["attempted_fp32_grid_points"] == 0
        assert work["fp64_gradient_retries"] == 0
        assert "vv10-pairs-and-nonlocal-geometry" in work["strict_components"]


@pytest.mark.parametrize(
    "case_name",
    (
        CASE,
        "water-tetramer-def2-svp-spherical",
        "ammonia-def2-svp-spherical",
        "water-def2-svp",
    ),
)
def test_public_complete_singlepoint_frozen_and_moved(case_name: str) -> None:
    """Public selection, real resident seeds and moving grids; no diagnostic flag."""
    pytest.importorskip("pyscf")
    case = benchmark_cases()[case_name]
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in case.atoms]
    moved[1][1][0] += 0.025
    expected = [
        independent_reference(case_name, atoms) for atoms in (case.atoms, moved)
    ]
    calculators = [
        Calculator(precision="fp64", **options(case_name)),
        PrecisionCalculator(policy="fast", **options(case_name)),
    ]
    try:
        for fast, calculator in zip((False, True), calculators, strict=True):
            with no_cpu_derivatives():
                check(calculator.singlepoint(case.atoms), expected[0], fast=fast)
                calculator.clear_cache()
                with calculator.prepare_batch([case.atoms]) as batch:
                    for geometry, oracle in zip(
                        (None, [np.array([position for _, position in moved])]),
                        expected,
                        strict=True,
                    ):
                        batch.set_warm_start_updates(True)
                        check(
                            batch.execute(geometry, strict=True).items[0],
                            oracle,
                            fast=fast,
                        )
                        batch.set_warm_start_updates(False)
                        for _ in range(2):
                            result = batch.execute(geometry, strict=True).items[0]
                            check(result, oracle, fast=fast)
                            assert result.warm_start_used
    finally:
        for calculator in calculators:
            calculator.clear_cache()


@pytest.mark.parametrize("explicit", (False, True))
def test_public_budget_rejection_and_explicit_audit(explicit: bool) -> None:
    policy = (
        PrecisionPolicy(
            "fast",
            target=TargetAccuracy(
                (
                    ObservableTarget("energy", "absolute", "Eh", absolute=1e-6),
                    ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-5),
                )
            ),
        )
        if explicit
        else PrecisionPolicy("fast", budget=TargetErrorBudget(1e-7, 1e-6))
    )
    calculator = PrecisionCalculator(policy=policy, **options())
    try:
        result = calculator.singlepoint(benchmark_cases()[CASE].atoms)
        assert result.precision_validation.status == "observed_met"
        assert result.precision_validation.solve_count == 2
        gradient = (result.precision or {}).get("gradient")
        assert (
            bool(gradient and gradient.get("attempted_fp32_geometry_points"))
            == explicit
        )
    finally:
        calculator.clear_cache()


@pytest.mark.parametrize("during_construction", (False, True))
def test_missing_optional_capability_audits_strict(
    during_construction: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capability absence differs from arithmetic failure, including before a lease."""
    symbol = (
        "__init__" if during_construction else "geometry_molecular_resident_weights"
    )
    original = getattr(runtime._CudaSources, symbol)

    def missing(owner: Any, *args: Any, **kwargs: Any) -> Any:
        selected = (
            kwargs.get("fp32_geometry")
            if during_construction
            else owner.fp32_geometry_requested
        )
        if selected:
            raise runtime._Fp32GeometryUnavailable(
                "injected optional composite transport"
            )
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(runtime._CudaSources, symbol, missing)
    calculator = PrecisionCalculator(policy="fast", **options())
    try:
        result = calculator.singlepoint(benchmark_cases()[CASE].atoms)
        report = result.precision_validation
        assert report.status == "observed_met" and report.solve_count == 2
        assert "fp32_geometry_capability_unavailable" in report.fallback_reasons
        work = result.precision["gradient"]
        assert (
            work["returned_mode"] == "fp64"
            and work["attempted_fp32_geometry_points"] == 0
        )
        assert work["fp64_gradient_retries"] == 0
    finally:
        calculator.clear_cache()


def test_retry_failed_peer_and_energy_evidence_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject real pending mixed work, drain, and never propagate it to a peer."""
    pytest.importorskip("pyscf")
    case = benchmark_cases()[CASE]
    expected = independent_reference(CASE)
    original = runtime._CudaSources._call
    injected = False

    def call(owner: Any, symbol: str, *args: Any) -> Any:
        nonlocal injected
        if (
            owner.fp32_geometry_requested
            and symbol == "stationary_finish_span"
            and not injected
        ):
            assert owner.metrics()["fp32_geometry_points"] > 0
            injected = True
            raise ArithmeticError("injected after enqueued composite geometry")
        return original(owner, symbol, *args)

    monkeypatch.setattr(runtime._CudaSources, "_call", call)
    calculator = PrecisionCalculator(policy="fast", **options())
    try:
        with calculator.prepare_batch([case.atoms, case.atoms]) as batch:
            failed = batch.execute([[0.0], None], strict=False)
            assert failed.failure_indices == (0,)
            assert not (failed.items[0].precision or {}).get("gradient")
            result = failed.items[1]
            assert (
                injected
                and result.precision_validation.status == "qualified_strict_final"
            )
            work = result.precision["gradient"]
            assert work["fp64_gradient_retries"] == 1
            assert (
                work["attempted_fp32_geometry_points"] > 0
                and work["returned_fp32_geometry_points"] == 0
            )
            check(result, expected, fast=False)
            for result in batch.execute(strict=True).items:
                check(result, expected, fast=True)
            energy = batch.execute(properties=("energy",), strict=True)
            for result in energy.items:
                assert result.forces is None and not (result.precision or {}).get(
                    "gradient"
                )
                assert result.precision_validation.solve_count == 2
                assert result.precision_validation.status == "observed_met"
    finally:
        calculator.clear_cache()

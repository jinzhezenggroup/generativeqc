"""Public DF Fast selects real arithmetic, with independent complete-model gates."""

from __future__ import annotations

import os
from dataclasses import replace
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
from test_df_fast_candidate_cuda import auxiliary_bases as spherical_auxiliary_bases

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
GRID = GridSpec(48, 16, 32)


def auxiliary_bases(case: Any) -> tuple[Any, Any]:
    """Select Cartesian/spherical auxiliary functions from the same pinned data.

    The retained file describes spherical functions. Cartesian qualification
    deliberately changes that representation in both engines, not its exponents
    or contractions. PySCF inherits the representation from its molecule.
    """
    native, reference_basis = spherical_auxiliary_bases(
        replace(case, basis_representation="spherical")
    )
    return replace(native, representation=case.basis_representation), reference_basis


def options(case_name: str, functional: str, grid: GridSpec = GRID) -> dict:
    """Identical scientific inputs and stopping criteria for both endpoints."""
    case = benchmark_cases()[case_name]
    auxiliary, _ = auxiliary_bases(case)
    return {
        "method": f"{functional.lower()}-rks",
        "basis": case.generativeqc_basis,
        "basis_representation": case.basis_representation,
        "auxiliary_basis": auxiliary,
        "density_fitting": "cuda",
        "ks_options": KsOptions(grid=grid),
        "device": "cuda",
        "initial_guess": None,
        "max_iterations": 180,
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }


def test_public_fitted_fp64_fast_shares_standard_execution() -> None:
    """A nonzero budget does not change an otherwise identical FP64 owner.

    Default stationarity controls do not qualify mixed geometry for this DF
    hybrid. Compare complete cold, frozen and moved endpoints to an independent
    moving-grid oracle; neither facade may obtain a reference density.
    """
    pytest.importorskip("pyscf")
    from test_dft_complete_cuda import no_cpu_derivatives

    case_name = "water-tetramer-def2-svp-spherical"
    case = benchmark_cases()[case_name]
    grid = GridSpec(24, 8, 16)
    _, auxiliary = auxiliary_bases(case)
    shared = options(case_name, "PBE0", grid)
    shared.update(energy_tolerance=1e-9, density_tolerance=1e-5)
    moved = tuple(
        (
            symbol,
            tuple(float(value) + (0.001 if index == 0 else 0.0) for value in position),
        )
        for index, (symbol, position) in enumerate(case.atoms)
    )
    references = [
        reference(
            case_name,
            grid,
            1,
            functional="PBE0",
            atoms=geometry,
            auxiliary_basis=auxiliary,
        )
        for geometry in (case.atoms, moved)
    ]
    observed: dict[str, list[Any]] = {}
    for policy in ("standard", "fast"):
        calculator = PrecisionCalculator(**shared, policy=policy)
        assert calculator._candidate_precision == "fp64"
        observed[policy] = []
        try:
            with calculator.prepare_batch([case.atoms], warm_start=True) as prepared:
                for geometry_index, geometry in (
                    (0, case.atoms),
                    (0, case.atoms),
                    (1, moved),
                ):
                    with no_cpu_derivatives():
                        endpoint = prepared.execute(
                            [np.asarray([position for _, position in geometry])],
                            strict=True,
                            properties=("energy", "forces"),
                        ).items[0]
                    assert endpoint.succeeded and endpoint.converged
                    assert endpoint.executed_backend == "cuda"
                    oracle = references[geometry_index]
                    assert abs(endpoint.energy - oracle["energy"]) < 1e-7
                    np.testing.assert_allclose(
                        endpoint.forces, oracle["forces"], atol=1e-6, rtol=0
                    )
                    assert endpoint.precision["effective_bits"] == 64
                    assert not endpoint.precision["mixed_stage_fock_builds"]
                    gradient = endpoint.precision.get("gradient") or {}
                    assert not gradient.get("returned_fp32_geometry_points", 0)
                    observed[policy].append(endpoint)
                    if len(observed[policy]) == 1:
                        prepared.set_warm_start_updates(False)
        finally:
            calculator.clear_cache()
    for standard, fast in zip(observed["standard"], observed["fast"], strict=True):
        assert abs(standard.energy - fast.energy) < 1e-10
        np.testing.assert_allclose(standard.forces, fast.forces, atol=1e-9, rtol=0)


@pytest.mark.parametrize("functional", ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X"))
@pytest.mark.parametrize("force_python", (False, True), ids=("selected", "python"))
@pytest.mark.parametrize(
    "case_name,grid",
    (
        ("water-def2-svp-spherical", GridSpec(24, 8, 16)),
        ("water-def2-svp", GRID),
        ("ammonia-def2-svp-spherical", GRID),
        ("water-tetramer-def2-svp-spherical", GRID),
    ),
)
def test_public_fitted_fast_selects_complete_model(
    case_name: str,
    grid: GridSpec,
    functional: str,
    force_python: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both selected-native and packaged fallback consumers retain oracle gates."""
    pytest.importorskip("pyscf")
    from test_dft_complete_cuda import no_cpu_derivatives

    if force_python:
        from generativeqc.batch import PreparedBatch

        monkeypatch.setattr(
            PreparedBatch, "_native_dft_force_eligible", lambda *arguments: False
        )
    case = benchmark_cases()[case_name]
    _, auxiliary = auxiliary_bases(case)
    oracle = reference(
        case_name, grid, 4, functional=functional, auxiliary_basis=auxiliary
    )
    strict = Calculator(**options(case_name, functional, grid), precision="fp64")
    fast = PrecisionCalculator(**options(case_name, functional, grid), policy="fast")
    try:
        with no_cpu_derivatives():
            baseline = strict.singlepoint(case.atoms)
            strict.clear_cache()
            candidate = fast.singlepoint(case.atoms)
        report = candidate.precision_validation
        assert report.status == "qualified_mixed_gradient"
        assert report.solve_count == 1
        assert report.solves[0].requested_precision == "fp64"
        assert "fp32_grid_storage_unavailable" not in report.fallback_reasons
        assert abs(candidate.energy - oracle["energy"]) < 1e-7
        assert abs(baseline.energy - oracle["energy"]) < 1e-7
        np.testing.assert_allclose(baseline.forces, oracle["forces"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            candidate.forces, oracle["forces"], atol=1e-5, rtol=0
        )
        assert np.sqrt(np.mean((candidate.forces - baseline.forces) ** 2)) < 3e-6
        np.testing.assert_allclose(candidate.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
        work = candidate.precision["gradient"]
        assert work["attempted_fp32_geometry_points"] == oracle["grid_points"]
        assert work["fp64_gradient_retries"] == 0
        if work["schema_version"] == 3 and not force_python:
            # The native owner qualifies weighted/point arithmetic as well as
            # geometry; its v3 receipt must not be relabeled as Python DF v4.
            assert (
                report.gradient_admission == f"{functional.lower()}-rks-budget-fp32-v1"
            )
            assert work["qualification"] == report.gradient_admission
            assert work["complete"] and work["returned_mode"] == "mixed"
            assert work["attempted_fp32_grid_points"] == oracle["grid_points"]
            assert {
                "final_scf",
                "one_electron_and_pulay",
                "global_reductions",
            } <= set(work["fp64_components"])
        else:
            assert work["schema_version"] == 4
            assert (
                report.gradient_admission
                == f"{functional.lower()}-df-rks-budget-fp32-v1"
            )
            assert work["returned_fp32_geometry_points"] == oracle["grid_points"]
            assert work["attempted_fp32_grid_points"] == 0
            assert "df-integrals-and-metric-response" in work["strict_components"]
    finally:
        strict.clear_cache()
        fast.clear_cache()


@pytest.mark.parametrize("explicit", (False, True))
def test_public_fitted_tight_budget_and_explicit_target_retain_audit(
    explicit: bool,
) -> None:
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
    case_name = "water-def2-svp-spherical"
    calculator = PrecisionCalculator(**options(case_name, "PBE0"), policy=policy)
    try:
        result = calculator.singlepoint(benchmark_cases()[case_name].atoms)
        assert result.precision_validation.solve_count == 2
        assert result.precision_validation.status == "observed_met"
        assert result.precision_validation.errors
        gradient = (result.precision or {}).get("gradient")
        if explicit:
            assert gradient["attempted_fp32_geometry_points"] > 0
        else:
            assert gradient is None or not gradient.get(
                "attempted_fp32_geometry_points", 0
            )
    finally:
        calculator.clear_cache()


def test_public_fitted_missing_capability_audits_strict_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional source cannot publish an invented mixed-work count."""
    case_name = "water-def2-svp-spherical"
    case = benchmark_cases()[case_name]
    calculator = PrecisionCalculator(**options(case_name, "PBE0"), policy="fast")
    original = runtime._CudaSources.geometry_molecular_resident_weights

    def unavailable(owner: Any, *args: Any, **kwargs: Any) -> Any:
        if owner.fp32_geometry_requested:
            raise runtime._Fp32GeometryUnavailable("injected absent mixed transport")
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(
        runtime._CudaSources, "geometry_molecular_resident_weights", unavailable
    )
    try:
        result = calculator.singlepoint(case.atoms)
        report = result.precision_validation
        assert report.status == "observed_met" and report.solve_count == 2
        assert "fp32_geometry_capability_unavailable" in report.fallback_reasons
        work = result.precision["gradient"]
        assert work["returned_mode"] == "fp64"
        assert work["attempted_fp32_geometry_points"] == 0
        assert work["fp64_gradient_retries"] == 0
    finally:
        calculator.clear_cache()


def test_public_fitted_retry_and_batch_evidence_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public retry and underlying per-item records use real pending CUDA work.

    The facade still exposes singlepoints only. Its retained native candidate is
    used separately to exercise item isolation and energy-only evidence expiry.
    """
    pytest.importorskip("pyscf")
    case_name = "water-def2-svp-spherical"
    case = benchmark_cases()[case_name]
    _, auxiliary = auxiliary_bases(case)
    oracle = reference(case_name, GRID, 4, auxiliary_basis=auxiliary)
    calculator = PrecisionCalculator(**options(case_name, "PBE0"), policy="fast")
    original_call = runtime._CudaSources._call
    injected = False

    def call(owner: Any, symbol: str, *args: Any) -> Any:
        nonlocal injected
        if (
            owner.fp32_geometry_requested
            and symbol == "stationary_finish"
            and not injected
        ):
            assert owner.metrics()["fp32_geometry_points"] > 0
            injected = True
            raise ArithmeticError("injected after public DF mixed submission")
        return original_call(owner, symbol, *args)

    monkeypatch.setattr(runtime._CudaSources, "_call", call)
    try:
        recovered = calculator.singlepoint(case.atoms)
        assert injected
        assert recovered.precision_validation.solve_count == 1
        assert recovered.precision_validation.status == "qualified_strict_final"
        work = recovered.precision["gradient"]
        assert work["fp64_gradient_retries"] == 1
        assert work["attempted_fp32_geometry_points"] > 0
        assert work["returned_fp32_geometry_points"] == 0
        np.testing.assert_allclose(
            recovered.forces, oracle["forces"], atol=1e-6, rtol=0
        )
        with calculator._candidate.prepare_batch([case.atoms, case.atoms]) as batch:
            failed = batch.execute([[0.0], None], strict=False)
            assert failed.failure_indices == (0,)
            assert not (failed.items[0].precision or {}).get("gradient")
            assert failed.items[1].precision["gradient"]["returned_mode"] == "mixed"
            mixed = batch.execute(strict=True)
            for item in mixed.items:
                assert item.precision["gradient"]["returned_mode"] == "mixed"
                np.testing.assert_allclose(
                    item.forces, oracle["forces"], atol=1e-5, rtol=0
                )
            energy = batch.execute(properties=("energy",), strict=True)
            for item in energy.items:
                assert item.forces is None
                assert not (item.precision or {}).get("gradient")
            assert mixed.items[1].precision["gradient"]["returned_mode"] == "mixed"
        replay = calculator.singlepoint(case.atoms, properties=("energy",))
        assert replay.forces is None
        assert not (replay.precision or {}).get("gradient")
        assert replay.precision_validation.solve_count == 2
        assert replay.precision_validation.status == "observed_met"
    finally:
        calculator.clear_cache()

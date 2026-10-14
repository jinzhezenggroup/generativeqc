"""Independent DF Hamiltonian gates for the private typed-geometry candidate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc import _stationary_cuda as runtime

from benchmarks._cases import benchmark_cases
from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
ROOT = Path(__file__).resolve().parents[2]
GRID = GridSpec(24, 8, 16)


def auxiliary_bases(case: Any) -> tuple[Any, Any]:
    """Use identical pinned inputs, not either engine's default auxiliary basis."""
    return load_retained_comparison_basis(
        ROOT
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json",
        case,
        role="auxiliary",
        compute_forces=True,
    )


@pytest.mark.parametrize("functional", ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X"))
@pytest.mark.parametrize(
    "case_name", ("water-def2-svp-spherical", "water-tetramer-def2-svp-spherical")
)
def test_fitted_candidate_keeps_metric_response_and_moving_grid(
    functional: str, case_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise exact fitted-model oracles, frozen seeds, movement and strict replay.

    Neither direct-ERI forces nor native strict forces are the independent
    oracle. PySCF constructs its own fitted metric and auxiliary response.
    """
    pytest.importorskip("pyscf")
    from test_dft_complete_cuda import no_cpu_derivatives

    case = benchmark_cases()[case_name]
    auxiliary, reference_auxiliary = auxiliary_bases(case)
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in case.atoms]
    moved[1][1][0] += 0.025
    oracles = [
        reference(
            case_name,
            GRID,
            4,
            functional=functional,
            atoms=atoms,
            auxiliary_basis=reference_auxiliary,
        )
        for atoms in (case.atoms, moved)
    ]
    selected = False
    records = []
    original = runtime.complete_rks_cuda_gradient_diagnostic

    def execute(*args: Any, **kwargs: Any) -> Any:
        kwargs["fp32_semilocal_geometry"] = selected
        result = original(*args, **kwargs)
        records.append(result.work)
        return result

    monkeypatch.setattr(runtime, "complete_rks_cuda_gradient_diagnostic", execute)
    calculator = Calculator(
        method=f"{functional.lower()}-rks",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        auxiliary_basis=auxiliary,
        density_fitting="cuda",
        device="cuda",
        precision="fp64",
        ks_options=KsOptions(grid=GRID),
        initial_guess=None,
        max_iterations=180,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
    )
    try:
        with calculator.prepare_batch([case.atoms], warm_start=True) as batch:
            for geometry, oracle in zip(
                (None, [np.array([position for _, position in moved])]),
                oracles,
                strict=True,
            ):
                selected = False
                batch.set_warm_start_updates(True)
                with no_cpu_derivatives():
                    strict = batch.execute(geometry, strict=True).items[0]
                assert abs(strict.energy - oracle["energy"]) < 1e-7
                np.testing.assert_allclose(
                    strict.forces, oracle["forces"], atol=1e-6, rtol=0
                )
                assert records[-1]["fp32_geometry_points"] == 0
                selected = True
                batch.set_warm_start_updates(False)
                for _ in range(2):
                    with no_cpu_derivatives():
                        candidate = batch.execute(geometry, strict=True).items[0]
                    assert abs(candidate.energy - oracle["energy"]) < 1e-7
                    np.testing.assert_allclose(
                        candidate.forces, oracle["forces"], atol=1e-5, rtol=0
                    )
                    assert (
                        np.sqrt(np.mean((candidate.forces - strict.forces) ** 2)) < 3e-6
                    )
                    np.testing.assert_allclose(
                        candidate.forces.sum(axis=0), 0, atol=2e-8, rtol=0
                    )
                    work = records[-1]
                    assert work["fp32_semilocal_geometry_requested"]
                    assert work["fp32_geometry_points"] == oracle["grid_points"]
                selected = False
                with no_cpu_derivatives():
                    replay = batch.execute(geometry, strict=True).items[0]
                np.testing.assert_allclose(
                    replay.forces, strict.forces, atol=1e-7, rtol=0
                )
    finally:
        calculator.clear_cache()


def test_fitted_pending_mixed_work_retries_without_contaminating_neighbor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain real pending streams before strict retry and subsequent mixed replay."""
    pytest.importorskip("pyscf")
    case_name = "water-def2-svp-spherical"
    case = benchmark_cases()[case_name]
    auxiliary, reference_auxiliary = auxiliary_bases(case)
    oracle = reference(case_name, GRID, 4, auxiliary_basis=reference_auxiliary)
    original_execute = runtime.complete_rks_cuda_gradient_diagnostic
    original_call = runtime._CudaSources._call
    injected = False
    records = []

    def call(owner: Any, symbol: str, *args: Any) -> Any:
        nonlocal injected
        if (
            owner.fp32_geometry_requested
            and symbol
            in (
                "stationary_finish",
                "stationary_finish_span",
                "stationary_finish_reduced",
            )
            and not injected
        ):
            assert owner.metrics()["fp32_geometry_points"] > 0
            injected = True
            raise ArithmeticError("injected after actual mixed DF geometry")
        return original_call(owner, symbol, *args)

    def execute(*args: Any, **kwargs: Any) -> Any:
        kwargs["fp32_semilocal_geometry"] = True
        result = original_execute(*args, **kwargs)
        records.append(result.work)
        return result

    monkeypatch.setattr(runtime._CudaSources, "_call", call)
    monkeypatch.setattr(runtime, "complete_rks_cuda_gradient_diagnostic", execute)
    calculator = Calculator(
        method="pbe0-rks",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        auxiliary_basis=auxiliary,
        density_fitting="cuda",
        device="cuda",
        precision="fp64",
        ks_options=KsOptions(grid=GRID),
        initial_guess=None,
        max_iterations=180,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
    )
    try:
        with calculator.prepare_batch([case.atoms, case.atoms]) as batch:
            result = batch.execute([[0.0], None], strict=False)
            assert result.failure_indices == (0,), tuple(
                item.status_message for item in result.items
            )
            assert result.items[0].forces is None
            assert injected and len(records) == 1
            work = records[0]
            assert work["fp64_geometry_retries"] == 1
            assert work["rejected_semilocal_work"]["fp32_geometry_points"] > 0
            assert work["fp32_geometry_points"] == 0
            np.testing.assert_allclose(
                result.items[1].forces, oracle["forces"], atol=1e-6, rtol=0
            )
            result = batch.execute(strict=True)
            for item, work in zip(result.items, records[-2:], strict=True):
                assert work["fp64_geometry_retries"] == 0
                assert work["fp32_geometry_points"] == oracle["grid_points"]
                np.testing.assert_allclose(
                    item.forces, oracle["forces"], atol=1e-5, rtol=0
                )
    finally:
        calculator.clear_cache()

"""Qualify the method-owned DF capacity crossover, not an explicit opt-in."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._ks_snapshot import NativeKsSnapshot

from benchmarks._cases import benchmark_cases
from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.df_component_ledger import read_trace
from benchmarks.readme_wb97mv import reference_engine

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a scheduler-allocated native CUDA library/device",
)
ROOT = Path(__file__).resolve().parents[2]


def test_restricted_df_auto_retains_values_and_preserves_force_oracle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bounded crossover must remove SCF regeneration without changing E/F."""
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("gpu4pyscf")
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    basis, reference_basis = load_comparison_basis(
        ROOT / "benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json",
        case,
        role="orbital",
        compute_forces=True,
    )
    auxiliary, reference_auxiliary = load_retained_comparison_basis(
        ROOT
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json",
        case,
        role="auxiliary",
        compute_forces=True,
    )
    grid = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    reference = reference_engine(case.atoms, reference_basis, grid, xc="PBE0")
    reference = reference.density_fit(auxbasis=reference_auxiliary)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-10
    reference.direct_scf_tol = 1e-14
    reference.max_cycle = 200
    reference_energy = float(reference.kernel())
    assert reference.converged
    gradient = reference.nuc_grad_method()
    gradient.grid_response = True
    reference_forces = cupy.asnumpy(-gradient.kernel())

    ordinary_derivatives = NativeKsSnapshot.density_fitted_integral_derivatives
    for storage in ("auto", "dense", "auto-host-response"):
        trace_path = tmp_path / f"{storage}.jsonl"
        progress_path = tmp_path / f"{storage}-progress.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", storage.split("-")[0])
        monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace_path))
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(progress_path))
        if storage == "auto-host-response":

            def bounded_derivatives(
                self: NativeKsSnapshot, atom_count: int, maximum_bytes: int
            ) -> tuple[np.ndarray, dict[str, int]]:
                """Reject optional device staging, not the independent DF owner."""
                return ordinary_derivatives(self, atom_count, min(maximum_bytes, 4096))

            monkeypatch.setattr(
                NativeKsSnapshot,
                "density_fitted_integral_derivatives",
                bounded_derivatives,
            )
        calculator = Calculator(
            method="pbe0-rks",
            basis=basis,
            auxiliary_basis=auxiliary,
            basis_representation="spherical",
            device="cuda",
            density_fitting="cuda",
            density_fitting_memory_budget_bytes=112 << 20,
            ks_options=KsOptions(grid=grid),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=200,
        )
        with calculator.prepare_batch([case.atoms], warm_start=True) as batch:
            cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            metrics = batch.last_density_fitting_metric_diagnostics()
            cold_rows = read_trace(trace_path)
            assert metrics and all(
                entry.streamed == (storage == "dense") for entry in metrics
            )
            assert all(entry.peak_device_bytes <= 112 << 20 for entry in metrics)
            assert cold.converged and not cold.warm_start_used
            assert cold.physical_residual_rms < 1e-9
            assert abs(cold.energy - reference_energy) <= 1e-8
            np.testing.assert_allclose(cold.forces, reference_forces, atol=3e-7, rtol=0)
            fock_rows = [
                row
                for row in cold_rows
                if row["operation"] in ("ri_j", "ri_k", "ri_k_occupied")
            ]
            assert fock_rows
            regeneration = sum(
                row["counters"].get("raw_panel_source_auxiliary_evaluations", 0)
                for row in fock_rows
            )
            if storage != "dense":
                assert regeneration == 0
                materialization = [
                    row
                    for row in cold_rows
                    if row["operation"] == "resident_three_center_materialization"
                ]
                assert len(materialization) == 1
                assert (
                    materialization[0]["counters"]["value_packed_pairs"] == 96 * 97 // 2
                )
            else:
                assert regeneration > 0
            replay = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
            assert replay.converged and replay.warm_start_used
            assert abs(replay.energy - reference_energy) <= 1e-8
            np.testing.assert_allclose(
                replay.forces, reference_forces, atol=3e-7, rtol=0
            )
            progress = [
                json.loads(line) for line in progress_path.read_text().splitlines()
            ]
            deferred = [
                row
                for row in progress
                if row["name"] == "deferred_one_electron_derivatives"
                and row["event"] == "BEGIN"
            ]
            assert len(deferred) == int(storage == "auto-host-response")

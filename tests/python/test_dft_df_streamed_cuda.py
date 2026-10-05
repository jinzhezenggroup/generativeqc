"""Real constrained-memory DF-KS replay against independently converged PySCF."""

import os
from typing import Any

import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from test_dft_df_public import WATER, pyscf_energy

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize(
    "method,functional", [("pbe-rks", "PBE"), ("pbe0-rks", "PBE0")]
)
def test_streamed_dft_cold_warm_and_geometry(
    method: str, functional: str, monkeypatch: Any
) -> None:
    """A bounded source, including its auxiliary tail, stays streamed on replay."""
    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    atoms = [
        (symbol, (x + 7 * copy, y, z))
        for copy in range(4)
        for symbol, (x, y, z) in WATER
    ]
    moved = [
        (symbol, (x + (0.02 if i == 1 else 0), y, z))
        for i, (symbol, (x, y, z)) in enumerate(atoms)
    ]
    calc = Calculator(
        method=method,
        basis="sto-3g",
        auxiliary_basis="def2-svp",
        basis_representation="cartesian",
        device="cuda",
        precision="fp64",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=8704 * 1024,
        ks_options=KsOptions(grid=GridSpec()),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    # The 12-atom reference deliberately exports 294,912 grid points. Bound
    # this independent host oracle explicitly without changing the native grid.
    expected = pyscf_energy(calc, atoms, 1, functional, max_grid_points=400_000)
    changed_expected = pyscf_energy(calc, moved, 1, functional, max_grid_points=400_000)
    with calc.prepare_batch([atoms]) as batch:
        for geometry, oracle in (
            (None, expected),
            (None, expected),
            (moved, changed_expected),
        ):
            coordinates = None if geometry is None else [[xyz for _, xyz in geometry]]
            result = batch.execute(
                coordinates, properties=("energy",), strict=True
            ).items[0]
            assert result.executed_backend == "cuda"
            assert abs(result.energy - oracle) < 1e-8
            assert result.physical_residual_rms < 1e-9
            diagnostics = batch.last_density_fitting_metric_diagnostics()
            assert diagnostics and all(item.streamed for item in diagnostics)
            # The native binding gate separately covers full/truncated metric
            # rank; this physical basis must keep all 100 auxiliary directions.
            assert all(item.effective_rank == 100 for item in diagnostics)

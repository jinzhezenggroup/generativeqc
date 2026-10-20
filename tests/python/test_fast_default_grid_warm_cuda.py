"""Default-grid Fast warm reuse against independent moving-grid observables."""

from __future__ import annotations

import os

import numpy as np
import pytest
from generativeqc import PrecisionCalculator

from benchmarks.readme_hf_scaling import scaling_cases

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)


def oracle(atoms: tuple, functional: str) -> tuple[float, np.ndarray]:
    """Independent CPU quadrature and full analytic moving-grid response."""
    from pyscf import dft, gto

    molecule = gto.M(atom=atoms, basis="def2-svp", unit="Bohr", verbose=0)
    engine = dft.RKS(molecule)
    engine.xc = functional
    engine.grids.level = 4
    engine.conv_tol = 1e-11
    engine.conv_tol_grad = 1e-7
    engine.direct_scf_tol = 1e-14
    energy = float(engine.kernel())
    assert engine.converged
    gradient = engine.nuc_grad_method()
    gradient.grid_response = True
    return energy, -gradient.kernel()


@pytest.mark.parametrize("functional", ("PBE", "PBE0", "R2SCAN", "M06-2X"))
def test_default_fast_frozen_and_moved_physical_endpoints(functional: str) -> None:
    """A strict warm replay retains E/F accuracy and cold/moved work remains live."""
    atoms = scaling_cases()["water-12"].atoms
    moved = tuple(
        (
            symbol,
            tuple(
                np.asarray(position) + ([0.003, -0.002, 0.001] if i == 1 else [0, 0, 0])
            ),
        )
        for i, (symbol, position) in enumerate(atoms)
    )
    calculator = PrecisionCalculator(
        policy="fast",
        method=functional.lower() + "-rks",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
    )
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as prepared:
            for geometry_index, geometry in enumerate((atoms, moved)):
                energy, forces = oracle(geometry, functional)
                prepared.set_warm_start_updates(True)
                coordinates = (
                    None
                    if geometry_index == 0
                    else [np.asarray([p for _, p in geometry])]
                )
                for replay in range(3):
                    batch = prepared.execute(coordinates, strict=True)
                    observed = batch.items[0]
                    assert abs(observed.energy - energy) / len(atoms) < 1e-5
                    difference = observed.forces - forces
                    assert np.max(np.abs(difference)) < 1e-4
                    assert np.sqrt(np.mean(difference**2)) < 3e-5
                    assert observed.precision_validation.solve_count == 1
                    if replay:
                        assert observed.warm_start_used
                        assert observed.precision["mixed_stage_fock_builds"] == 0
                        assert observed.precision["effective_bits"] == 64
                        assert observed.precision["final_residual_audits"] > 0
                        assert (
                            observed.precision_validation.status
                            == "qualified_budget_fock_strict_warm"
                        )
                    else:
                        assert observed.precision["mixed_stage_fock_builds"] > 0
                        prepared.set_warm_start_updates(False)
                    # None always means the preparation geometry, not the last
                    # explicit geometry. Repeat the moved coordinates so this
                    # assertion compares forces for the same physical system.
    finally:
        calculator.clear_cache()

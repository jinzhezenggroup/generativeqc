"""Complete d/f CCSD(T) forces after native quartet-based RHF preparation."""

from __future__ import annotations

import os
from functools import cache

import numpy as np
import pytest
from generativeqc import Calculator
from generativeqc.calculator import Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RCCSDT_CUDA_TEST") != "1",
    reason="requires a finite Slurm CUDA allocation",
)


@pytest.mark.parametrize("angular", (2, 3))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_angular_reference_complete_force_matches_independent_fci(
    angular: int, representation: str
) -> None:
    """Two electrons make CCSD exact; independent FCI differences check forces.

    These small spherical bases still require the Cartesian quartet transform,
    despite lying below ordinary HF's small-system ERI-cache threshold.
    """
    pyscf = pytest.importorskip("pyscf")
    fci = pytest.importorskip("pyscf.fci")
    pool = pytest.importorskip("threadpoolctl")
    assert os.environ.get("SLURM_JOB_ID"), "real CUDA qualification requires Slurm"
    atoms = [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, angular, (Primitive(0.8, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    calculator = Calculator(
        method="ccsd(t)",
        basis=basis,
        basis_representation=representation,
        device="cuda",
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=150,
        ccsd_energy_tolerance=1e-13,
        ccsd_residual_tolerance=1e-11,
    )
    actual = calculator.singlepoint(atoms, charge=1, properties=("energy", "forces"))
    with calculator.prepare_batch([atoms], charges=[1]) as prepared:
        cold = prepared.execute(properties=("energy", "forces"), strict=True).items[0]
        warm = prepared.execute(properties=("energy", "forces"), strict=True).items[0]
        changed = prepared.execute(
            coordinates=[[(0.0, 0.0, -0.7), (0.0, 0.0, 0.71)]],
            properties=("energy", "forces"),
            strict=True,
        ).items[0]

    @cache
    def independent_energy(displacement: float) -> float:
        mol = pyscf.gto.M(
            atom=[atoms[0], ("H", (0.0, 0.0, 0.7 + displacement))],
            basis={
                "He": [[0, [1.5, 1.0]], [angular, [0.8, 1.0]]],
                "H": [[0, [1.2, 1.0]]],
            },
            charge=1,
            spin=0,
            unit="Bohr",
            cart=representation == "cartesian",
            verbose=0,
        )
        mf = mol.RHF()
        mf.conv_tol = 1e-13
        mf.kernel()
        assert mf.converged
        solver = fci.FCI(mf)
        solver.conv_tol = 1e-13
        energy, _ = solver.kernel()
        assert solver.converged
        return float(energy)

    for result, displacement in (
        (actual, 0.0),
        (cold, 0.0),
        (warm, 0.0),
        (changed, 0.01),
    ):
        assert result.converged and result.forces is not None
        assert result.correlation is not None
        assert result.correlation.force_provenance_flags & 0x8
        assert result.correlation.response_absolute_residual < 1e-9
        with pool.threadpool_limits(limits=1):
            assert result.energy == pytest.approx(
                independent_energy(displacement), abs=3e-10
            )
            step = 1e-4
            derivative = (
                independent_energy(displacement + step)
                - independent_energy(displacement - step)
            ) / (2 * step)
        assert result.forces[1, 2] == pytest.approx(-derivative, abs=3e-7)
        np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=2e-9, rtol=0)

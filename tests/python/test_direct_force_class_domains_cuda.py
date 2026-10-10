"""Independent FCI force qualification of mixed-basis class-domain scheduling."""

import os

import numpy as np
import pytest
from generativeqc import Calculator
from generativeqc.calculator import Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RCCSDT_CUDA_TEST") != "1",
    reason="requires a finite Slurm CUDA allocation",
)


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("fallback", (False, True))
def test_mixed_spdf_complete_force_matches_independent_fci(
    representation: str, fallback: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two electrons make CCSD exact; keep all classes and check every coordinate."""
    pyscf = pytest.importorskip("pyscf")
    fci = pytest.importorskip("pyscf.fci")
    pool = pytest.importorskip("threadpoolctl")
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    if fallback:
        monkeypatch.setenv("GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES", "0")
    else:
        monkeypatch.delenv(
            "GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES", raising=False
        )
    atoms = [("He", (0.0, 0.1, -0.7)), ("H", (0.2, -0.1, 0.7))]
    basis = tuple(Shell(0, angular, (Primitive(0.8, 1.0),)) for angular in range(4)) + (
        Shell(1, 0, (Primitive(1.2, 1.0),)),
        Shell(1, 2, (Primitive(0.9, 1.0),)),
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

    def independent_energy(coordinates: np.ndarray) -> float:
        molecule = pyscf.gto.M(
            atom=[(atom, coordinates[index]) for index, (atom, _) in enumerate(atoms)],
            basis={
                "He": [[angular, [0.8, 1.0]] for angular in range(4)],
                "H": [[0, [1.2, 1.0]], [2, [0.9, 1.0]]],
            },
            charge=1,
            spin=0,
            unit="Bohr",
            cart=representation == "cartesian",
            verbose=0,
        )
        reference = molecule.RHF()
        reference.conv_tol = 1e-13
        reference.kernel()
        assert reference.converged
        solver = fci.FCI(reference)
        solver.conv_tol = 1e-13
        energy, _ = solver.kernel()
        assert solver.converged
        return float(energy)

    assert actual.converged and actual.forces is not None
    assert actual.correlation is not None
    assert actual.correlation.force_provenance_flags & 0x8
    assert actual.correlation.response_absolute_residual < 1e-9
    coordinates = np.asarray([position for _, position in atoms])
    expected = np.empty_like(coordinates)
    with pool.threadpool_limits(limits=1):
        assert actual.energy == pytest.approx(
            independent_energy(coordinates), abs=3e-10
        )
        step = 1e-4
        for atom in range(2):
            for axis in range(3):
                plus, minus = coordinates.copy(), coordinates.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                expected[atom, axis] = -(
                    independent_energy(plus) - independent_energy(minus)
                ) / (2 * step)
    np.testing.assert_allclose(actual.forces, expected, atol=3e-7, rtol=0)
    np.testing.assert_allclose(actual.forces.sum(axis=0), 0, atol=2e-9, rtol=0)

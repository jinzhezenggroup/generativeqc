"""Independent E+F gates for bounded, whole-shell exact HF Fock execution."""

from __future__ import annotations

import os

import numpy as np
import pytest
from generativeqc import Calculator
from generativeqc.calculator import Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_BOUNDED_RHF_CUDA_TEST") != "1",
    reason="requires a finite Slurm real-GPU allocation",
)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("screening", (1e-14, 1e-10))
def test_bounded_whole_shell_force_and_geometry_replay(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    representation: str,
    screening: float,
) -> None:
    """Cover f-containing classes, both spins, batch offsets and graph retirement.

    Signed primitive contractions and a non-axial geometry exercise component
    normalization and cancellation. PySCF only supplies assertions, never a
    native initial density or integral. Explicitly disabled reuse is the exact
    low-resource fallback, not a second approximate scientific model.
    """
    assert os.environ.get("SLURM_JOB_ID"), "GPU execution must use Slurm"
    pyscf = pytest.importorskip("pyscf")
    pyscf.lib.num_threads(1)
    spin = int(method == "uhf")
    charge = 1 - spin
    basis = {
        "He": [
            [0, [1.5, 1.0], [0.45, -0.18]],
            *[[angular, [1.5 - 0.2 * angular, 1.0]] for angular in range(1, 4)],
        ],
        "H": [[0, [1.2, 1.0]]],
    }
    shells = tuple(
        Shell(
            atom,
            angular,
            tuple(Primitive(exponent, coefficient) for exponent, coefficient in rows),
        )
        for atom, symbol in enumerate(("He", "H"))
        for angular, *rows in basis[symbol]
    )
    geometries = [
        [("He", (0.0, 0.0, 0.0)), ("H", (0.3, -0.2, distance))]
        for distance in (1.4, 1.6, 1.45)
    ]
    expected = []
    for geometry in geometries:
        molecule = pyscf.gto.M(
            atom=geometry,
            basis=basis,
            unit="Bohr",
            cart=representation == "cartesian",
            charge=charge,
            spin=spin,
            verbose=0,
        )
        solver = molecule.UHF() if spin else molecule.RHF()
        solver.conv_tol = 1e-13
        solver.conv_tol_grad = 1e-10
        solver.direct_scf_tol = 0
        solver.max_cycle = 150
        energy = solver.kernel()
        assert solver.converged
        expected.append((energy, -solver.nuc_grad_method().kernel()))

    monkeypatch.setenv("GENERATIVEQC_BOUNDED_DIRECT_STREAMING", "force")
    monkeypatch.setenv("GENERATIVEQC_RHF_RESIDENT_VALUES", "0")
    for reuse in ("0", "1"):
        monkeypatch.setenv("GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES", reuse)
        calculator = Calculator(
            method=method,
            basis=shells,
            basis_representation=representation,
            device="cuda",
            precision="fp64",
            max_iterations=150,
            energy_tolerance=1e-12,
            density_tolerance=1e-11,
            screening_tolerance=screening,
        )
        with calculator.prepare_batch(
            geometries[:2], charges=[charge] * 2, multiplicities=[spin + 1] * 2
        ) as prepared:
            cold = prepared.execute(properties=("energy", "forces"), strict=True)
            warm = prepared.execute(properties=("energy", "forces"), strict=True)
            moved = prepared.execute(
                coordinates=[
                    [coordinates for _, coordinates in geometry]
                    for geometry in (geometries[2], geometries[1])
                ],
                properties=("energy", "forces"),
                strict=True,
            )
        for result, indices in ((cold, (0, 1)), (warm, (0, 1)), (moved, (2, 1))):
            for item, index in zip(result.items, indices, strict=True):
                assert item.converged and item.forces is not None
                energy, forces = expected[index]
                assert item.energy == pytest.approx(energy, abs=2e-9)
                np.testing.assert_allclose(item.forces, forces, atol=2e-8, rtol=0)

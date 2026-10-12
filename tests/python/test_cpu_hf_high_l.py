"""Independent complete CPU HF gates for mixed contracted s/p/d/f bases."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, Primitive, Shell

from _cpu_hf_high_l_reference import load_reference


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize(("method", "charge", "spin"), (("rhf", 1, 0), ("uhf", 0, 1)))
def test_mixed_spdf_energy_and_forces_match_pyscf(
    representation: str, method: str, charge: int, spin: int
) -> None:
    """Check both value-only dispatch and analytic forces at a moved geometry."""
    atoms = [("He", (0.1, -0.2, -0.7)), ("H", (-0.3, 0.4, 0.8))]
    reference_basis = {
        "He": [
            [0, [1.5, 0.8], [0.4, -0.1]],
            [1, [0.7, 1.0]],
            [2, [0.8, 1.0]],
            [3, [0.6, 1.0]],
        ],
        "H": [[0, [1.2, 1.0]]],
    }
    basis = tuple(
        Shell(
            atom_index,
            record[0],
            tuple(Primitive(*primitive) for primitive in record[1:]),
        )
        for atom_index, element in enumerate(("He", "H"))
        for record in reference_basis[element]
    )
    calculator = Calculator(
        method=method,
        basis=basis,
        basis_representation=representation,
        device="cpu",
        density_fitting="none",
        initial_guess=None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    for displacement in (0.0, 0.013):
        coordinates = [(element, list(position)) for element, position in atoms]
        coordinates[-1][1][0] += displacement
        reference_energy, forces = load_reference(
            atoms=coordinates,
            basis=reference_basis,
            representation=representation,
            method=method,
            charge=charge,
            spin=spin,
            displacement=displacement,
        )
        energy = calculator.singlepoint(
            coordinates,
            charge=charge,
            multiplicity=spin + 1,
            properties=("energy",),
        )
        complete = calculator.singlepoint(
            coordinates,
            charge=charge,
            multiplicity=spin + 1,
            properties=("energy", "forces"),
        )
        _assert_results_match_reference(energy, complete, reference_energy, forces)


def _assert_results_match_reference(
    energy: Any,
    complete: Any,
    reference_energy: float,
    forces: list[list[float]],
) -> None:
    """Keep the original endpoint gates shared with explicit negative controls."""
    assert energy.converged and complete.converged
    assert energy.executed_backend == complete.executed_backend
    assert complete.executed_backend in ("cpu", "cpu_reference")
    assert energy.energy == pytest.approx(reference_energy, abs=1e-9)
    assert complete.energy == pytest.approx(reference_energy, abs=1e-9)
    np.testing.assert_allclose(complete.forces, forces, atol=1e-8, rtol=0)
    np.testing.assert_allclose(complete.forces.sum(axis=0), 0, atol=2e-10, rtol=0)

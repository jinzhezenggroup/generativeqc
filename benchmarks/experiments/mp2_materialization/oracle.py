"""Independently qualify complete CPU E+F observations with matched primitives.

Only basis inputs come from the native package. PySCF/libcint independently
constructs integrals, solves RHF/MP2 and differentiates the complete energy.
Using PySCF's rounded built-in STO-3G table would change the Hamiltonian.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyscf
from generativeqc import Atom, Calculator
from pyscf import gto, mp, scf
from pyscf.data.elements import ELEMENTS


def main() -> None:
    """Retain the independent inputs/results and check every observed endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    cases = {
        "h2": [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))],
        "lih": [(3, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 3.0))],
        "water": [
            (8, (0.0, 0.0, 0.0)),
            (1, (1.43, 0.0, 1.11)),
            (1, (-1.43, 0.0, 1.11)),
        ],
    }
    references = {}
    energy_errors, force_errors = [], []
    observations = sorted(arguments.observations.glob("endpoint-*.json")) + sorted(
        arguments.observations.glob("memory-*.endpoints.json")
    )
    if not observations:
        raise ValueError("no complete endpoint observations")
    for name, atoms in cases.items():
        labels = [
            f"{ELEMENTS[atomic_number]}{index}"
            for index, (atomic_number, _) in enumerate(atoms)
        ]
        matched_basis = {label: [] for label in labels}
        shells = Calculator(method="mp2", basis="sto-3g")._shells_for_atoms(
            tuple(Atom.from_value(atom) for atom in atoms)
        )
        for shell in shells:
            matched_basis[labels[shell.atom_index]].append(
                [shell.angular_momentum]
                + [
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ]
            )
        molecule = gto.M(
            atom=[(label, atom[1]) for label, atom in zip(labels, atoms, strict=True)],
            basis=matched_basis,
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        reference = scf.RHF(molecule)
        reference.conv_tol = 1e-13
        reference.kernel()
        assert reference.converged
        correlation = mp.MP2(reference).run()
        forces = -correlation.nuc_grad_method().kernel()
        references[name] = {
            "atoms_bohr": atoms,
            "matched_basis": matched_basis,
            "energy": correlation.e_tot,
            "forces": forces.tolist(),
        }
        for filename in observations:
            for sample in json.loads(filename.read_text())["cases"][name]["samples"]:
                assert sample["converged"], filename
                energy_error = abs(sample["energy"] - correlation.e_tot)
                force_error = float(
                    np.max(np.abs(np.asarray(sample["forces"]) - forces))
                )
                assert energy_error < 1e-9, (filename, name, energy_error)
                assert force_error < 2e-6, (filename, name, force_error)
                np.testing.assert_allclose(
                    np.sum(sample["forces"], axis=0), 0, atol=2e-8, rtol=0
                )
                energy_errors.append(energy_error)
                force_errors.append(force_error)
    arguments.output.write_text(
        json.dumps(
            {
                "schema": "generativeqc.mp2-independent-complete-endpoint.v1",
                "pyscf_version": pyscf.__version__,
                "cases": references,
                "observations": [filename.name for filename in observations],
                "complete_endpoints": len(energy_errors),
                "energy_gate": 1e-9,
                "force_gate": 2e-6,
                "max_energy_error": max(energy_errors),
                "max_force_error": max(force_errors),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

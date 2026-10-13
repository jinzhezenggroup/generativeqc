"""Generate independent PySCF E+F and displaced-energy CBS acceptance records.

This is offline qualification infrastructure, never a production CBS backend.
The physical H2 case uses cc-pVDZ/cc-pVTZ. The water case is explicitly a
synthetic-basis algebra test with nonzero triples, not a predictive CBS model.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc import Atom, BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc._model_resolution import snapshot_basis
from threadpoolctl import threadpool_limits

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
PYSCF_VERSION = "2.14.0"


def hydrogen_basis(name: str) -> BasisSet:
    """Snapshot the independently packaged published Dunning basis tables."""
    from pyscf.gto import basis

    source = Path(basis.__file__).parent / basis.ALIAS[name.replace("-", "")]
    shells = tuple(
        BasisShell(
            angular_momentum=shell[0],
            exponents=tuple(str(row[0]) for row in shell[1:]),
            coefficients=tuple(
                tuple(str(row[column]) for row in shell[1:])
                for column in range(1, len(shell[1]))
            ),
            source_group=index,
        )
        for index, shell in enumerate(basis.load(name, "H"))
    )
    return BasisSet(
        name,
        (ElementBasis(1, shells),),
        BasisProvenance(
            "PySCF packaged Dunning basis tables",
            PYSCF_VERSION,
            "Apache-2.0",
            hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
        "spherical",
    )


def qualification_cases() -> dict[str, tuple[tuple[Atom, ...], BasisSet, BasisSet]]:
    """Use checked physical geometry inputs and a labeled algebra stress pair."""
    from tools.cc_gradient_fixtures import inputs
    from tools.generativeqc_posthf.fixtures import load_fixture

    hydrogen, _ = load_fixture("h2")
    water = inputs("h2o")
    low = snapshot_basis("sto-3g", "spherical")
    low = replace(
        low,
        elements=tuple(
            element for element in low.elements if element.atomic_number in (1, 8)
        ),
    )
    elements = tuple(
        replace(
            element,
            shells=tuple(
                replace(
                    shell,
                    exponents=tuple(
                        str(float(value) * 0.8) for value in shell.exponents
                    ),
                )
                if element.atomic_number == 1
                else shell
                for shell in element.shells
            ),
        )
        for element in low.elements
    )
    high = replace(low, name="water-algebra-high", elements=elements)

    def atoms(record: dict) -> tuple[Atom, ...]:
        return tuple(
            Atom(number, tuple(coordinates))
            for number, coordinates in zip(
                record["atomic_numbers"], record["coordinates"], strict=True
            )
        )

    return {
        "h2-dz-tz": (
            atoms(hydrogen["inputs"]),
            hydrogen_basis("cc-pvdz"),
            hydrogen_basis("cc-pvtz"),
        ),
        "water-algebra": (atoms(water), low, high),
    }


def oracle_component(
    atoms: Sequence[Atom], basis: BasisSet, method: str, *, gradients: bool
) -> dict[str, Any]:
    """Converge independent RHF/MP2/CCSD(T), including the triples Lambda."""
    from pyscf import cc, gto, mp, scf
    from pyscf.cc import ccsd_t_lambda
    from pyscf.data.elements import ELEMENTS
    from pyscf.grad import ccsd_t as grad_ccsd_t

    labels = [
        f"{ELEMENTS[atom.atomic_number]}{index}" for index, atom in enumerate(atoms)
    ]
    tables = {label: [] for label in labels}
    for shell in basis.shells_for(atoms):
        tables[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ],
            ]
        )
    molecule = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, atoms, strict=True)
        ],
        basis=tables,
        unit="Bohr",
        cart=basis.representation == "cartesian",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-13
    mean_field.conv_tol_grad = 1e-11
    mean_field.max_cycle = 200
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent RHF did not converge")
    energy = float(mean_field.e_tot)
    triples = 0.0
    if method == "rhf":
        gradient = mean_field.nuc_grad_method().kernel() if gradients else None
    elif method == "mp2":
        correlation = mp.MP2(mean_field, frozen=0)
        correlation.kernel()
        energy = float(correlation.e_tot)
        gradient = correlation.nuc_grad_method().kernel() if gradients else None
    elif method == "ccsd(t)":
        coupled = cc.CCSD(mean_field, frozen=0)
        coupled.max_cycle = 150
        coupled.conv_tol = 1e-13
        coupled.conv_tol_normt = 1e-11
        coupled.kernel()
        if not coupled.converged:
            raise RuntimeError("independent CCSD did not converge")
        eris = coupled.ao2mo()
        triples = float(coupled.ccsd_t(eris=eris))
        energy = float(coupled.e_tot + triples)
        gradient = None
        if gradients:
            converged, lambda_singles, lambda_doubles = ccsd_t_lambda.kernel(
                coupled,
                eris,
                coupled.t1,
                coupled.t2,
                max_cycle=150,
                tol=1e-11,
            )
            if not converged:
                raise RuntimeError("independent CCSD(T) Lambda did not converge")
            gradient = grad_ccsd_t.Gradients(coupled).kernel(
                coupled.t1,
                coupled.t2,
                lambda_singles,
                lambda_doubles,
                eris=eris,
            )
    else:
        raise ValueError(f"unknown oracle method {method}")
    return {
        "energy": energy,
        "forces": None if gradient is None else (-np.asarray(gradient)).tolist(),
        "hf_energy": float(mean_field.e_tot),
        "triples_energy": triples,
        "basis_functions": molecule.nao_nr(),
    }


def oracle_composite(
    atoms: Sequence[Atom], low: BasisSet, high: BasisSet, *, gradients: bool
) -> dict[str, Any]:
    """Apply direct scalar formulas, independently of production CBS weights."""
    components = {
        "hf_low": oracle_component(atoms, low, "rhf", gradients=gradients),
        "hf_high": oracle_component(atoms, high, "rhf", gradients=gradients),
        "mp2_low": oracle_component(atoms, low, "mp2", gradients=gradients),
        "mp2_high": oracle_component(atoms, high, "mp2", gradients=gradients),
        "correction_ccsdt": oracle_component(
            atoms, low, "ccsd(t)", gradients=gradients
        ),
    }

    def combine(values: dict) -> tuple[Any, Any, Any]:
        hf = (math.exp(1.63) * values["hf_high"] - values["hf_low"]) / (
            math.exp(1.63) - 1
        )
        correlation = (
            27 * (values["mp2_high"] - values["hf_high"])
            - 8 * (values["mp2_low"] - values["hf_low"])
        ) / 19
        correction = values["correction_ccsdt"] - values["mp2_low"]
        return hf, correlation, correction

    hf_energy, correlation_energy, delta_energy = combine(
        {role: record["energy"] for role, record in components.items()}
    )
    forces = None
    if gradients:
        force_parts = combine(
            {role: np.asarray(record["forces"]) for role, record in components.items()}
        )
        forces = sum(force_parts).tolist()
    return {
        "energy": hf_energy + correlation_energy + delta_energy,
        "forces": forces,
        "hf_energy": hf_energy,
        "mp2_correlation_energy": correlation_energy,
        "delta_ccsdt_energy": delta_energy,
        "components": components,
    }


def main() -> None:
    """Write pinned canonical bases and independently checked composite records."""
    import pyscf
    from pyscf.cc import ccsd_t, ccsd_t_lambda
    from pyscf.grad import ccsd, mp2
    from pyscf.grad import ccsd_t as grad_ccsd_t

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tests/reference_data/cbs"
    )
    arguments = parser.parse_args()
    if pyscf.__version__ != PYSCF_VERSION:
        raise RuntimeError(f"reference generation requires PySCF {PYSCF_VERSION}")
    arguments.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for name, (atoms, low, high) in qualification_cases().items():
            low.write(arguments.output / f"{name}-low.json")
            high.write(arguments.output / f"{name}-high.json")
            reference = oracle_composite(atoms, low, high, gradients=True)
            direction = np.random.default_rng(155206).normal(size=(len(atoms), 3))
            direction -= direction.mean(axis=0)
            direction /= np.linalg.norm(direction)
            coordinates = np.asarray([atom.position for atom in atoms])
            derivatives = []
            for step in (1e-3, 3e-4):
                energies = []
                for sign in (-1, 1):
                    displaced = tuple(
                        Atom(atom.atomic_number, tuple(position))
                        for atom, position in zip(
                            atoms, coordinates + sign * step * direction, strict=True
                        )
                    )
                    energies.append(
                        oracle_composite(displaced, low, high, gradients=False)[
                            "energy"
                        ]
                    )
                derivative = (energies[1] - energies[0]) / (2 * step)
                analytic = -float(np.vdot(reference["forces"], direction))
                derivatives.append(
                    {
                        "step_bohr": step,
                        "derivative": derivative,
                        "analytic": analytic,
                        "absolute_error": abs(derivative - analytic),
                    }
                )
            record = {
                "schema": "generativeqc.ccsdt_cbs_reference",
                "version": 1,
                "pyscf": pyscf.__version__,
                "case": name,
                "physical_cbs_basis_pair": name == "h2-dz-tz",
                "atoms": [(atom.atomic_number, atom.position) for atom in atoms],
                "low_basis": f"{name}-low.json",
                "high_basis": f"{name}-high.json",
                "basis_identities": [low.identity, high.identity],
                "cardinals": [2, 3],
                "correlation_exponent": 3.0,
                "hf_exponent": 1.63,
                "charge": 0,
                "multiplicity": 1,
                "frozen_core": 0,
                "density_fitting": False,
                "reference": reference,
                "direction": direction.tolist(),
                "finite_difference": derivatives,
                "source_sha256": {
                    module.__name__: hashlib.sha256(
                        Path(inspect.getfile(module)).read_bytes()
                    ).hexdigest()
                    for module in (ccsd_t, ccsd_t_lambda, ccsd, grad_ccsd_t, mp2)
                },
            }
            (arguments.output / f"{name}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
            print(
                name,
                "E",
                reference["energy"],
                "FD errors",
                [item["absolute_error"] for item in derivatives],
                flush=True,
            )


if __name__ == "__main__":
    main()

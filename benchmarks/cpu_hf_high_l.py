"""Qualify complete fresh-object CPU HF endpoints with loaded d/f shells.

Run baseline and candidate libraries in separate processes with the same options.
The independent PySCF oracle and source-derived work census are outside timing;
the census describes expected source work, not measured hardware instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path


def quartet_census(shells: tuple) -> dict[str, int]:
    """Count canonical Cartesian work and eligible shared-geometry evaluations."""
    totals = {
        "shell_quartets": 0,
        "primitive_components": 0,
        "spd_primitive_components": 0,
        "spd_shared_geometries": 0,
        "f_primitive_components": 0,
        "f_shared_geometries": 0,
        "fallback_primitive_components": 0,
    }
    for first, bra_first in enumerate(shells):
        for second in range(first + 1):
            bra_second = shells[second]
            for third in range(first + 1):
                for fourth in range(third + 1):
                    if first == third and second < fourth:
                        continue
                    selected = (bra_first, bra_second, shells[third], shells[fourth])
                    sizes = [
                        (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                        for shell in selected
                    ]
                    bra = sizes[0] * sizes[1]
                    ket = sizes[2] * sizes[3]
                    if first == second:
                        bra = sizes[0] * (sizes[0] + 1) // 2
                    if third == fourth:
                        ket = sizes[2] * (sizes[2] + 1) // 2
                    components = bra * ket
                    if first == third and second == fourth:
                        components = bra * (bra + 1) // 2
                    primitives = 1
                    for shell in selected:
                        primitives *= len(shell.primitives)
                    work = components * primitives
                    totals["shell_quartets"] += 1
                    totals["primitive_components"] += work
                    if all(shell.angular_momentum <= 2 for shell in selected):
                        totals["spd_primitive_components"] += work
                        totals["spd_shared_geometries"] += primitives
                    else:
                        totals["fallback_primitive_components"] += work
                        if max(shell.angular_momentum for shell in selected) == 3:
                            totals["f_primitive_components"] += work
                            totals["f_shared_geometries"] += primitives
    return totals


def main() -> None:
    """Record fresh complete calls, geometry changes, oracle gates and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("h2", "heh-spdf", "water-svp", "water-tzvp", "formaldehyde-tzvp"),
        default="water-tzvp",
    )
    parser.add_argument(
        "--representation", choices=("cartesian", "spherical"), default="spherical"
    )
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    from generativeqc import Atom, Calculator, Primitive, Shell
    from pyscf import __version__ as pyscf_version
    from pyscf import gto, lib, scf

    lib.num_threads(1)
    charge = 0
    if args.case == "heh-spdf":
        atoms = [("He", (0.1, -0.2, -0.7)), ("H", (-0.3, 0.4, 0.8))]
        oracle_basis = {
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
            for record in oracle_basis[element]
        )
        charge = 1
    elif args.case == "h2":
        atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
        basis = "sto-3g"
    elif args.case.startswith("water"):
        atoms = [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -1.43, 1.1)),
            ("H", (0.0, 1.43, 1.1)),
        ]
        basis = "def2-svp" if args.case.endswith("svp") else "def2-tzvp"
    else:
        atoms = [
            ("C", (0.0, 0.0, 0.0)),
            ("O", (0.0, 0.0, 2.3)),
            ("H", (0.0, -1.77, -1.1)),
            ("H", (0.0, 1.77, -1.1)),
        ]
        basis = "def2-tzvp"
    if args.case != "heh-spdf":
        oracle_basis = basis
    options = {
        "method": "rhf",
        "basis": basis,
        "basis_representation": args.representation,
        "device": "cpu",
        "density_fitting": "none",
        "initial_guess": None,
        "energy_tolerance": 1e-11,
        "density_tolerance": 1e-9,
        "max_iterations": 100,
    }
    inspection = Calculator(**options)
    shells = inspection._shells_for_atoms(
        tuple(Atom.from_value(atom) for atom in atoms)
    )
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    digest = hashlib.sha256()
    with library.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "case": args.case,
        "representation": args.representation,
        "forces": args.forces,
        "charge": charge,
        "options": {key: value for key, value in options.items() if key != "basis"},
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library": str(library),
        "library_sha256": digest.hexdigest(),
        "host": platform.node(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "thread_settings": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "pyscf_version": pyscf_version,
        "timing_scope": "fresh Calculator construction plus complete singlepoint",
        "source_derived_census": quartet_census(shells),
        "census_contract": "value-only work eligibility, not executed force or hardware counters",
        "shell_angular_momenta": [shell.angular_momentum for shell in shells],
        "phases": {},
    }
    properties = ("energy", "forces") if args.forces else ("energy",)
    for phase in ("original", "moved"):
        coordinates = [(element, list(position)) for element, position in atoms]
        if phase == "moved":
            coordinates[-1][1][0] += 0.013
            coordinates[-1][1][2] -= 0.007
        molecule = gto.M(
            atom=coordinates,
            unit="Bohr",
            basis=oracle_basis,
            charge=charge,
            cart=args.representation == "cartesian",
            verbose=0,
        )
        reference = scf.RHF(molecule)
        reference.init_guess = "1e"
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-10
        reference.max_cycle = 100
        reference.kernel()
        newton_refined = not reference.converged
        if newton_refined:
            reference = reference.newton()
            reference.max_cycle = 100
            reference.kernel()
        if not reference.converged:
            raise RuntimeError("independent RHF oracle did not converge")
        forces = -reference.nuc_grad_method().kernel() if args.forces else None
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            calculator = Calculator(**options)
            result = calculator.singlepoint(
                coordinates, charge=charge, properties=properties
            )
            elapsed = time.perf_counter() - started
            energy_error = abs(result.energy - reference.e_tot)
            force_error = (
                float(np.max(np.abs(result.forces - forces))) if args.forces else None
            )
            if (
                result.executed_backend not in ("cpu", "cpu_reference")
                or not result.converged
                or energy_error > 1e-8
                or (force_error is not None and force_error > 1e-7)
            ):
                raise RuntimeError(
                    f"independent CPU HF acceptance failed: "
                    f"backend={result.executed_backend}, {energy_error=}, {force_error=}"
                )
            samples.append(
                {
                    "seconds": elapsed,
                    "energy": result.energy,
                    "iterations": result.iterations,
                    "executed_backend": result.executed_backend,
                    "energy_error": energy_error,
                    "force_error": force_error,
                }
            )
            calculator.clear_cache()
        report["phases"][phase] = {
            "oracle_newton_refined": newton_refined,
            "samples": samples,
            "median_seconds": statistics.median(
                sample["seconds"] for sample in samples
            ),
        }
        print(phase, report["phases"][phase]["median_seconds"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

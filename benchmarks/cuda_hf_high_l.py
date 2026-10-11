"""Qualify complete strict-CUDA HF endpoints against an independent CPU oracle.

Run each frozen library in its own process under Slurm. Cold includes calculator
construction, warm reuses that calculator, and moved changes its geometry without
injecting an oracle density. First-process CUDA startup is included only in the
first cold sample. Domain counts are unscreened logical bounds, not measured GPU
executions; use the native canonical-work gate for executed source counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

CASES = (
    "h2",
    "water-svp",
    "water-tzvp",
    "water-ccpvtz",
    "water-dimer-tzvp",
    "formaldehyde-tzvp",
)


def fixture(name: str) -> tuple[list[tuple[str, list[float]]], str]:
    """Return fixed Bohr geometries and unmodified standard orbital bases."""
    if name not in CASES:
        raise ValueError(f"unknown CUDA HF case: {name}")
    if name == "h2":
        return [("H", [0.0, 0.0, -0.7]), ("H", [0.0, 0.0, 0.7])], "sto-3g"
    if name == "formaldehyde-tzvp":
        return [
            ("C", [0.0, 0.0, 0.0]),
            ("O", [0.0, 0.0, 2.3]),
            ("H", [0.0, -1.77, -1.1]),
            ("H", [0.0, 1.77, -1.1]),
        ], "def2-tzvp"
    atoms = [
        ("O", [0.0, 0.0, 0.0]),
        ("H", [0.0, -1.43, 1.1]),
        ("H", [0.0, 1.43, 1.1]),
    ]
    if name == "water-dimer-tzvp":
        atoms += [
            (element, [position[0] + 6.0, position[1], position[2]])
            for element, position in atoms
        ]
    basis = {"water-svp": "def2-svp", "water-ccpvtz": "cc-pvtz"}.get(name, "def2-tzvp")
    return atoms, basis


def source_domain(shells: Any, representation: str) -> dict[str, Any]:
    """Count the complete Cartesian source domain without assuming screening."""
    primitive_counts = []
    public_aos = 0
    records = []
    for shell in shells:
        angular = shell.angular_momentum
        components = (angular + 1) * (angular + 2) // 2
        primitives = len(shell.primitives)
        primitive_counts.extend([primitives] * components)
        public_aos += components if representation == "cartesian" else 2 * angular + 1
        records.append([shell.atom_index, angular, primitives])
    pair_weights = [
        first * second
        for index, first in enumerate(primitive_counts)
        for second in primitive_counts[: index + 1]
    ]
    pairs = len(pair_weights)
    return {
        "shells": records,
        "f_shells": sum(record[1] == 3 for record in records),
        "cartesian_aos": len(primitive_counts),
        "public_aos": public_aos,
        "unscreened_cartesian_canonical_quartets": pairs * (pairs + 1) // 2,
        "unscreened_primitive_component_products": (
            sum(pair_weights) ** 2 + sum(weight**2 for weight in pair_weights)
        )
        // 2,
        "executed_primitive_component_evaluations": None,
    }


def validate_sample(sample: dict[str, Any], forces: bool) -> None:
    """Reject backend substitution and retain the fixed independent gates."""
    if (
        sample["executed_backend"] != "cuda"
        or not sample["converged"]
        or not 0.0 <= sample["energy_error"] <= 1e-8
        or (forces and not 0.0 <= sample["force_error"] <= 1e-7)
    ):
        raise RuntimeError(f"independent strict-CUDA HF acceptance failed: {sample}")


def digest(path: Path) -> str:
    """Hash the actual library/driver bytes outside all endpoint timings."""
    owner = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            owner.update(chunk)
    return owner.hexdigest()


def json_safe(value: Any) -> Any:
    """Preserve failed nonfinite measurements as null in valid JSON receipts."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {name: json_safe(item) for name, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Measure fresh/reused/moved calculators without timing the oracle."""
    import numpy as np
    from generativeqc import Atom, Calculator
    from pyscf import __version__ as pyscf_version
    from pyscf import gto, lib, scf

    lib.num_threads(1)
    atoms, basis = fixture(args.case)
    moved = [(element, list(position)) for element, position in atoms]
    moved[-1][1][0] += 0.013
    moved[-1][1][2] -= 0.007
    references = {}
    for phase, coordinates in (("original", atoms), ("moved", moved)):
        molecule = gto.M(
            atom=coordinates,
            unit="Bohr",
            basis=basis,
            cart=args.representation == "cartesian",
            verbose=0,
        )
        reference = scf.RHF(molecule)
        reference.init_guess = "1e"
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-10
        reference.max_cycle = 100
        reference.kernel()
        refined = not reference.converged
        if refined:
            reference = reference.newton()
            reference.max_cycle = 100
            reference.kernel()
        if not reference.converged:
            raise RuntimeError("independent PySCF RHF oracle did not converge")
        references[phase] = {
            "energy": float(reference.e_tot),
            "forces": -reference.nuc_grad_method().kernel() if args.forces else None,
            "newton_refined": refined,
            "public_aos": molecule.nao_nr(),
        }
    report["oracle"] = {
        "pyscf_version": pyscf_version,
        "energies": {phase: value["energy"] for phase, value in references.items()},
        "newton_refined": {
            phase: value["newton_refined"] for phase, value in references.items()
        },
    }
    properties = ("energy", "forces") if args.forces else ("energy",)
    for repeat in range(args.repeats):
        calculator = None
        try:
            for phase, coordinates, oracle_phase in (
                ("cold", atoms, "original"),
                ("warm", atoms, "original"),
                ("moved", moved, "moved"),
            ):
                report["active_measurement"] = {"repeat": repeat, "phase": phase}
                started = time.perf_counter()
                if calculator is None:
                    calculator = Calculator(
                        method="rhf",
                        basis=basis,
                        basis_representation=args.representation,
                        device="cuda",
                        precision="fp64",
                        density_fitting="none",
                        initial_guess=None,
                        energy_tolerance=1e-11,
                        density_tolerance=1e-9,
                        max_iterations=100,
                    )
                result = calculator.singlepoint(coordinates, properties=properties)
                elapsed = time.perf_counter() - started
                reference = references[oracle_phase]
                sample = {
                    "repeat": repeat,
                    "phase": phase,
                    "seconds": elapsed,
                    "energy": result.energy,
                    "iterations": result.iterations,
                    "executed_backend": result.executed_backend,
                    "converged": result.converged,
                    "energy_error": abs(result.energy - reference["energy"]),
                    "force_error": float(
                        np.max(np.abs(result.forces - reference["forces"]))
                    )
                    if args.forces
                    else None,
                }
                report["samples"].append(sample)
                validate_sample(sample, args.forces)
                if "source_domain" not in report:
                    shells = calculator._shells_for_atoms(
                        tuple(Atom.from_value(atom) for atom in atoms)
                    )
                    report["source_domain"] = source_domain(shells, args.representation)
                    if report["source_domain"]["public_aos"] != reference["public_aos"]:
                        raise RuntimeError("native/oracle AO domains differ")
                    if (
                        args.case not in ("h2", "water-svp")
                        and not report["source_domain"]["f_shells"]
                    ):
                        raise RuntimeError("high-l endpoint lacks a loaded f shell")
                print(json.dumps(sample, sort_keys=True), flush=True)
        finally:
            if calculator is not None:
                calculator.clear_cache()
    report.pop("active_measurement", None)
    report["median_seconds"] = {
        phase: statistics.median(
            sample["seconds"]
            for sample in report["samples"]
            if sample["phase"] == phase
        )
        for phase in ("cold", "warm", "moved")
    }
    import cupy as cp
    from _support import cuda_accelerator_metadata

    report["accelerator"] = cuda_accelerator_metadata(cp)


def main() -> int:
    """Preserve partial failure receipts instead of treating missing gates as pass."""
    from _support import raw_output_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="water-tzvp")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("cartesian", "spherical"), default="spherical"
    )
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not os.environ.get("SLURM_JOB_ID"):
        parser.error(
            "positive repeats and a real finite-time Slurm allocation are required"
        )
    library = args.library.resolve(strict=True)
    os.environ["GENERATIVEQC_LIBRARY"] = str(library)
    report: dict[str, Any] = {
        "schema": "generativeqc.cuda-hf-high-l.v1",
        "status": "running",
        "case": args.case,
        "representation": args.representation,
        "forces": args.forces,
        "precision": "fp64",
        "library": str(library),
        "library_sha256": digest(library),
        "driver_sha256": digest(Path(__file__)),
        "samples": [],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("GENERATIVEQC_")
            or name
            in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
            )
        },
    }
    try:
        run(args, report)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

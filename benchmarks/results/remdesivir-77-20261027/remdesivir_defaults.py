"""Remdesivir variant of benchmarks/readme_defaults.py.

Reuses that module's timed endpoints unchanged (run_native / run_reference:
one cold E+F plus warm E+F calls from the frozen cold density, stock defaults,
grid response on). Only the geometry and the orbital basis are replaced, and
the DF auxiliary tables are frozen for the remdesivir elements (H, C, N, O, P)
instead of water-3.

Run from the generativeqc checkout with PYTHONPATH=.:python.
"""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path

import benchmarks.readme_defaults as rd
import benchmarks.readme_hf_scaling as scaling

BASES = ("def2-svp", "def2-tzvp")


def read_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    """Angstrom xyz -> Bohr tuples, converted once with PySCF's constant."""
    from pyscf.lib.parameters import BOHR

    lines = path.read_text().splitlines()
    count = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        symbol, *xyz = line.split()
        atoms.append((symbol, tuple(float(value) / BOHR for value in xyz)))
    return atoms


def configure(basis: str, atoms: list) -> None:
    """Point the shared helpers at this geometry and orbital basis."""

    def orbital_basis(method: str):
        return rd.reference_r2scan3c_basis() if method == "r2scan-3c" else basis

    def native_calculator(method: str, arm: str, mode: str, inputs: Path):
        from generativeqc import Calculator, PrecisionCalculator

        options = {
            "method": rd.METHODS[method][0],
            "device": "cuda",
            "basis_representation": "spherical",
        }
        if method != "r2scan-3c":
            options["basis"] = basis
        if mode == "df":
            options.update(
                density_fitting="cuda",
                auxiliary_basis=inputs / f"{method}-auxiliary.json",
            )
            budget = int(os.environ.get("GQC_DF_BUDGET_BYTES", "0"))
            if budget:
                options["density_fitting_memory_budget_bytes"] = budget
        return (
            PrecisionCalculator(policy="fast", **options)
            if arm == "fast"
            else Calculator(**options)
        )

    rd.orbital_basis = orbital_basis
    rd.native_calculator = native_calculator
    # prepare_inputs freezes PySCF's default auxiliary tables from "water-3".
    scaling.scaling_cases = lambda: {"water-3": types.SimpleNamespace(atoms=atoms)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=(*rd.MEASUREMENT_ARMS, "inputs"))
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--basis", choices=BASES, default="def2-svp")
    parser.add_argument("--method", choices=rd.METHODS)
    parser.add_argument("--mode", choices=("direct", "df"), default="direct")
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    atoms = read_xyz(args.xyz)
    configure(args.basis, atoms)
    if args.arm == "inputs":
        rd.prepare_inputs(args.inputs)
        return
    if not (args.method and args.output and args.source_revision):
        parser.error("measured arms require --method, --output and --source-revision")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU work requires a Slurm GPU allocation")
    if args.output.exists():
        raise FileExistsError(args.output)
    import cupy as cp

    from benchmarks._support import cuda_accelerator_metadata, environment_metadata

    record = {
        "schema": "remdesivir-defaults.v1 (from " + rd.SCHEMA + ")",
        "status": "running",
        "system": args.xyz.stem,
        "arm": args.arm,
        "method": args.method,
        "mode": args.mode,
        "atoms": len(atoms),
        "geometry_bohr": atoms,
        "warm_repeats": args.warm_repeats,
        "orbital_basis": "def2-mTZVPP" if args.method == "r2scan-3c" else args.basis,
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
        "source_revision": args.source_revision,
        "runner_sha256": rd.digest(Path(__file__)),
        "readme_defaults_sha256": rd.digest(Path(rd.__file__)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "thread_controls": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "SLURM_CPUS_PER_TASK",
            )
        },
        "accelerator": cuda_accelerator_metadata(cp),
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                "cupy": ("cupy-cuda12x",),
                "numpy": ("numpy",),
            }
        ),
        "protocol": {
            "cold": "constructor+prepare+SCF+forces+host-export",
            "warm": "same-owner; frozen engine-local cold density",
            "grid_overrides": False,
            "accuracy_qualification": "none (cross-engine differences only)",
        },
        "samples": [],
    }
    if args.mode == "df":
        record["auxiliary_input_sha256"] = rd.digest(
            args.inputs / f"{args.method}-auxiliary.json"
        )

    def retain(sample: dict) -> None:
        record["samples"].append(sample)
        rd.save(args.output, record)
        print(json.dumps({k: sample[k] for k in ("phase", "repeat", "seconds", "iterations")}), flush=True)
        # nsys --capture-range=cudaProfilerApi then records the warm calls only.
        if os.environ.get("PROFILE_WARM") and sample["phase"] == "cold":
            cp.cuda.runtime.deviceSynchronize()
            cp.cuda.profiler.start()

    rd.save(args.output, record)
    try:
        (rd.run_native if args.arm in ("standard", "fast") else rd.run_reference)(
            args, atoms, record, retain
        )
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        rd.save(args.output, record)


if __name__ == "__main__":
    main()

"""Measure complete prepared CCSD(T) cold, warm and changed-geometry endpoints.

Set GENERATIVEQC_LIBRARY to the candidate native library and PYTHONPATH to
``python:.``. This driver retains exact outputs, public phase/work diagnostics,
the native binary digest, and partial progress if a later execution fails.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from generativeqc import Calculator

from benchmarks.readme_hf_scaling import scaling_cases


def main() -> None:
    """Run one prepared item without resetting its warm state between repeats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=3)
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget", type=int, default=8 << 30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    moved = [
        (z, [x + (0.01 if i == 1 and axis == 2 else 0.0) for axis, x in enumerate(xyz)])
        for i, (z, xyz) in enumerate(atoms)
    ]
    properties = ("energy", "forces") if args.forces else ("energy",)
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "library": str(library),
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "atoms": args.atoms,
        "basis": "sto-3g",
        "device": args.device,
        "forces": args.forces,
        "budget": args.budget,
        "rows": [],
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    started = perf_counter()
    calc = Calculator(
        method="ccsd(t)",
        basis="sto-3g",
        basis_representation="spherical",
        device=args.device,
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=150,
        ccsd_energy_tolerance=1e-12,
        ccsd_residual_tolerance=1e-10,
        correlation_memory_budget_bytes=args.budget,
    )
    with calc.prepare_batch([atoms]) as prepared:
        record["prepare_seconds"] = perf_counter() - started
        for label, coords in [
            ("cold", None),
            ("warm", None),
            ("warm-repeat", None),
            ("changed", [xyz for _, xyz in moved]),
        ]:
            started = perf_counter()
            batch = prepared.execute(
                properties=properties, coordinates=[coords], strict=True
            )
            elapsed = perf_counter() - started
            result = batch.items[0]
            row = {
                "label": label,
                "seconds": elapsed,
                "energy": result.energy,
                "converged": result.converged,
                "forces": None if result.forces is None else result.forces.tolist(),
                "warm_start_used": result.warm_start_used,
                "warm_start_fallback": result.warm_start_fallback,
                "correlation": asdict(result.correlation),
                "performance": asdict(result.cc_performance),
            }
            record["rows"].append(row)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    record["status"] = "measured"
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

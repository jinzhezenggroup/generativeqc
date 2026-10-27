"""Measure complete cold/warm/displaced DF HF energy-and-force endpoints.

Run each process under a finite Slurm allocation. Clean timings exclude the
separate traced call; every output vector and semantic work counter is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", default="compact")
    parser.add_argument("--buckets", default="packet")
    parser.add_argument("--budget-mib", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "GPU endpoints require Slurm"
    for name in (
        "GENERATIVEQC_DF_TRACE",
        "GENERATIVEQC_DF_SHELL_COUNTERS",
        "GENERATIVEQC_DF_SHELL_WORK",
    ):
        os.environ.pop(name, None)
    os.environ["GENERATIVEQC_DF_WEIGHTED_EXECUTION"] = "shell"
    os.environ["GENERATIVEQC_DF_SHELL_SCHEDULE"] = args.schedule
    os.environ["GENERATIVEQC_DF_PRIMITIVE_BUCKETS"] = args.buckets
    from generativeqc import Calculator, _native

    from benchmarks.df_component_ledger import read_trace

    water = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    copies = {"water": 1, "water8": 8, "water16": 16, "oh": 1}[args.case]
    if args.case == "oh":
        atoms = water[:2]
    else:
        atoms = [
            (symbol, (r[0] + 7 * (index % 4), r[1] + 7 * (index // 4), r[2]))
            for index in range(copies)
            for symbol, r in water
        ]
    spin = int(args.case == "oh")
    coordinates = np.asarray([r for _, r in atoms])
    moved = coordinates.copy()
    moved[1, 2] += 0.015
    rows = []
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    begin = time.perf_counter()
    calc = Calculator(
        device="cuda",
        method="uhf" if spin else "rhf",
        basis="def2-svp",
        basis_representation="spherical",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=args.budget_mib << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    with calc.prepare_batch(
        [atoms] * args.batch, multiplicities=[spin + 1] * args.batch
    ) as owner:
        for phase in ("cold", "warm1", "warm2", "warm3", "moved"):
            if phase != "cold":
                begin = time.perf_counter()
            result = owner.execute(
                [moved] * args.batch if phase == "moved" else None,
                properties=("energy", "forces"),
                strict=True,
            )
            elapsed = time.perf_counter() - begin
            rows.append(
                {
                    "phase": phase,
                    "seconds": elapsed,
                    "items": [
                        {
                            "energy": item.energy,
                            "forces": np.asarray(item.forces).tolist(),
                            "iterations": item.iterations,
                            "fock_builds": item.fock_builds,
                        }
                        for item in result.items
                    ],
                }
            )
            if phase == "cold":
                owner.set_warm_start_updates(False)
        trace = args.output.with_suffix(".trace.jsonl")
        os.environ["GENERATIVEQC_DF_TRACE"] = str(trace)
        os.environ["GENERATIVEQC_DF_SHELL_COUNTERS"] = "1"
        owner.execute(properties=("energy", "forces"), strict=True)
        records = read_trace(trace)
    native = _native.load_library()
    payload = {
        "schema": "generativeqc.df-schedule-endpoint.v1",
        "case": args.case,
        "basis": "def2-svp",
        "representation": "spherical",
        "atoms": atoms,
        "moved_coordinates": moved.tolist(),
        "batch": args.batch,
        "schedule": args.schedule,
        "buckets": args.buckets,
        "budget_bytes": args.budget_mib << 20,
        "library": str(library),
        "library_sha256": digest,
        "source_identity": native.generativeqc_get_source_identity().decode(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "samples": rows,
        "trace": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"case": args.case, "samples": [r["seconds"] for r in rows]}))


if __name__ == "__main__":
    main()

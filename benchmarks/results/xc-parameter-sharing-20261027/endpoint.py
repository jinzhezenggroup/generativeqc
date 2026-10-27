"""Finite-Slurm, complete automatic KS timing with a separate oracle campaign."""

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
    parser.add_argument("--case", choices=("h4", "water"), default="h4")
    parser.add_argument("--name", required=True)
    parser.add_argument("--spins", type=int, required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import Calculator, KsOptions, _native
    from generativeqc_compiler.dft.grid import GridSpec

    base = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.1, 0.2, 1.4)),
        ("H", (1.6, 0.4, 0.3)),
        ("H", (1.8, 0.3, 1.8)),
    ]
    if args.case == "water":
        base = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    tile = 37 if args.case == "h4" else 1024
    energy_tolerance = 1e-12 if args.case == "h4" else 1e-10
    density_tolerance = 1e-10 if args.case == "h4" else 1e-8
    max_iterations = 200 if args.case == "h4" else 400
    atoms = [
        (s, (p[0] + 8 * (i % 2), p[1] + 8 * (i // 2), p[2]))
        for i in range(args.copies)
        for s, p in base
    ]
    moved = np.array([p for _, p in atoms])
    moved[1, 2] += 0.07
    library = Path(os.environ["GENERATIVEQC_LIBRARY"])
    with library.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    start = time.perf_counter()
    calc = Calculator(
        method=f"libxc-{'rks' if args.spins == 1 else 'uks'}:{args.name}",
        basis="sto-3g",
        device="cuda",
        precision="fp64",
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12),
            tile_points=tile,
        ),
        max_iterations=max_iterations,
        energy_tolerance=energy_tolerance,
        density_tolerance=density_tolerance,
        screening_tolerance=1e-12,
    )
    rows = []
    with calc.prepare_batch(
        [atoms], charges=[args.spins - 1], multiplicities=[args.spins], warm_start=True
    ) as batch:
        for phase in ("cold", "warm1", "warm2", "warm3", "moved"):
            if phase != "cold":
                start = time.perf_counter()
            result = batch.execute(
                coordinates=[moved] if phase == "moved" else None,
                strict=True,
                properties=("energy",),
            ).items[0]
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "phase": phase,
                    "seconds": elapsed,
                    "energy": result.energy,
                    "diagnostic": result.ks_diagnostic.to_payload(),
                    "precision": result.precision,
                    "iterations": result.iterations,
                    "fock_builds": result.fock_builds,
                }
            )
            if phase == "cold":
                batch.set_warm_start_updates(False)
    payload = {
        "case": args.case,
        "name": args.name,
        "spins": args.spins,
        "copies": args.copies,
        "atoms": atoms,
        "moved": moved.tolist(),
        "grid": {"radial_points": 12, "angular_polar": 6, "angular_azimuth": 12},
        "tile_points": tile,
        "energy_tolerance": energy_tolerance,
        "density_tolerance": density_tolerance,
        "max_iterations": max_iterations,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "library_sha256": digest,
        "source_identity": _native.load_library()
        .generativeqc_get_source_identity()
        .decode(),
        "samples": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "case": args.case,
                "name": args.name,
                "spins": args.spins,
                "copies": args.copies,
                "seconds": [r["seconds"] for r in rows],
            }
        )
    )


if __name__ == "__main__":
    main()

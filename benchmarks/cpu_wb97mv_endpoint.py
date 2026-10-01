"""Measure complete one-thread CPU WB97M-V endpoints with binary provenance.

Use identical settings and alternating fresh processes for baseline/candidate.
Cold is the first singlepoint after import, warm is repeated one-shot execution,
not density/plan reuse. Every sample includes preparation and complete SCF.
Explicit grids are benchmark inputs, not changes to production grid defaults.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from generativeqc import Calculator, GridSpec, KsOptions, _native

CASES = {
    "water": [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (1.43233673, 0.0, 1.10715266)),
        ("H", (-1.43233673, 0.0, 1.10715266)),
    ],
    "oh": [("O", (0.02, -0.03, 0.04)), ("H", (0.21, 0.31, 1.78))],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="water")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--grid", nargs=3, type=int, default=(16, 6, 12))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    method = "wb97m-v-uks" if args.case == "oh" else "wb97m-v-rks"
    multiplicity = 2 if args.case == "oh" else 1
    grid = GridSpec(
        radial_points=args.grid[0],
        angular_polar=args.grid[1],
        angular_azimuth=args.grid[2],
    )
    calc = Calculator(
        method=method,
        basis=args.basis,
        device="cpu",
        ks_options=KsOptions(grid=grid),
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        max_iterations=160,
    )
    library = Path(_native.load_library()._name)
    record = {
        "method": method,
        "case": args.case,
        "atoms_bohr": CASES[args.case],
        "multiplicity": multiplicity,
        "basis": args.basis,
        "grid": dataclasses.asdict(grid),
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "native_library": str(library),
        "native_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "threads": {
            key: os.getenv(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "samples": [],
    }
    atoms = CASES[args.case]
    phases = [("cold", atoms)] + [("warm", atoms)] * args.repeats
    phases.append(
        (
            "changed_geometry",
            [(symbol, tuple(1.01 * x for x in xyz)) for symbol, xyz in atoms],
        )
    )
    for phase, geometry in phases:
        start = time.perf_counter()
        result = calc.singlepoint(
            geometry, multiplicity=multiplicity, properties=("energy",)
        )
        elapsed = time.perf_counter() - start
        if not result.converged or result.ks_diagnostic is None:
            raise RuntimeError("SCF did not converge or omitted its work diagnostic")
        record["samples"].append(
            {
                "phase": phase,
                "seconds": elapsed,
                "energy": result.energy,
                "iterations": result.iterations,
                "ks_diagnostic": dataclasses.asdict(result.ks_diagnostic),
            }
        )
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

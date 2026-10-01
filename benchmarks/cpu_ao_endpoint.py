"""Measure complete CPU DFT endpoints with explicit binary provenance.

Run in separate processes with GENERATIVEQC_LIBRARY selecting each native build.
Repeated one-shot calls are process-warm, not prepared-plan or density reuse.
Timing excludes import and native compilation, and includes every singlepoint's
preparation, SCF and requested property work. Cold means first endpoint in this
process, not a cold OS page cache. All repeat energies are retained for gating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from generativeqc import Calculator, _native

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (1.43233673, 0.0, 1.10715266)),
    ("H", (-1.43233673, 0.0, 1.10715266)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="r2scan-rks")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--forces", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    calc = Calculator(method=args.method, basis=args.basis, device="cpu")
    library = Path(_native.load_library()._name)
    result = {
        "method": args.method,
        "basis": args.basis,
        "native_library": str(library),
        "native_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "threads": {
            k: os.getenv(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "samples": [],
    }
    properties = ("energy", "forces") if args.forces else ("energy",)
    for phase, atoms in (
        [("cold", WATER)]
        + [("warm", WATER)] * args.repeats
        + [
            (
                "changed_geometry",
                [(symbol, tuple(1.01 * x for x in xyz)) for symbol, xyz in WATER],
            )
        ]
    ):
        start = time.perf_counter()
        item = calc.singlepoint(atoms, properties=properties)
        elapsed = time.perf_counter() - start
        if not item.converged:
            raise RuntimeError("SCF did not converge")
        result["samples"].append(
            {
                "phase": phase,
                "seconds": elapsed,
                "energy": item.energy,
                "iterations": item.iterations,
                "forces": None if item.forces is None else item.forces.tolist(),
                "ks_work": None
                if item.ks_diagnostic is None
                else {
                    key: getattr(item.ks_diagnostic, key)
                    for key in (
                        "fock_builds",
                        "tile_points",
                        "ao_order",
                        "scf_domain",
                        "initial_density_used",
                    )
                },
                "grid_points": None
                if item.ks_diagnostic is None
                else item.ks_diagnostic.grid_points,
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

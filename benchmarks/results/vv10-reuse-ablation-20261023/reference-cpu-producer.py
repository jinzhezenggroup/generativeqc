"""Fresh independent PySCF/GPU4PySCF targets with full grid response.

The public README oracle solver is unchanged. The wrapper supplies truthful
qz provenance, dense XC 5/6 and VV10 3/4 grids, plus an independent CPU spot.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(3, 12, 48, 96), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from benchmarks._support import environment_metadata
    from benchmarks.readme_defaults import COMMON_GATES, digest, save
    from benchmarks.readme_defaults_accuracy import solve
    from benchmarks.readme_hf_scaling import scaling_cases

    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    record = {
        "schema": "generativeqc.vv10-independent-reference.v1",
        "method": "wb97m-v", "atoms": args.atoms, "backend": args.backend,
        "geometry_bohr": atoms, "basis": "def2-svp", "spherical": True,
        "charge": 0, "spin": 0,
        "allocation": {"kind": "qz-notebook" if args.backend == "cpu" else "qz-job", "name": args.allocation},
        "source_revision": args.source_revision,
        "driver_sha256": digest(Path(__file__)),
        "solver_sha256": digest(Path(solve.__code__.co_filename)),
        "common_gates": COMMON_GATES,
        "stability_gate_fraction": 0.25,
        "environment": environment_metadata(distributions={
            "pyscf": ("pyscf",), "gpu4pyscf": ("gpu4pyscf-cuda12x",),
            "cupy": ("cupy-cuda12x",), "numpy": ("numpy",),
        }),
        "samples": [], "status": "running",
    }
    save(args.output, record)
    try:
        for level in ((6,) if args.backend == "cpu" else (5, 6)):
            record["samples"].append(solve("wb97m-v", atoms, backend=args.backend, level=level))
            save(args.output, record)
            print(f"completed {args.backend}/{args.atoms}/level{level}", flush=True)
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, record)


if __name__ == "__main__":
    main()

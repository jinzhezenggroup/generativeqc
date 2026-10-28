"""Source-isolated complete corrected-KS endpoints; used by qualify.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.backend == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real CUDA endpoints require a finite Slurm allocation")
    os.environ["GENERATIVEQC_LIBRARY"] = str(args.library.resolve())
    sys.path[:0] = [str(args.source / "python"), str(args.source)]
    import numpy as np
    from generativeqc import Calculator, GridSpec, ResourceBudget
    from generativeqc.ks import KsOptions
    from generativeqc_compiler.method import resolve_method

    systems = [
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        [("O", (0.0, 0.0, 0.0)), ("H", (1.8, 0.0, 0.0)), ("H", (-0.45, 1.75, 0.0))],
    ]
    moved = np.array([p for _, p in systems[0]])
    moved[1, 0] += 0.15
    configurations = []
    for method in ("PBE-D3(BJ)", "PBE0-D3(0)"):
        for spin in ("unpolarized", "polarized"):
            calculator = Calculator(
                method=resolve_method(method, spin=spin),
                device=args.backend,
                ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
                initial_guess=None,
                resource_budget=ResourceBudget(
                    host_bytes=2 << 30, device_bytes=2 << 30
                ),
            )
            configurations.append((method, spin, calculator))

    def record(result: Any) -> dict[str, Any]:
        diagnostic = result.ks_diagnostic
        if not result.converged or diagnostic is None or result.dispersion is None:
            raise RuntimeError("incomplete corrected endpoint")
        return {
            "energy": result.energy,
            "forces": None if result.forces is None else result.forces.tolist(),
            "iterations": result.iterations,
            "fock_builds": diagnostic.fock_builds,
            "grid_points": diagnostic.grid_points,
            "occupation": diagnostic.occupations,
            "correction_energy": result.dispersion.energy,
            "correction_gradient": None
            if result.dispersion.gradient is None
            else result.dispersion.gradient.tolist(),
            "coulomb_exchange_work": diagnostic.coulomb_exchange_work,
        }

    def run() -> list[dict[str, Any]]:
        rows = []
        for method, spin, calculator in configurations:
            charge, multiplicity = (0, 1) if spin == "unpolarized" else (1, 2)
            for properties in (("energy",), ("energy", "forces")):
                key = f"{method}/{spin}/{'+'.join(properties)}"
                started = time.perf_counter()
                scalar = calculator.singlepoint(
                    systems[0],
                    charge=charge,
                    multiplicity=multiplicity,
                    properties=properties,
                )
                scalar_seconds = time.perf_counter() - started
                rows.append(
                    {
                        "case": key,
                        "phase": "scalar_complete",
                        "seconds": scalar_seconds,
                        "items": [record(scalar)],
                    }
                )
                started = time.perf_counter()
                prepare_started = started
                with calculator.prepare_batch(
                    systems, charges=[charge] * 2, multiplicities=[multiplicity] * 2
                ) as batch:
                    preparation = time.perf_counter() - prepare_started
                    for phase, coordinates in (
                        ("cold", None),
                        ("warm", None),
                        ("warm_repeat", None),
                        ("moved", [moved, None]),
                        ("moved_warm", [moved, None]),
                        ("moved_repeat", [moved, None]),
                    ):
                        execution_started = time.perf_counter()
                        result = batch.execute(
                            coordinates, properties=properties, strict=True
                        )
                        seconds = time.perf_counter() - execution_started
                        rows.append(
                            {
                                "case": key,
                                "phase": phase,
                                "seconds": seconds
                                + (preparation if phase == "cold" else 0),
                                "items": [record(item) for item in result.items],
                            }
                        )
                    close_started = time.perf_counter()
                close_seconds = time.perf_counter() - close_started
                rows.append(
                    {
                        "case": key,
                        "phase": "fleet_protocol_complete",
                        "seconds": time.perf_counter() - started,
                        "preparation_seconds": preparation,
                        "close_seconds": close_seconds,
                        "items": [],
                    }
                )
        return rows

    metadata = {
        "source": str(args.source),
        "library": str(args.library),
        "library_sha256": hashlib.file_digest(
            args.library.open("rb"), "sha256"
        ).hexdigest(),
        "backend": args.backend,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.once:
        print(json.dumps({"metadata": metadata, "rows": run()}), flush=True)
        return
    print(json.dumps({"ready": metadata}), flush=True)
    for line in sys.stdin:
        if json.loads(line).get("stop"):
            return
        print(json.dumps({"rows": run()}), flush=True)


if __name__ == "__main__":
    main()

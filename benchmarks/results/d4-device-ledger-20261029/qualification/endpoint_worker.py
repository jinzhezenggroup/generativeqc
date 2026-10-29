"""Source-isolated complete named D4 endpoints; used by qualify.py."""

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
    from generativeqc import GridSpec, Calculator
    from generativeqc_compiler.method import resolve_method
    from generativeqc.ks import KsOptions

    systems = [
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        [("O", (0.0, 0.0, 0.0)), ("H", (1.8, 0.0, 0.0)), ("H", (-0.45, 1.75, 0.0))],
    ]
    moved = np.array([p for _, p in systems[0]])
    moved[1, 0] += 0.15
    configurations = []
    for method, spin in (("PBE-D4(BJ-EEQ-ATM)", "unpolarized"),
                         ("PBE-D4(BJ-EEQ-ATM)", "polarized"),
                         ("PBE0-D4(BJ-EEQ-ATM)", "unpolarized"),
                         ("PBE0-D4(BJ-EEQ-ATM)", "polarized")):
        configurations.append((method, spin, {
            "method": resolve_method(method, spin=spin),
            "ks_options": KsOptions(grid=GridSpec(24, 12, 24)),
            "basis": "def2-svp", "basis_representation": "spherical"}))

    def create(method: str, options: dict) -> Any:
        return Calculator(device=args.backend, initial_guess=None, **options)

    from generativeqc import dispersion
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("complete named D4 endpoint retried Python correction owner")
    dispersion.D4CorrectionBatch = forbidden

    def record(result: Any) -> dict[str, Any]:
        if not result.converged or result.dispersion is None:
            raise RuntimeError("incomplete named D4 endpoint")
        if result.dispersion.backend != args.backend:
            raise RuntimeError("D4 correction changed backend")
        diagnostic = result.ks_diagnostic
        precision = result.precision or {}
        return {
            "energy": result.energy,
            "forces": None if result.forces is None else result.forces.tolist(),
            "iterations": result.iterations,
            "fock_builds": diagnostic.fock_builds if diagnostic else None,
            "grid_points": diagnostic.grid_points if diagnostic else None,
            "occupation": diagnostic.occupations if diagnostic else None,
            "coulomb_exchange_work": diagnostic.coulomb_exchange_work if diagnostic else None,
            "policy_work": {
                "requested_mode": precision.get("requested_mode"),
                "strict_stage_fock_builds": precision.get("strict_stage_fock_builds"),
                "operators": precision.get("operators"),
                "dispersion_backend": result.dispersion.backend,
            },
            "dispersion_energy": result.dispersion.energy,
            "charges": result.dispersion.charges.tolist(),
        }

    def run() -> list[dict[str, Any]]:
        rows = []
        for method, spin, options in configurations:
            charge, multiplicity = (0, 1) if spin == "unpolarized" else (1, 2)
            for properties in (("energy",), ("energy", "forces")):
                key = f"{method}/{spin}/{'+'.join(properties)}"
                started = time.perf_counter()
                calculator = create(method, options)
                scalar = calculator.singlepoint(
                    systems[0],
                    charge=charge,
                    multiplicity=multiplicity,
                    properties=properties,
                )
                calculator.clear_cache()
                del calculator
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
                calculator = create(method, options)
                with calculator.prepare_batch(
                    systems, charges=[charge, 0], multiplicities=[multiplicity, 1]
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
                        try:
                            result = batch.execute(
                                coordinates, properties=properties, strict=True
                            )
                        except Exception as error:
                            raise RuntimeError(f"{key}/{phase}: {error}") from error
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
                calculator.clear_cache()
                del batch, calculator
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
        "constructor_included": True,
        "python_correction_owner": "poisoned",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
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

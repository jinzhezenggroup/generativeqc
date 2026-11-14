"""Replay the exact original/moved public endpoint protocol on a bound build.

This opt-in evidence driver never changes solver settings after a failed call.
RSS is the process high-water mark. CUDA free-memory snapshots are not peak
memory or allocation counters; absent workspace/peak telemetry remains null.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np

GATE = 5e-7


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist(path: Path, report: dict) -> None:
    """Retain completed observations even when a later request fails closed."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--property", choices=("energy", "force"), default="force")
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing qualification receipt")
    inputs = json.loads(args.inputs.read_text())
    oracle = json.loads(args.oracle.read_text())
    if oracle.get("xtb_config") != {"smooth_cutoff": 0.0}:
        raise ValueError("independent oracle must explicitly select sharp D4 cutoffs")
    if oracle["input_sha256"] != digest(args.inputs):
        raise ValueError("oracle input identity mismatch")
    for line in args.source_manifest.read_text().splitlines():
        identity, name = line.split("  ", 1)
        if digest(args.source_root / name) != identity:
            raise ValueError("source snapshot mismatch: " + name)
    cases = [case for case in inputs["cases"] if case["name"] in args.cases]
    if len(cases) != len(set(args.cases)) or len(args.cases) != len(set(args.cases)):
        raise ValueError("unknown or duplicate case selection")
    references = {(row["case"], row["geometry"]): row for row in oracle["rows"]}
    runtime = None
    if args.device == "cuda":
        if not os.environ.get("SLURM_JOB_ID") or not os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ):
            raise RuntimeError("CUDA replay requires unchanged Slurm visibility")
        runtime = ctypes.CDLL(os.environ["GFN2_CUDART"])
        runtime.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]

    def memory() -> dict:
        snapshot = {
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "device_used_snapshot_bytes": None,
            "device_peak_bytes": None,
            "workspace_bytes": None,
            "allocation_count": None,
        }
        if runtime is not None:
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            status = runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            if status:
                raise RuntimeError(f"cudaMemGetInfo failed: {status}")
            snapshot["device_used_snapshot_bytes"] = total.value - free.value
        return snapshot

    def synchronize() -> None:
        if runtime is not None and runtime.cudaDeviceSynchronize():
            raise RuntimeError("CUDA endpoint synchronization failed")

    from generativeqc import Calculator

    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    report = {
        "schema": "gfn2-public-endpoint-v1",
        "status": "running",
        "device": args.device,
        "property": args.property,
        "library_sha256": digest(library),
        "driver_sha256": digest(Path(__file__)),
        "source_manifest_sha256": digest(args.source_manifest),
        "input_sha256": digest(args.inputs),
        "oracle_sha256": digest(args.oracle),
        "xtb_config": oracle["xtb_config"],
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "settings": {
            "max_iterations": args.max_iterations,
            "history": args.history,
            "energy_tolerance": 1e-10,
            "charge_tolerance": 1e-8,
            "temperature_kelvin": 300,
            "fresh_scc_every_call": True,
        },
        "gate": GATE,
        "memory_contract": "Linux process RSS high-water; device-wide used snapshots only; "
        "device peak, allocator and workspace counters unavailable",
        "timing_contract": "host-returned synchronized public call; constructor charged to cold; "
        "imports and memory snapshots outside timing; no profiler",
        "rows": [],
        "cleanup": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    persist(args.output, report)
    calculator = None
    try:
        for case in cases:
            before = memory()
            started = time.perf_counter()
            calculator = Calculator(
                method="gfn2-xtb",
                device=args.device,
                max_iterations=args.max_iterations,
                energy_tolerance=1e-10,
                density_tolerance=1e-8,
                diis_history=args.history,
            )
            construction_seconds = time.perf_counter() - started
            for geometry, coordinates in enumerate(case["geometries"]):
                expected = references[case["name"], geometry]
                if expected["geometry_sha256"] != case["geometry_sha256"][geometry]:
                    raise ValueError("reference geometry identity mismatch")
                atoms = list(zip(case["symbols"], coordinates, strict=True))
                for repeat in range(6):
                    synchronize()
                    started = time.perf_counter()
                    result = calculator.singlepoint(
                        atoms,
                        charge=case["charge"],
                        multiplicity=case["multiplicity"],
                        properties=("energy", "forces")
                        if args.property == "force"
                        else ("energy",),
                    )
                    forces = (
                        None
                        if result.forces is None
                        else np.asarray(result.forces).copy()
                    )
                    synchronize()
                    seconds = time.perf_counter() - started
                    energy_error = abs(result.energy - expected["energy"])
                    force_error = None
                    if args.property == "force":
                        if (
                            forces is None
                            or forces.shape != np.asarray(coordinates).shape
                        ):
                            raise ValueError(
                                "missing or incorrectly shaped complete forces"
                            )
                        force_error = float(
                            np.max(np.abs(forces - np.asarray(expected["forces"])))
                        )
                    backend = "cuda" if args.device == "cuda" else "cpu_reference"
                    if (
                        not result.converged
                        or result.executed_backend != backend
                        or result.iterations <= 0
                        or not math.isfinite(energy_error)
                        or energy_error > GATE
                        or (
                            force_error is not None
                            and (not math.isfinite(force_error) or force_error > GATE)
                        )
                    ):
                        raise ValueError(
                            f"failed independent science/execution gate: {energy_error}, {force_error}"
                        )
                    row = {
                        "case": case["name"],
                        "geometry": geometry,
                        "geometry_sha256": case["geometry_sha256"][geometry],
                        "phase": ("cold" if geometry == 0 else "moved")
                        if repeat == 0
                        else ("warm" if geometry == 0 else "moved-warm"),
                        "repeat": repeat,
                        "seconds": seconds,
                        "complete_seconds": seconds
                        + (construction_seconds if geometry == repeat == 0 else 0),
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "executed_backend": result.executed_backend,
                        "physical_residual_rms": result.physical_residual_rms,
                        "logical_fock_builds": None,
                        "energy": result.energy,
                        "forces": None if forces is None else forces.tolist(),
                        "energy_error": energy_error,
                        "force_error": force_error,
                        "memory_after": memory(),
                    }
                    report["rows"].append(row)
                    persist(args.output, report)
                    print(
                        case["name"],
                        geometry,
                        repeat,
                        result.iterations,
                        seconds,
                        flush=True,
                    )
            calculator.clear_cache()
            calculator = None
            del result
            gc.collect()
            synchronize()
            report["cleanup"].append(
                {"case": case["name"], "before": before, "after": memory()}
            )
            persist(args.output, report)
        if len(report["rows"]) != 12 * len(cases):
            raise ValueError("incomplete original/moved/five-repeat protocol")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if calculator is not None:
            calculator.clear_cache()
        persist(args.output, report)


if __name__ == "__main__":
    main()

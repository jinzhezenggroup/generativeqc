"""Native XC strategy versus incumbent, complete cold/warm/moved E+F trajectories.

Clean timings and separate diagnostic trajectories are retained independently.
An independent source-matched reference is mandatory; kernel timing or batching
coverage alone is not qualification. Run every real-GPU invocation through Slurm.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions

from benchmarks._support import cuda_accelerator_metadata, environment_metadata
from benchmarks.df_component_ledger import aggregate, read_trace
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_omol25 import check_record, load_comparison_basis, protocol
from benchmarks.readme_pbe0 import PBE0


@contextmanager
def selection(enabled: bool, trace: Path | None, strategy: str = "ragged") -> Any:
    """Freeze owner selection and keep diagnostic I/O outside clean timings."""
    values = {
        "GENERATIVEQC_CUDA_KS_CHUNK": None,
        "GENERATIVEQC_DF_PROGRESS_TRACE": str(trace) if trace else None,
        "GENERATIVEQC_DF_TRACE": str(trace) + ".cuda" if trace else None,
        "GENERATIVEQC_DF_HOST_TRACE": None,
    }
    if strategy == "ragged":
        values.update(
            GENERATIVEQC_CUDA_XC_RAGGED_BATCH=None if enabled else "0",
            GENERATIVEQC_CUDA_XC_COMPACT_BATCH="1",
            GENERATIVEQC_CUDA_XC_BATCH_TILES="32",
            GENERATIVEQC_CUDA_XC_BATCH_BYTES=str(32 * 1024 * 1024),
        )
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def prepared_snapshots(
    path: Path, name: str = "cuda_xc_batch_plan"
) -> list[dict[str, Any]]:
    """Decode completed cumulative receipts, never sum iteration snapshots."""
    scopes: dict[int, dict[str, Any]] = {}
    completed = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["name"] != name:
            continue
        identifier = event["id"]
        if event["event"] == "BEGIN":
            scopes[identifier] = {}
        elif event["event"] == "VALUE":
            scopes[identifier][event["key"]] = event["value"]
        elif event["event"] == "END":
            if event["status"] != "returned":
                raise ValueError("incomplete prepared execution receipt")
            completed.append(scopes.pop(identifier))
    if scopes:
        raise ValueError("unterminated prepared execution receipt")
    return completed


def main(strategy: str = "ragged") -> None:
    """Save complete endpoint vectors, original gates and actual work counts."""
    if strategy != "ragged":
        raise ValueError("unsupported SCF/XC strategy experiment")
    import cupy as cp
    from generativeqc import _native

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(12, 48, 96), required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--feasibility", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("real-GPU execution requires a finite Slurm allocation")
    if args.repeats < 5 and not (args.feasibility and args.repeats == 1):
        parser.error("at least five repeats required, or --feasibility --repeats 1")
    if args.output.exists():
        parser.error("output already exists; choose a fresh evidence path")
    opener = gzip.open if args.reference.suffix == ".gz" else open
    with opener(args.reference, "rt") as source:
        reference = json.load(source)
    case = scaling_cases()[f"water-{args.atoms}"]
    basis, _ = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    scientific = protocol(args.atoms, basis, grid, 5, benchmark=PBE0)
    if reference["stage"] != "complete" or reference["protocol"] != scientific:
        raise ValueError("reference does not match the complete scientific protocol")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": f"generativeqc.pbe0_{strategy}",
        "version": 1,
        "protocol": scientific,
        "feasibility": args.feasibility,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "environment": environment_metadata(accelerator=cuda_accelerator_metadata(cp)),
        "execution_controls": {
            "cuda_ks_chunk": "default",
            "precision": "fp64",
            "strategy": strategy,
        },
        "native_binary_sha256": hashlib.sha256(
            Path(_native.load_library(device="cuda")._name).resolve().read_bytes()
        ).hexdigest(),
        "samples": [],
    }
    root = Path(__file__).resolve().parents[1]
    paths = (
        "src/dft/cuda_ks.cpp",
        "src/scf/cuda_fock_execution.cpp",
        "benchmarks/pbe0_xc_trajectories.py",
    )
    if strategy == "ragged":
        paths += (
            "src/dft/cuda_xc.cpp",
            "src/dft/cuda_xc.hpp",
            "src/dft/cuda_xc_kernels.cuh",
            "python/generativeqc_compiler/dft/xc_point_batch_cuda.py",
            "python/generativeqc_compiler/dft/xc_tile_batch_cuda.py",
            "benchmarks/pbe0_ragged.py",
        )
    record["source_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }

    def save(stage: str) -> None:
        record["stage"] = stage
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    save("started")
    for repeat in range(args.repeats + 1):
        diagnostic = repeat == args.repeats
        arms = (False, True) if repeat % 2 == 0 else (True, False)
        for enabled in arms:
            arm = strategy if enabled else "exact"
            trace = (
                args.output.with_suffix(f".{arm}.progress.jsonl")
                if diagnostic
                else None
            )
            component_trace = Path(str(trace) + ".cuda") if trace else None
            if trace and (trace.exists() or Path(str(trace) + ".cuda").exists()):
                raise ValueError("diagnostic path already exists")
            with selection(enabled, trace, strategy):
                calculator = Calculator(
                    method="pbe0-rks",
                    basis=basis,
                    device="cuda",
                    basis_representation="spherical",
                    ks_options=KsOptions(grid=grid, tile_points=256),
                    precision="fp64",
                    energy_tolerance=1e-12,
                    density_tolerance=1e-10,
                    screening_tolerance=1e-12,
                    max_iterations=100,
                )
                cp.cuda.Stream.null.synchronize()
                started = perf_counter()
                owner = calculator.prepare_batch(
                    [scientific["geometries_bohr"][0]], warm_start=True
                )
                preparation_seconds = perf_counter() - started
                try:
                    for phase, geometry in (
                        ("cold", 0),
                        ("warm", 0),
                        ("moved", 1),
                        ("moved-warm", 1),
                    ):
                        previous_components = (
                            len(read_trace(component_trace))
                            if component_trace and component_trace.exists()
                            else 0
                        )
                        coords = (
                            None
                            if geometry == 0
                            else [
                                np.asarray(
                                    [
                                        position
                                        for _, position in scientific[
                                            "geometries_bohr"
                                        ][1]
                                    ]
                                )
                            ]
                        )
                        cp.cuda.Stream.null.synchronize()
                        started = perf_counter()
                        item = owner.execute(
                            coords, strict=False, properties=("energy", "forces")
                        ).items[0]
                        cp.cuda.Stream.null.synchronize()
                        seconds = perf_counter() - started
                        oracle = next(
                            row
                            for row in reference["records"]
                            if row["geometry"] == geometry
                            and row["phase"] in ("cold", "moved")
                        )
                        sample = {
                            "arm": arm,
                            "repeat": repeat,
                            "diagnostic": diagnostic,
                            "phase": phase,
                            "seconds": seconds,
                            "preparation_seconds": preparation_seconds
                            if phase == "cold"
                            else 0.0,
                            "endpoint_seconds": seconds
                            + (preparation_seconds if phase == "cold" else 0.0),
                            "energy": item.energy,
                            "forces": item.forces.tolist()
                            if item.forces is not None
                            else None,
                            "iterations": item.iterations,
                            "fock_builds": item.fock_builds,
                            "converged": item.converged,
                            "status": item.status,
                            "status_message": item.status_message,
                            "warm_start_used": item.warm_start_used,
                            "warm_start_fallback": item.warm_start_fallback,
                            "ks_diagnostic": item.ks_diagnostic.to_payload(),
                        }
                        validation = check_record(sample, oracle)
                        sample.update(
                            energy_error=validation["energy_error"],
                            force_max_error=validation["force_error"],
                            gate=validation["gate"],
                        )
                        if trace:
                            sample["component_profile"] = aggregate(
                                [
                                    row
                                    for row in read_trace(component_trace)[
                                        previous_components:
                                    ]
                                    if row["operation"].startswith("cuda_ks_")
                                ]
                            )
                        if trace and strategy == "ragged":
                            sample["batch_receipts"] = prepared_snapshots(
                                trace, "cuda_xc_batch_plan"
                            )[-1:]
                        record["samples"].append(sample)
                        save(f"{arm}/{repeat}/{phase}")
                        if not validation["gate"]:
                            raise AssertionError(
                                f"independent E+F gate failed: {arm}/{phase}: "
                                f"energy={validation['energy_error']}, "
                                f"force={validation['force_error']}"
                            )
                        if (
                            trace
                            and enabled
                            and strategy == "ragged"
                            and not any(
                                row["ragged"] and row["compact_large_tiles"] > 0
                                if args.atoms >= 48
                                else row["ragged"] and row["compact_groups"] > 0
                                for row in sample["batch_receipts"]
                            )
                        ):
                            raise AssertionError("requested Ragged XC did not execute")
                finally:
                    owner.close()
    save("complete")


if __name__ == "__main__":
    main()

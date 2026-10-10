"""Fresh-process PBE0 E+F acceptance for independently selected Rys-K tasks.

Run every worker under finite Slurm allocation. The endpoint includes calculator
construction, preparation, first SCF/analytic force call and owner teardown;
imports and input decoding are excluded. No CUDA context, density or owner is
primed. Persistent compiler/artifact caches are shared by both execution modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ in {None, ""}:
    from _support import raw_output_path
else:
    from ._support import raw_output_path


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject incomplete or unmatched samples before interpreting cold time."""
    candidate = (
        "default"
        if any(sample["mode"] == "default" for sample in samples)
        else "rys-task"
    )
    groups = {
        mode: [sample for sample in samples if sample["mode"] == mode]
        for mode in ("incumbent", candidate)
    }
    failures = []
    if min(map(len, groups.values())) < 3:
        failures.append("at least three fresh-process samples per mode required")
    for field in (
        "library_sha256",
        "protocol",
        "host",
        "slurm_job_id",
        "cuda_visible_devices",
        "rys_task_classes",
        "k_task_schedule",
    ):
        if (
            len({json.dumps(sample.get(field), sort_keys=True) for sample in samples})
            != 1
        ):
            failures.append(f"unmatched {field}")
    if len({sample["pid"] for sample in samples}) != len(samples):
        failures.append("samples must use distinct processes")
    for sample in samples:
        if sample.get("schema") == "rys-task-cold-ef.v2" and sample.get(
            "k_task_schedule"
        ) not in {"incumbent", "fill", "primitive", "work"}:
            failures.append("missing or invalid K task schedule")
        if sample["mode"] not in groups:
            failures.append("unexpected execution mode")
        if sample["mode"] == "default" and (
            "k_lowering_environment" not in sample
            or sample["k_lowering_environment"] is not None
        ):
            failures.append("default worker must unset the K lowering selector")
        if sample.get("status") != "measured":
            failures.append("incomplete endpoint record")
        acceptance = sample.get("acceptance", {})
        if not acceptance.get("gate") or not sample.get("converged"):
            failures.append("independent energy/force gate failed")
        for field, tolerance in (("energy_error", 1e-8), ("force_error", 1e-7)):
            error = acceptance.get(field)
            if error is None or not math.isfinite(error) or not 0 <= error <= tolerance:
                failures.append(f"invalid independent {field}")
        if sample.get("warm_start_used", True) or sample.get("context_primed", True):
            failures.append("endpoint is not cold")
        seconds = sample.get("complete_seconds")
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            failures.append("invalid complete endpoint time")
    if failures:
        return {"gate": False, "failures": failures}
    medians = {
        mode: statistics.median(sample["complete_seconds"] for sample in group)
        if group
        else None
        for mode, group in groups.items()
    }
    means = {
        mode: statistics.mean(sample["complete_seconds"] for sample in group)
        for mode, group in groups.items()
    }
    ratio = medians["incumbent"] / medians[candidate] if all(medians.values()) else None
    if ratio is None or ratio <= 1:
        failures.append("no complete cold E+F median advantage")
    if means[candidate] >= means["incumbent"]:
        failures.append("no complete cold E+F mean advantage")
    return {
        "gate": not failures,
        "failures": failures,
        "median_seconds": medians,
        "mean_seconds": means,
        "speedup": ratio,
        "sample_seconds": {
            mode: [sample["complete_seconds"] for sample in group]
            for mode, group in groups.items()
        },
        "fock_builds": {
            mode: [sample["fock_builds"] for sample in group]
            for mode, group in groups.items()
        },
        "max_energy_error": max(
            (sample["acceptance"]["energy_error"] for sample in samples), default=None
        ),
        "max_force_error": max(
            (sample["acceptance"]["force_error"] for sample in samples), default=None
        ),
        "k_task_schedule": samples[0].get("k_task_schedule"),
    }


def configure_lowering(mode: str) -> str | None:
    """Exercise the actual unset default, even when a parent exports a selector."""
    variable = "GENERATIVEQC_DIRECT_K_FOCK_LOWERING"
    if mode == "default":
        os.environ.pop(variable, None)
    else:
        os.environ[variable] = mode
    return os.environ.get(variable)


def worker(args: argparse.Namespace) -> None:
    """Measure one complete endpoint without preparatory device operations."""
    if not os.environ.get("SLURM_JOB_ID") or "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("a finite Slurm GPU allocation is required")
    lowering_environment = configure_lowering(args.mode)
    from generativeqc import Calculator, GridSpec, KsOptions

    from benchmarks.compare_df_direct_endpoint import check_endpoint
    from benchmarks.compare_gpu4pyscf_batch import (
        load_comparison_basis,
        native_build_metadata,
    )
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_omol25 import protocol
    from benchmarks.readme_pbe0 import PBE0

    case = scaling_cases()[f"water-{args.atoms}"]
    basis_path = Path("benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json")
    basis, _ = load_comparison_basis(
        basis_path, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    science = protocol(args.atoms, basis, grid, 5, benchmark=PBE0)
    reference = json.loads(args.reference.read_text())
    if reference["protocol"] != science:
        raise ValueError("independent reference protocol does not match")
    record = {
        "schema": "rys-task-cold-ef.v2",
        "mode": args.mode,
        "k_lowering_environment": lowering_environment,
        "k_task_schedule": os.environ.get("GENERATIVEQC_DIRECT_K_TASK_SCHEDULE")
        or "work",
        "protocol": science,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "library_sha256": hashlib.sha256(
            Path(os.environ["GENERATIVEQC_LIBRARY"]).read_bytes()
        ).hexdigest(),
        "context_primed": False,
        "rys_task_classes": os.environ.get(
            "GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES", "all"
        ),
        "supplied_density": False,
        "timing_scope": "construction + prepare + first SCF/analytic forces + owner teardown",
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    start = perf_counter()
    calculator = Calculator(
        method="pbe0-rks",
        basis=basis,
        device="cuda",
        precision="fp64",
        basis_representation="spherical",
        ks_options=KsOptions(grid=grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=100,
    )
    with calculator.prepare_batch([case.atoms], warm_start=False) as owner:
        prepared = perf_counter()
        print(
            json.dumps({"stage": "prepared", "seconds": prepared - start}), flush=True
        )
        item = owner.execute(strict=True, properties=("energy", "forces")).items[0]
    elapsed = perf_counter() - start
    record.update(
        status="measured",
        complete_seconds=elapsed,
        prepare_seconds=prepared - start,
        energy=item.energy,
        forces=item.forces.tolist(),
        converged=item.converged,
        warm_start_used=item.warm_start_used,
        iterations=item.iterations,
        fock_builds=item.fock_builds,
        physical_residual_rms=item.physical_residual_rms,
        ks_diagnostic=item.ks_diagnostic.to_payload(),
        acceptance=check_endpoint(item, reference),
        native_build=native_build_metadata(calculator),
    )
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "mode",
                    "complete_seconds",
                    "prepare_seconds",
                    "iterations",
                    "fock_builds",
                    "acceptance",
                )
            }
        ),
        flush=True,
    )
    if not record["acceptance"]["gate"]:
        raise RuntimeError("independent cold E+F acceptance failed")


def main() -> None:
    """Measure under Slurm or summarize already retained endpoint evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("default", "incumbent", "rys-task", "rys", "block")
    )
    parser.add_argument("--atoms", type=int, choices=(48, 96), default=96)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--samples", type=Path, nargs="+")
    args = parser.parse_args()
    if args.samples:
        result = summarize([json.loads(path.read_text()) for path in args.samples])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
        if not result["gate"]:
            raise SystemExit(1)
    elif args.mode and args.reference:
        worker(args)
    else:
        parser.error("provide --samples, or both --mode and --reference")


if __name__ == "__main__":
    main()

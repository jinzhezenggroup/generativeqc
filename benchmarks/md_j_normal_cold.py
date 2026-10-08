"""Qualify 96-atom PBE0 using fresh processes and a same-library J-only toggle.

Run the parent through finite Slurm allocation. Children preserve the assigned
CUDA visibility, instantiate fresh calculators and retain no density or owner.
The normal production K, XC, SCF and finalization are identical in both modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def collect(samples: list[dict[str, Any]], oracle_energy: float) -> dict[str, Any]:
    """Require complete, matched numerical endpoints before a cold speedup."""
    failures = []
    groups = {
        mode: [row for row in samples if row["mode"] == mode]
        for mode in ("normal", "md-j")
    }
    if min(map(len, groups.values())) < 3:
        failures.append("at least three fresh-process samples per mode required")
    for field in (
        "library_sha256",
        "input_sha256",
        "normal_revision",
        "slurm_job_id",
        "cuda_visible_devices",
    ):
        if len({row[field] for row in samples}) != 1:
            failures.append(f"unmatched {field}")
    if len({row["pid"] for row in samples}) != len(samples):
        failures.append("samples must execute in distinct fresh processes")
    for row in samples:
        contract = {
            "natoms": 96,
            "nao": 768,
            "grid_points": 294912,
            "grid_points_source": "native-diagnostic",
            "precision": "fp64",
            "screening": 1e-12,
            "energy_tolerance": 1e-11,
            "density_tolerance": 1e-9,
            "process_cold": True,
            "context_primed": False,
            "supplied_density": False,
        }
        if any(row.get(field) != expected for field, expected in contract.items()):
            failures.append("unmatched 96-atom FP64 cold endpoint prescription")
        if not row["converged"] or row["backend"] != "cuda" or row["warm_start_used"]:
            failures.append("incomplete, non-CUDA or warm endpoint")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            failures.append("invalid complete endpoint time")
        error = abs(row["energy"] - oracle_energy)
        if not math.isfinite(error) or error > 3e-9:
            failures.append("independent energy gate failed")
        residual = row["physical_residual"]
        if not math.isfinite(residual) or residual > 1e-9:
            failures.append("physical residual gate failed")
        if row["mode"] == "md-j" and row.get("md_j_calls", 0) != row["fock_builds"]:
            failures.append("MD-J execution count does not match complete Fock count")
        if row["mode"] == "md-j" and not row.get("md_j_default", False):
            failures.append("MD-J must execute through unset/default admission")
        if row["mode"] == "normal" and row.get("md_j_calls", 0) != 0:
            failures.append("normal baseline must not execute MD-J")
    medians = {
        mode: statistics.median(row["seconds"] for row in rows) if rows else None
        for mode, rows in groups.items()
    }
    speedup = (
        medians["normal"] / medians["md-j"]
        if medians["normal"] and medians["md-j"]
        else None
    )
    if speedup is None or not math.isfinite(speedup) or speedup <= 1.0:
        failures.append(
            "complete cold endpoint has no advantage over optimized normal PBE0"
        )
    return {
        "samples": samples,
        "median_seconds": medians,
        "normal_over_md_speedup": speedup,
        "oracle_energy": oracle_energy,
        "energy_gate_hartree": 3e-9,
        "physical_residual_gate": 1e-9,
        "same_library": True,
        "baseline_kind": "optimized-normal",
        "j_only_toggle": True,
        "performance_acceptance_passed": not failures,
        "gate_failures": failures,
    }


def worker(input_path: Path, output_path: Path, mode: str) -> None:
    """Time construction, preparation, full energy-only SCF and owner teardown."""
    from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell

    payload = json.loads(input_path.read_text())
    if len(payload["atoms"]) != 96 or payload["nao"] != 768:
        raise ValueError("acceptance requires exactly 96 atoms and 768 AOs")
    basis = [
        Shell(
            shell["atom_index"],
            shell["angular_momentum"],
            tuple(Primitive(*primitive) for primitive in shell["primitives"]),
        )
        for shell in payload["shells"]
    ]
    start = time.perf_counter()
    calculator = Calculator(
        method="pbe0-rks",
        basis=basis,
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
        screening_tolerance=1e-12,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        max_iterations=150,
    )
    with calculator.prepare_batch([payload["atoms"]], warm_start=False) as owner:
        prepare_seconds = time.perf_counter() - start
        print(
            json.dumps({"stage": "prepared", "mode": mode, "seconds": prepare_seconds}),
            flush=True,
        )
        item = owner.execute(properties=("energy",), strict=True).items[0]
    seconds = time.perf_counter() - start
    grid_points = item.ks_diagnostic.grid_points
    if grid_points != 294912:
        raise RuntimeError(f"native grid has {grid_points} points, expected 294912")
    record = {
        "mode": mode,
        "seconds": seconds,
        "prepare_seconds": prepare_seconds,
        "energy": item.energy,
        "iterations": item.iterations,
        "fock_builds": item.ks_diagnostic.fock_builds,
        "physical_residual": item.ks_diagnostic.physical_residual_max,
        "converged": item.converged,
        "backend": item.executed_backend,
        "warm_start_used": item.warm_start_used,
        "natoms": 96,
        "nao": 768,
        "grid_points": grid_points,
        "grid_points_source": "native-diagnostic",
        "grid_point_visits": grid_points * item.ks_diagnostic.fock_builds,
        "precision": "fp64",
        "screening": 1e-12,
        "energy_tolerance": 1e-11,
        "density_tolerance": 1e-9,
        "pid": os.getpid(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "library_sha256": hashlib.sha256(
            Path(os.environ["GENERATIVEQC_LIBRARY"]).read_bytes()
        ).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "normal_revision": os.environ["GENERATIVEQC_BENCHMARK_SOURCE"],
        "md_j_default": mode == "md-j"
        and "GENERATIVEQC_DISABLE_MD_J" not in os.environ,
        "process_cold": True,
        "context_primed": False,
        "supplied_density": False,
        "timing_scope": "fresh calculator + prepare + complete energy-only SCF + owner teardown; imports and process shutdown excluded",
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=("normal", "md-j"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("every real-GPU execution must run through srun")
    if args.worker:
        worker(args.input, args.output, args.worker)
        return
    if args.repeats < 3 or args.oracle is None:
        parser.error(
            "three or more repeats and an independent oracle record are required"
        )
    oracle = json.loads(args.oracle.read_text())
    if (
        not oracle["converged"]
        or oracle["natoms"] != 96
        or oracle["nao"] != 768
        or oracle["grid_points"] != 294912
    ):
        parser.error(
            "independent oracle must qualify the same complete 96-atom prescription"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for repetition in range(args.repeats):
        for mode in ("normal", "md-j") if repetition % 2 == 0 else ("md-j", "normal"):
            destination = args.output.with_name(
                f"{args.output.stem}-{repetition + 1}-{mode}.json"
            )
            environment = dict(os.environ, GENERATIVEQC_MD_J_COUNTS="1")
            if mode == "normal":
                environment["GENERATIVEQC_DISABLE_MD_J"] = "1"
            else:
                environment.pop("GENERATIVEQC_DISABLE_MD_J", None)
            command = [
                sys.executable,
                __file__,
                "--input",
                str(args.input.resolve()),
                "--output",
                str(destination.resolve()),
                "--worker",
                mode,
            ]
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=480,
            )
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", file=sys.stderr, flush=True)
            counts = [
                line
                for line in result.stderr.splitlines()
                if line.startswith("MD_J_NORMAL_COUNTS=")
            ]
            if len(counts) != 1:
                raise RuntimeError(
                    "complete production owner did not report its J execution census"
                )
            record = json.loads(destination.read_text())
            record["md_j_calls"] = int(counts[0].split("=", 1)[1])
            candidates = [
                line
                for line in result.stderr.splitlines()
                if line.startswith("MD_J_RESIDUAL_CANDIDATES=")
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "prepared J residual did not report its candidate census"
                )
            record["md_j_residual_candidate_upper_bound_per_fock"] = int(
                candidates[0].split("=", 1)[1]
            )
            record["md_j_residual_candidate_upper_bound"] = (
                record["md_j_residual_candidate_upper_bound_per_fock"]
                * record["md_j_calls"]
            )
            record["md_j_density_bound_builds"] = 2 * record["md_j_calls"]
            destination.write_text(json.dumps(record, indent=2) + "\n")
            samples.append(record)
            partial = collect(samples, oracle["energy"])
            args.output.write_text(json.dumps(partial, indent=2) + "\n")
    report = collect(samples, oracle["energy"])
    report["oracle_record_sha256"] = hashlib.sha256(
        args.oracle.read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "median_seconds",
                    "normal_over_md_speedup",
                    "performance_acceptance_passed",
                    "gate_failures",
                )
            }
        ),
        flush=True,
    )
    if report["gate_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

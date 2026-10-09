"""Matched cold FP64 Lambda reuse and forward-W/cadence controls under Slurm.

The production endpoint receives molecular input only. Independent oracle data
is read after execution, never supplied as RHF/CCSD/Lambda state. Clean and
instrumented observations are separate; nested operator times are never added
to their Lambda parent. Failed/incomplete runs retain raw diagnostics but do
not publish an accepted summary.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks._retention import raw_output_path


def checksum(path: Path) -> str:
    """Content identity for one frozen input, executable or shared library."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trace_categories(path: Path) -> dict:
    """Retain immediate Lambda children separately from their GMRES children."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    beginnings = {row["id"]: row for row in rows if row["event"] == "BEGIN"}
    endings = {row["id"]: row for row in rows if row["event"] == "END"}
    parents = [
        row for row in beginnings.values() if row["name"] == "cc_lambda_complete"
    ]
    if len(parents) != 1:
        raise ValueError("expected exactly one completed cold Lambda owner")
    parent = parents[0]
    children = [row for row in beginnings.values() if row["parent"] == parent["id"]]
    if any(row["id"] not in endings for row in [parent, *children]):
        raise ValueError("incomplete Lambda phase trace")
    phases = {row["name"]: endings[row["id"]]["elapsed_ms"] / 1000 for row in children}
    parent_seconds = endings[parent["id"]]["elapsed_ms"] / 1000
    if sum(phases.values()) > parent_seconds + 1e-6:
        raise ValueError("Lambda immediate-child times exceed their parent")
    gmres = next(row for row in children if row["name"] == "cc_lambda_gmres")
    actions = []
    for row in beginnings.values():
        if row["parent"] != gmres["id"]:
            continue
        if row["id"] not in endings:
            raise ValueError("incomplete physical GMRES action")
        actions.append(
            {
                "name": row["name"],
                "seconds": endings[row["id"]]["elapsed_ms"] / 1000,
                "work": {
                    value["key"]: value["value"]
                    for value in rows
                    if value["event"] == "VALUE" and value["id"] == row["id"]
                },
            }
        )
    return {
        "parent_seconds": parent_seconds,
        "immediate_children_seconds": phases,
        "unattributed_parent_seconds": parent_seconds - sum(phases.values()),
        "nested_physical_actions": actions,
    }


def scientific_gates(record: dict, oracle: dict, fd: dict, atoms: np.ndarray) -> dict:
    """Use original physical tolerances and independently retained PySCF energies."""
    if oracle["pyscf_version"] != "2.14.0":
        raise ValueError("requires the independently pinned PySCF 2.14.0 oracle")
    if record["resident_jk_discarded_attempt"] or record.get(
        "recycling_discarded_primal_attempt", False
    ):
        raise ValueError("discarded attempted work is not eligible for matched timing")
    energy_error = abs(record["total_energy"] - oracle["total_energy"])
    if not np.isfinite(energy_error) or energy_error > 3e-9:
        raise ValueError("independent same-Hamiltonian energy gate failed")
    for key in ("lambda_residual", "z_residual", "stationarity"):
        if not np.isfinite(record[key]) or record[key] > 1e-9:
            raise ValueError(f"exact physical {key} gate failed")
    force = np.asarray(record["forces"]).reshape(-1, 3)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=3e-8, rtol=0)
    errors = []
    for row in fd["rows"]:
        step = row["step_bohr"]
        minus, plus = row["minus"], row["plus"]
        if minus["pyscf_version"] != "2.14.0" or plus["pyscf_version"] != "2.14.0":
            raise ValueError("finite-difference reference version mismatch")
        low = np.asarray([position for _, position in minus["atoms_bohr"]])
        high = np.asarray([position for _, position in plus["atoms_bohr"]])
        np.testing.assert_allclose((high + low) / 2, atoms, atol=1e-12, rtol=0)
        direction = (high - low) / (2 * step)
        derivative = (plus["total_energy"] - minus["total_energy"]) / (2 * step)
        error = abs(float(np.sum(force * direction)) + derivative)
        if not np.isfinite(error) or error > 3e-7:
            raise ValueError("independent two-step directional force gate failed")
        errors.append({"step_bohr": step, "absolute_error": error})
    if {row["step_bohr"] for row in fd["rows"]} != {1e-4, 3e-5}:
        raise ValueError("both original finite-difference steps are mandatory")
    return {"energy_error": energy_error, "directional_force_errors": errors}


def summarize(observations: list[dict]) -> dict:
    """Medians/ranges are descriptive evidence, not a statistical qualification."""
    keys = (
        "native_seconds",
        "reference_seconds",
        "source_seconds",
        "ccsd_seconds",
        "triples_seconds",
        "lambda_seconds",
        "source_response_seconds",
        "orbital_seconds",
        "lambda_iterations",
        "lambda_actions",
        "lambda_gemm_calls",
        "lambda_work",
        "lambda_packing_output_bytes",
        "lambda_capacity",
    )
    grouped = {}
    for row in observations:
        grouped.setdefault(row["configuration"], []).append(row["endpoint"])
    return {
        name: {
            key: {
                "median": statistics.median(row[key] for row in rows),
                "minimum": min(row[key] for row in rows),
                "maximum": max(row[key] for row in rows),
            }
            for key in keys
        }
        for name, rows in grouped.items()
    }


def endpoint_controls(lambda_batch_limit: int = 8) -> list[str]:
    """Keep the established cold protocol while varying only Lambda Q batching."""
    if lambda_batch_limit < 1:
        raise ValueError("Lambda batch limit must be positive")
    return [
        "1",
        "1",
        "1",
        "1",
        str(lambda_batch_limit),
        "8",
        "8",
        "0",
        "0",
        "2",
        "1",
        "30",
        "0",
        "0",
        "1",
        "auto",
        "1",
        "0",
        "auto",
        "0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--finite-differences", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not os.environ.get("SLURM_JOB_ID") or not visibility:
        raise RuntimeError(
            "requires a finite Slurm GPU allocation with assigned visibility"
        )
    if args.repetitions < (1 if args.screen else 2) or args.timeout < 1:
        raise ValueError(
            "matched qualification requires repeated, bounded observations"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    retained_input = (
        Path(__file__).resolve().parent / "results/rccsd-diis-ring-1900/ethane230.input"
    )
    if checksum(args.input) != checksum(retained_input):
        raise ValueError(
            "this retained-oracle qualification requires the exact 230-AO input"
        )
    frozen = {
        str(path.resolve()): checksum(path)
        for path in (args.endpoint, args.library, args.input)
    }
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={visibility}",
            "--query-gpu=uuid,name,driver_version,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    fd = json.loads(args.finite_differences.read_text())
    lines = args.input.read_text().splitlines()
    atoms = np.asarray(
        [
            [float(value) for value in line.split()[1:]]
            for line in lines[1 : 1 + int(lines[0].split()[0])]
        ]
    )
    configurations = (
        [(0, 30, 0), (0, 30, 1)]
        if args.screen
        else list(itertools.product((0, 1), (1, 30), (0, 1)))
    )
    controls = endpoint_controls()
    observations = []
    for repetition in range(args.repetitions):
        order = configurations if repetition % 2 == 0 else configurations[::-1]
        for mixed, cadence, reuse in order:
            name = f"w{mixed}_c{cadence}_r{reuse}"
            prefix = args.output / f"{name}_sample{repetition}"
            env = dict(os.environ)
            env.pop("GENERATIVEQC_DF_PROGRESS_TRACE", None)
            if args.instrumented:
                env["GENERATIVEQC_DF_PROGRESS_TRACE"] = str(
                    prefix.with_suffix(".trace.jsonl")
                )
            command = [
                str(args.endpoint.resolve()),
                str(args.input.resolve()),
                str(prefix.with_suffix(".json").resolve()),
                *controls,
                str(mixed),
                str(cadence),
                str(reuse),
                str(reuse),
                "0",
            ]
            print(f"starting {prefix.name}", flush=True)
            started = time.perf_counter()
            with prefix.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=True,
                )
            endpoint = json.loads(prefix.with_suffix(".json").read_text())
            oracle = json.loads(args.oracle.read_text())
            row = {
                "configuration": name,
                "repetition": repetition,
                "command": command,
                "process_seconds": time.perf_counter() - started,
                "endpoint": endpoint,
                "scientific_gates": scientific_gates(endpoint, oracle, fd, atoms),
            }
            if args.instrumented:
                row["trace"] = trace_categories(prefix.with_suffix(".trace.jsonl"))
            for path, identity in frozen.items():
                if checksum(Path(path)) != identity:
                    raise RuntimeError(
                        "input or binary changed during frozen qualification"
                    )
            if observations:
                np.testing.assert_allclose(
                    endpoint["forces"],
                    observations[0]["endpoint"]["forces"],
                    atol=3e-7,
                    rtol=0,
                )
            observations.append(row)
            with (args.output / "observations.jsonl").open("a") as journal:
                journal.write(json.dumps(row) + "\n")
    result = {
        "schema": "generativeqc.df-lambda-core-reuse.ablation.v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": visibility,
        "gpu": gpu,
        "frozen_sha256": frozen,
        "instrumented": args.instrumented,
        "screen_only": args.screen,
        "repetitions": args.repetitions,
        "oracle_sha256": checksum(args.oracle),
        "finite_difference_sha256": checksum(args.finite_differences),
        "observations": observations,
        "summary": summarize(observations),
    }
    (args.output / "accepted-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

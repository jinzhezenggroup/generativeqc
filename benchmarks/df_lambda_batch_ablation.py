"""Cold strict-FP64 Lambda batching/replay experiments with unchanged gates.

One finite Slurm allocation owns the GPU for every fresh-process sample.
Screening is explicitly unqualified; matched results require forward/reverse
repetitions. Library/input identities are frozen and oracle files are read only
after production has completed. Neither partial runs nor failed samples publish
a completed summary. This is a scheduling ablation, not performance promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks.df_lambda_core_reuse_ablation import (
    checksum,
    endpoint_controls,
    scientific_gates,
    summarize,
    trace_categories,
)


def cold_command(
    endpoint: Path,
    molecular_input: Path,
    output: Path,
    batch: int,
    primal_matrix: bool = False,
) -> list[str]:
    """Preserve strict W/cadence thirty and both qualified optional optimizations."""
    command = [
        str(endpoint.resolve()),
        str(molecular_input.resolve()),
        str(output.resolve()),
        *endpoint_controls(batch),
        "0",
        "30",
        "1",
        "1",
        "1" if primal_matrix else "0",
    ]
    return command


def run_sweep(args: argparse.Namespace) -> dict:
    """Run bounded cold processes, publishing only after every sample passes."""
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not visibility:
        raise RuntimeError(
            "requires finite Slurm GPU allocation and assigned visibility"
        )
    limits = args.batch_limits
    if (
        not limits
        or any(limit < 1 for limit in limits)
        or len(set(limits)) != len(limits)
    ):
        raise ValueError("batch limits must be distinct positive integers")
    if 8 not in limits:
        raise ValueError("the qualified batch-eight baseline is mandatory")
    if args.repetitions < (1 if args.screen else 2) or args.timeout < 1:
        raise ValueError("matched qualification requires repeated bounded observations")
    if getattr(args, "default_candidate", False) and (
        getattr(args, "replay_candidates", False) or set(limits) != {8, 32}
    ):
        raise ValueError("default comparison requires only batches eight/thirty-two")
    retained_input = (
        Path(__file__).resolve().parent / "results/rccsd-diis-ring-1900/ethane230.input"
    )
    if checksum(args.input) != checksum(retained_input):
        raise ValueError("requires the exact independently qualified 230-AO input")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = (
        args.endpoint,
        args.library,
        args.input,
        Path(__file__),
        Path(__file__).with_name("df_lambda_core_reuse_ablation.py"),
    )
    frozen = {str(path.resolve()): checksum(path) for path in paths}
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={visibility}",
            "--query-gpu=uuid,name,driver_version,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    lines = args.input.read_text().splitlines()
    atom_rows = [line.split() for line in lines[1 : 1 + int(lines[0].split()[0])]]
    atoms = np.asarray([[float(value) for value in row[1:4]] for row in atom_rows])
    observations = []
    baseline = None
    # Start from the baseline, then reverse all cells to expose ordering drift.
    forward = [
        (limit, False) for limit in [8, *[limit for limit in limits if limit != 8]]
    ]
    if getattr(args, "default_candidate", False):
        forward = [(8, False), (32, True)]
    elif getattr(args, "replay_candidates", False):
        forward += [(limit, True) for limit, _ in forward]
    for repetition in range(args.repetitions):
        order = forward if repetition % 2 == 0 else forward[::-1]
        for limit, replay in order:
            configuration = f"b{limit}" + ("_p1" if replay else "")
            prefix = args.output / f"{configuration}_sample{repetition}"
            environment = dict(os.environ)
            environment.pop("GENERATIVEQC_DF_PROGRESS_TRACE", None)
            if args.instrumented:
                environment["GENERATIVEQC_DF_PROGRESS_TRACE"] = str(
                    prefix.with_suffix(".trace.jsonl")
                )
            command = cold_command(
                args.endpoint, args.input, prefix.with_suffix(".json"), limit, replay
            )
            print(f"starting {prefix.name}", flush=True)
            started = time.perf_counter()
            with prefix.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=True,
                )
            endpoint = json.loads(prefix.with_suffix(".json").read_text())
            # Oracle data is deliberately loaded only after the native endpoint.
            references = {}
            for path in (args.oracle, args.finite_differences):
                content = path.read_bytes()
                identity = hashlib.sha256(content).hexdigest()
                key = str(path.resolve())
                if frozen.setdefault(key, identity) != identity:
                    raise RuntimeError(
                        "independent reference changed during frozen ablation"
                    )
                references[key] = json.loads(content)
            oracle = references[str(args.oracle.resolve())]
            finite_differences = references[str(args.finite_differences.resolve())]
            gates = scientific_gates(endpoint, oracle, finite_differences, atoms)
            for field in (
                "lambda_matrix_gemm",
                "lambda_core_reuse",
                "lambda_audit_matrix",
            ):
                if not endpoint[field]:
                    raise ValueError(
                        f"{field} refused; not a comparable batch-only sample"
                    )
            actual = endpoint["lambda_batch_size"]
            if replay and not endpoint.get("lambda_primal_matrix", False):
                raise ValueError("requested original-graph matrix replay was refused")
            if actual < 1 or actual > limit:
                raise ValueError("invalid admitted Lambda batch size")
            if baseline is None:
                baseline = endpoint
            np.testing.assert_allclose(
                endpoint["forces"], baseline["forces"], atol=3e-7, rtol=0
            )
            for field in ("lambda_iterations", "lambda_actions"):
                if endpoint[field] != baseline[field]:
                    raise ValueError("batch-only comparison changed solver work")
            for path, identity in frozen.items():
                if checksum(Path(path)) != identity:
                    raise RuntimeError(
                        "source, input or binary changed during frozen ablation"
                    )
            row = {
                "configuration": configuration,
                "primal_matrix_requested": replay,
                "admitted_primal_matrix": endpoint.get("lambda_primal_matrix", False),
                "repetition": repetition,
                "requested_batch_limit": limit,
                "admitted_batch_size": actual,
                "command": command,
                "process_seconds": time.perf_counter() - started,
                "endpoint": endpoint,
                "scientific_gates": gates,
            }
            if args.instrumented:
                row["trace"] = trace_categories(prefix.with_suffix(".trace.jsonl"))
            observations.append(row)
            with (args.output / "observations.jsonl").open("a") as journal:
                journal.write(json.dumps(row) + "\n")
            print(
                f"finished {prefix.name}: E+F={endpoint['native_seconds']:.3f}s "
                f"Lambda={endpoint['lambda_seconds']:.3f}s actual_batch={actual}",
                flush=True,
            )
    result = {
        "schema": "generativeqc.df-lambda-batch.ablation.v1",
        "slurm_job_id": job,
        "cuda_visible_devices": visibility,
        "gpu": gpu,
        "frozen_sha256": frozen,
        "screen_only": args.screen,
        "instrumented": args.instrumented,
        "performance_promotion": False,
        "observations": observations,
        "summary": summarize(observations),
    }
    filename = "screen-summary.json" if args.screen else "matched-summary.json"
    (args.output / filename).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Parse the frozen endpoint protocol and execute a Slurm-owned batch sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint", "library", "input", "oracle", "finite-differences"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--batch-limits", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--replay-candidates", action="store_true")
    parser.add_argument("--default-candidate", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    run_sweep(parser.parse_args())


if __name__ == "__main__":
    main()

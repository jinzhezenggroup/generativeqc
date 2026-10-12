"""Measure cold DF-CCSD(T) endpoints, not isolated FP32 graph timings.

Run inside a finite Slurm GPU allocation. Each sample starts a fresh process;
only the CCSD warmup selector and residual switch target differ between modes.
Compilation and JSON serialization are outside the native endpoint timer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def digest(path: Path) -> str:
    """Bind measurement records to the actual executable, library and input."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def controls(forces: bool, threshold: float | None, *, auto: bool = False) -> list[str]:
    """Keep stock endpoint schedules; zero selects automatic CCSD Q tiling."""
    arguments = [
        str(value)
        for value in (
            1,
            1,
            int(forces),
            1,
            32,
            6,
            0,
            0,
            0,
            2,
            1,
            30,
            0,
            0,
            0,
            "auto",
            1,
            0,
            "auto",
            0,
            0,
            30,
            1,
            1,
            1,
            0,
            0,
            "auto" if auto else int(threshold is not None),
            0,
            threshold if threshold is not None else 1e-5,
        )
    ]
    return arguments[:-1] if auto else arguments


def maximum_force_error(left: dict, right: dict) -> float | None:
    """Energy-only records have no force error, rather than a fabricated zero."""
    if not left["forces_requested"]:
        return None
    differences = [
        abs(actual - expected)
        for actual, expected in zip(left["forces"], right["forces"], strict=True)
    ]
    return max(differences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--forces", action="store_true")
    parser.add_argument(
        "--auto", action="store_true", help="also qualify the native default policy"
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-patch", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("requires Slurm-assigned GPU visibility")
    if args.repeats < 1 or any(not 1e-7 <= value <= 1e-2 for value in args.thresholds):
        parser.error("require positive repeats and switch residuals in [1e-7, 1e-2]")
    if bool(args.source_commit) != bool(args.source_patch):
        parser.error("provide both source identity arguments for an exported snapshot")
    args.output.mkdir(parents=True, exist_ok=True)
    binary, library, source = (
        path.resolve() for path in (args.binary, args.library, args.input)
    )
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "LD_LIBRARY_PATH": str(library.parent)
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    patch = (
        args.source_patch.read_bytes()
        if args.source_patch
        else subprocess.check_output(["git", "diff", "--binary"], cwd=root)
    )
    source_commit = (
        args.source_commit
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    )
    provenance = {
        "source_commit": source_commit,
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "binary_sha256": digest(binary),
        "library_sha256": digest(library),
        "runner_sha256": digest(Path(__file__)),
        "input_sha256": digest(source),
        "job": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                "--query-gpu=uuid,name,driver_version",
                "--format=csv",
            ],
            text=True,
        ).strip(),
        "forces": args.forces,
        "auto": args.auto,
        "thresholds": args.thresholds,
        "repeats": args.repeats,
        "scope": "complete cold conventional-RHF + correlation-only-DF CCSD(T) endpoint",
    }
    (args.output / "source.patch").write_bytes(patch)
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    samples: list[dict] = []
    modes = [
        ("fp64", None),
        *((f"mixed-{value:g}", value) for value in args.thresholds),
        *([("auto", None)] if args.auto else []),
    ]
    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else list(reversed(modes))
        current = {}
        for name, threshold in order:
            output = args.output / f"{name}-{repeat + 1}.json"
            command = [
                str(binary),
                str(source),
                str(output),
                *controls(args.forces, threshold, auto=name == "auto"),
            ]
            print(f"repeat={repeat + 1} mode={name} started", flush=True)
            started = time.perf_counter()
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=log,
                    check=True,
                    timeout=args.timeout,
                    env=environment,
                )
            result = json.loads(output.read_text())
            sample = {
                "mode": name,
                "repeat": repeat + 1,
                "process_seconds": time.perf_counter() - started,
                "command": command,
                "result": result,
            }
            samples.append(sample)
            current[name] = result
            (args.output / "samples.json").write_text(
                json.dumps(samples, indent=2) + "\n"
            )
            if max(result["ccsd_replay_r1_max"], result["ccsd_replay_r2_max"]) > 1e-10:
                raise RuntimeError(f"{name}: independent FP64 replay residual failed")
            if (
                args.forces
                and max(
                    result["lambda_residual"],
                    result["z_residual"],
                    result["stationarity"],
                )
                > 1e-8
            ):
                raise RuntimeError(f"{name}: response certificate failed")
            print(
                f"mode={name} native={result['native_seconds']:.6f}s CCSD={result['ccsd_seconds']:.6f}s "
                f"evaluations={result['ccsd_evaluations']} FP32={result['ccsd_mixed_evaluations']}",
                flush=True,
            )
        baseline = current["fp64"]
        for name, _ in modes[1:]:
            candidate = current[name]
            energy_error = abs(candidate["total_energy"] - baseline["total_energy"])
            force_error = maximum_force_error(candidate, baseline)
            if energy_error > 2e-9 or (force_error is not None and force_error > 1e-7):
                raise RuntimeError(
                    f"{name}: energy/force comparison failed: {energy_error}, {force_error}"
                )
    summary = {}
    for name, _ in modes:
        selected = [sample for sample in samples if sample["mode"] == name]
        references = {
            sample["repeat"]: sample["result"]
            for sample in samples
            if sample["mode"] == "fp64"
        }
        summary[name] = {
            "native_seconds_median": statistics.median(
                sample["result"]["native_seconds"] for sample in selected
            ),
            "ccsd_seconds_median": statistics.median(
                sample["result"]["ccsd_seconds"] for sample in selected
            ),
            "process_seconds_median": statistics.median(
                sample["process_seconds"] for sample in selected
            ),
            "evaluations": [
                sample["result"]["ccsd_evaluations"] for sample in selected
            ],
            "mixed_evaluations": [
                sample["result"]["ccsd_mixed_evaluations"] for sample in selected
            ],
            "strict_evaluations": [
                sample["result"]["ccsd_evaluations"]
                - sample["result"]["ccsd_mixed_evaluations"]
                for sample in selected
            ],
            "maximum_paired_energy_error": max(
                abs(
                    sample["result"]["total_energy"]
                    - references[sample["repeat"]]["total_energy"]
                )
                for sample in selected
            ),
            "maximum_paired_force_error": (
                max(
                    maximum_force_error(sample["result"], references[sample["repeat"]])
                    for sample in selected
                )
                if args.forces
                else None
            ),
            "maximum_replay_residual": max(
                max(
                    sample["result"]["ccsd_replay_r1_max"],
                    sample["result"]["ccsd_replay_r2_max"],
                )
                for sample in selected
            ),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

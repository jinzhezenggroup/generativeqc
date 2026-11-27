"""Recompute issue #3185 acceptance and timing from retained endpoint samples."""

from __future__ import annotations

import argparse
import json
import lzma
import math
from pathlib import Path
from statistics import median

GATES = {"energy_per_atom": 1e-5, "force_max": 1e-4, "force_rms": 3e-5}
SYSTEMS = {"paper-water96": 8, "n-c32h66": 3}


def observable_errors(sample: dict, target: dict) -> dict:
    """Return absolute intensive energy and full Cartesian force errors."""
    forces, expected = sample["forces"], target["forces"]
    values = [
        first - second
        for row, ref in zip(forces, expected, strict=True)
        for first, second in zip(row, ref, strict=True)
    ]
    if len(values) != 3 * len(expected):
        raise ValueError("invalid Cartesian force shape")
    return {
        "energy_per_atom": abs(sample["energy"] - target["energy"]) / len(expected),
        "force_max": max(map(abs, values)),
        "force_rms": math.sqrt(math.fsum(value**2 for value in values) / len(values)),
    }


def passes(values: dict, fraction: float = 1.0) -> bool:
    """Apply the frozen gates and require finite numerical observations."""
    return all(
        math.isfinite(values[key]) and values[key] <= limit * fraction
        for key, limit in GATES.items()
    )


def compare_reference(sample: dict, reference: list[dict]) -> dict:
    """Require independent grid stability and a source-matched endpoint."""
    low, high = sorted(reference, key=lambda row: row["level"])
    if (
        [low["level"], high["level"]] != [4, 5]
        or not (low["complete"] and high["complete"])
        or low["geometry_sha256"] != high["geometry_sha256"]
        or sample["geometry_sha256"] != high["geometry_sha256"]
    ):
        raise ValueError("missing or inconsistent independent reference")
    grid = observable_errors(low, high)
    outcome = {
        "grid_stability": grid,
        "grid_stable": passes(grid, 0.25),
        "errors": None,
        "passed": False,
    }
    if sample["complete"]:
        outcome["errors"] = observable_errors(sample, high)
        outcome["passed"] = outcome["grid_stable"] and passes(outcome["errors"])
    return outcome


def trajectory_comparison(data: dict, system: str, failure_step: int) -> dict:
    """Keep failed and skipped steps distinct from qualified E+F timings."""
    references = data["reference"][system]
    if set(references) != {str(step) for step in (1, 5, 10, failure_step)}:
        raise ValueError("unexpected independent reference coverage")
    rows = []
    for repeat in (1, 2):
        arms = {
            arm: {
                sample["step"]: sample
                for sample in data["trajectory"][f"{arm}-{system}-r{repeat}"]["samples"]
            }
            for arm in ("master", "candidate")
        }
        for step in range(11):
            output = {"repeat": repeat, "step": step}
            for arm, steps in arms.items():
                sample = steps.get(step)
                if sample is None:
                    output[arm] = None
                    continue
                fields = (
                    "seconds",
                    "iterations",
                    "fock_builds",
                    "complete",
                    "status",
                    "native_detail",
                    "terminal_attempt_history_rows",
                    "terminal_attempt_fock_builds",
                    "seed_step",
                    "warm_start_fallback",
                )
                output[arm] = {key: sample[key] for key in fields}
                if str(step) in references:
                    output[arm]["observables"] = compare_reference(
                        sample, references[str(step)]
                    )
            rows.append(output)
    candidate_complete = all(
        row["candidate"] and row["candidate"]["complete"] for row in rows
    )
    candidate_gates = all(
        row["candidate"] and row["candidate"]["observables"]["passed"]
        for row in rows
        if str(row["step"]) in references
    )
    return {
        "steps": rows,
        "expected_baseline_failure_step": failure_step,
        "master_failures": [
            row["step"]
            for row in rows
            if not row["master"] or not row["master"]["complete"]
        ],
        "candidate_complete": candidate_complete,
        "candidate_observable_gates": candidate_gates,
        "candidate_moved_seconds": [
            math.fsum(
                row["candidate"]["seconds"]
                for row in rows
                if row["step"] and row["repeat"] == repeat
            )
            for repeat in (1, 2)
        ]
        if candidate_complete
        else None,
    }


def cold3_comparison(data: dict) -> dict:
    """Compare three fresh-process cold endpoints and paired warm medians."""
    result = {}
    for arm in ("master", "candidate"):
        processes = [data["cold3"][f"{arm}-cold3-r{repeat}"] for repeat in (1, 2, 3)]
        if any(process["status"] != "complete" for process in processes):
            result[arm] = None
            continue
        groups = [process["samples"] for process in processes]
        if any(
            [sample["phase"] for sample in group] != ["cold", "warm", "warm", "warm"]
            for group in groups
        ):
            raise ValueError("invalid cold3 phases")
        cold = [group[0]["seconds"] for group in groups]
        warm = [[sample["seconds"] for sample in group[1:]] for group in groups]
        result[arm] = {
            "cold_seconds": cold,
            "warm_seconds_by_process": warm,
            "iterations": [
                [sample["iterations"] for sample in group] for group in groups
            ],
            "fock_builds": [
                [sample["fock_builds"] for sample in group] for group in groups
            ],
            "cold_median": median(cold),
            "warm_process_median": median(median(group) for group in warm),
        }
    if all(result.values()):
        result["candidate_over_master"] = {
            key: result["candidate"][key] / result["master"][key]
            for key in ("cold_median", "warm_process_median")
        }
    return result


def summarize(data: dict) -> dict:
    """Evaluate the complete paired replay and the untouched cold cohort."""
    return {
        "gates": GATES,
        "grid_stability_fraction": 0.25,
        "trajectory": {
            system: trajectory_comparison(data, system, step)
            for system, step in SYSTEMS.items()
        },
        "cold3": cold3_comparison(data),
    }


def main() -> None:
    """Print the summary JSON without altering the retained data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("samples.json.xz"),
    )
    args = parser.parse_args()
    with lzma.open(args.samples, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    print(json.dumps(summarize(data), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

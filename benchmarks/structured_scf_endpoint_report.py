"""Fail-closed matched endpoint crossover report; never changes provider defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PHASES = ("cold", "warm_first", "warm", "geometry_refresh", "moved_warm")
IDENTITY_FIELDS = (
    "model",
    "source",
    "library_sha256",
    "batch",
    "graph",
    "host",
    "gpu_inventory",
    "cuda_visible_devices",
)


def endpoint_samples(receipt: dict) -> dict:
    """A batch is one timed endpoint, not batch-size independent timing samples."""
    if receipt.get("status") != "PASS" or receipt.get("instrumented", True):
        raise ValueError(
            "failed or instrumented endpoints cannot qualify profitability"
        )
    batch = receipt["batch"]
    samples = {}
    for phase, name in enumerate(PHASES):
        rows = [row for row in receipt["rows"] if row["phase"] == phase]
        if len(rows) != batch or {row["item"] for row in rows} != set(range(batch)):
            raise ValueError("missing or duplicate scientific batch members")
        if not all(row["assessment"]["accepted"] for row in rows):
            raise ValueError("every scientific row must pass independent gates")
        times = [
            row.get("cold_total_seconds", row["complete_seconds"])
            if phase == 0
            else row["complete_seconds"]
            for row in rows
        ]
        if not all(np.isfinite(value) and value > 0 for value in times):
            raise ValueError("missing, nonfinite or nonpositive endpoint cost")
        if len(set(times)) != 1:
            raise ValueError("batch members must share their complete endpoint cost")
        samples[name] = times[0]
    return samples


def compare(baseline: list[dict], candidate: list[dict], mask: int) -> dict:
    """Require three independent pairs, reported work and no hidden phase regression."""
    result = {"mask": mask, "profitable": None, "default_promoted": False}
    try:
        if len(baseline) != len(candidate) or len(baseline) < 3:
            raise ValueError(
                "at least three matched independent process pairs are required"
            )
        baseline = sorted(baseline, key=lambda value: value["repeat"])
        candidate = sorted(candidate, key=lambda value: value["repeat"])
        if len({value["repeat"] for value in baseline}) != len(baseline):
            raise ValueError("duplicate repetition identifiers")
        costs = {name: [] for name in PHASES}
        work = []
        for generated, offered in zip(baseline, candidate, strict=True):
            if generated["mask"] != 0 or offered["mask"] != mask:
                raise ValueError("incorrect ablation pair")
            if generated["repeat"] != offered["repeat"] or any(
                generated.get(field) is None or generated[field] != offered.get(field)
                for field in IDENTITY_FIELDS
            ):
                raise ValueError(
                    "source, library, scientific inputs or environment mismatch"
                )
            native_costs, library_costs = map(endpoint_samples, (generated, offered))
            for name in PHASES:
                costs[name].append(native_costs[name] / library_costs[name])
            work.append(
                {
                    "repeat": generated["repeat"],
                    "generated_iterations": [
                        row["iterations"] for row in generated["rows"]
                    ],
                    "candidate_iterations": [
                        row["iterations"] for row in offered["rows"]
                    ],
                    "generated_seconds": native_costs,
                    "candidate_seconds": library_costs,
                }
            )
        ratios = {
            name: {
                "median_speedup": float(np.median(values)),
                "lower_quartile_speedup": float(np.quantile(values, 0.25)),
                "paired_speedups": values,
            }
            for name, values in costs.items()
        }
        targets = ("cold",) if mask == 4 else ("warm", "moved_warm")
        if mask == 120:
            targets = ("cold", "warm", "geometry_refresh", "moved_warm")
        if mask == 127:
            targets = PHASES
        result.update(
            {
                "matched_pairs": len(work),
                "phases": ratios,
                "semantic_work": work,
                "policy": {
                    "target_phases": targets,
                    "minimum_median_speedup": 1.03,
                    "minimum_target_lower_quartile": 1.0,
                    "minimum_guard_lower_quartile": 0.95,
                },
                "profitable": all(
                    ratios[name]["median_speedup"] >= 1.03
                    and ratios[name]["lower_quartile_speedup"] >= 1.0
                    for name in targets
                )
                and all(
                    value["lower_quartile_speedup"] >= 0.95 for value in ratios.values()
                ),
            }
        )
        if mask == 2 and candidate[0]["model"]["method"].endswith("hf"):
            result["profitable"] = None
            result["reason"] = (
                "HF retains determinant-consistent P F(P) P; signed Gram is not executed"
            )
    except (KeyError, TypeError, ValueError) as error:
        result["reason"] = str(error)
    return result


def report(receipts: list[dict]) -> dict:
    """Keep incompatible environments and failed receipts separate, never cherry-pick."""
    groups = {}
    for receipt in receipts:
        identity = json.dumps(
            {field: receipt.get(field) for field in IDENTITY_FIELDS}, sort_keys=True
        )
        groups.setdefault(identity, {}).setdefault(receipt["mask"], []).append(receipt)
    return {
        "default_promoted": False,
        "profiles": [
            {
                "identity": json.loads(identity),
                "comparisons": [
                    compare(masks.get(0, []), values, mask)
                    for mask, values in sorted(masks.items())
                    if mask != 0
                ],
            }
            for identity, masks in sorted(groups.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report([json.loads(path.read_text()) for path in args.receipts])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

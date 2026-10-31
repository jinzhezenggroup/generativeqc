"""Recompute the historical size, compilation and runtime tradeoff receipts."""

from __future__ import annotations

import json
import lzma
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as stream:
        data = json.load(stream)
    groups = defaultdict(dict)
    for row in data["point_samples"]["records"]:
        key = row["name"], row["spins"], row["points"], row["tile"]
        groups[key][row["revision"]] = statistics.median(row["milliseconds"])
    ratios = [row["candidate"] / row["baseline"] for row in groups.values()]
    assert max(ratios) <= 1.05
    endpoints = defaultdict(lambda: defaultdict(list))
    max_energy_error = 0.0
    for name, before in data["endpoints"].items():
        if "-baseline-" not in name:
            continue
        after = data["endpoints"][name.replace("-baseline-", "-candidate-")]
        for field in (
            "name",
            "spins",
            "copies",
            "atoms",
            "moved",
            "grid",
            "tile_points",
            "energy_tolerance",
            "density_tolerance",
            "max_iterations",
        ):
            assert before[field] == after[field]
        for left, right in zip(before["samples"], after["samples"], strict=True):
            assert left["phase"] == right["phase"]
            assert left["iterations"] == right["iterations"]
            assert left["fock_builds"] == right["fock_builds"]
            assert left["precision"] == right["precision"]
            max_energy_error = max(
                max_energy_error, abs(left["energy"] - right["energy"])
            )
        case = "water4" if before["atoms"][0][0] == "O" else "H4"
        group = f"{case}/{before['name']}/{before['spins']}"
        for arm, record in (("baseline", before), ("candidate", after)):
            for phase in ("cold", "warm", "moved"):
                endpoints[group][arm, phase].append(
                    statistics.median(
                        row["seconds"]
                        for row in record["samples"]
                        if row["phase"].startswith(phase)
                    )
                )
    assert max_energy_error <= 3e-9
    timing = {}
    for case, rows in endpoints.items():
        timing[case] = {}
        for phase in ("cold", "warm", "moved"):
            before, after = [
                statistics.median(rows[arm, phase]) for arm in ("baseline", "candidate")
            ]
            ratio = after / before
            assert ratio <= 1.05, (case, phase, ratio)
            timing[case][phase] = {
                "baseline_seconds": before,
                "candidate_seconds": after,
                "ratio": ratio,
            }
    for oracle in data["oracle_receipts"].values():
        for row in oracle["rows"]:
            assert row["energy_error"] < 3e-9
            assert row["fixed_energy_error"] < 3e-10
            assert row["density_max_abs_error"] < 2e-7
            assert row["fock_max_abs_error"] < 3e-9
            assert row["directional_error"] < 5e-8
    compile_trial = data["compile_trial"]["records"]
    sizes = data["cmake_object_bytes"]
    assert sizes["candidate"] < sizes["baseline"]
    counts = {
        arm: sum("automatic_evaluate_points" in row["symbol"] for row in values)
        for arm, values in data["resources"].items()
    }
    print(
        json.dumps(
            {
                "default": "literal specialization; sharing is opt-in",
                "point_kernel_counts": counts,
                "object_bytes": sizes,
                "object_byte_reduction": 1 - sizes["candidate"] / sizes["baseline"],
                "compile_seconds": {
                    arm: row["compile_seconds"] for arm, row in compile_trial.items()
                },
                "generation_seconds": {
                    arm: row["generation_seconds"] for arm, row in compile_trial.items()
                },
                "point_median_ratio": statistics.median(ratios),
                "point_maximum_ratio": max(ratios),
                "maximum_energy_error": max_energy_error,
                "endpoints": timing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

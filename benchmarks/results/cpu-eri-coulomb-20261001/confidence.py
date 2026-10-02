"""Reproduce process-pair bootstrap intervals from retained complete endpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from verify import HERE, members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = dict(members())
    campaign = "matched-endpoints-bytecode-normalized"
    status = json.loads(records[f"{campaign}/run-status.json"])
    processes = {
        (p["case"], p["engine"], p["repeat"]): json.loads(
            records[f"{campaign}/{p['json_file']}"]
        )["rows"]
        for p in status
    }
    rng = np.random.default_rng(20261001)
    result = {
        "seed": 20261001,
        "resamples": 20000,
        "resampling_unit": "paired independent processes; two warm calls reduced to a process median",
        "rows": {},
    }
    for case in ["h2-sto3g", "water-sto3g", "water-svp", "formaldehyde-svp"]:
        result["rows"][case] = {}
        for phase in ["cold", "warm", "changed_geometry"]:
            ratios = []
            for repeat in range(1, 21):
                medians = {
                    engine: statistics.median(
                        row["wall_seconds"]
                        for row in processes[case, engine, repeat]
                        if row["phase"] == phase
                    )
                    for engine in ["baseline", "candidate"]
                }
                ratios.append(medians["candidate"] / medians["baseline"])
            samples = np.median(
                rng.choice(ratios, (20000, len(ratios)), replace=True), axis=1
            )
            result["rows"][case][phase] = {
                "paired_candidate_over_baseline": ratios,
                "median": statistics.median(ratios),
                "bootstrap_95_percent_interval": np.percentile(
                    samples, [2.5, 97.5]
                ).tolist(),
                "faster_pairs": sum(r < 1 for r in ratios),
                "pair_count": len(ratios),
            }
    expected = json.loads((HERE / "paired-process-confidence.json").read_text())
    if result != expected:
        raise ValueError("Computed intervals differ from the published record")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("All 12 process-pair estimates and percentile intervals reproduce exactly.")


if __name__ == "__main__":
    main()

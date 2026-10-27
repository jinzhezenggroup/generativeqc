"""Recompute descriptive medians and all-pairs force parity from retained samples.

This is an exploratory summary, not the repository's performance promotion test.
It has no GPU dependency and does not consume historical reference timings.
"""

from __future__ import annotations

import json
import lzma
import statistics
from pathlib import Path


def main() -> None:
    """Read this publication's raw final-source samples and print derived values."""
    root = Path(__file__).resolve().parents[1]
    with lzma.open(root / "samples.json.xz", "rt") as stream:
        samples = json.load(stream)
    for nbf in sorted({row["nbf"] for row in samples}):
        arms = {
            arm: [
                row for row in samples if row["nbf"] == nbf and row["selection"] == arm
            ]
            for arm in ("baseline", "candidate")
        }
        medians = {
            arm: statistics.median(row["whole_process_seconds"] for row in rows)
            for arm, rows in arms.items()
        }
        force_error = max(
            abs(x - y)
            for baseline in arms["baseline"]
            for candidate in arms["candidate"]
            for x, y in zip(
                baseline["endpoint"]["forces"],
                candidate["endpoint"]["forces"],
                strict=True,
            )
        )
        print(
            json.dumps(
                {
                    "nbf": nbf,
                    "samples_per_arm": {arm: len(rows) for arm, rows in arms.items()},
                    "wall_seconds_median": medians,
                    "descriptive_gain_percent": 100
                    * (1 - medians["candidate"] / medians["baseline"]),
                    "maximum_cross_arm_force_difference": force_error,
                    "performance_promotion": False,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

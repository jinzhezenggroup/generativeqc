"""Compare the conventional C5–C7 endpoints and retain incomplete outcomes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

Record = dict[str, Any]
ROOT = Path(__file__).resolve().parent
CASES = {"C5": 17, "C6": 20, "C7": 23}
ARMS = ("native-conv", "psi4-conv", "pyscf-conv")
LIMITS = {"energy": 1e-7, "max_force": 3e-6, "rms_force": 1e-6}


def completed(attempts: list[Record], case: str, arm: str) -> list[Record]:
    accepted = []
    seen = set()
    for attempt in attempts:
        label = attempt["label"]
        row = attempt["record"]
        if attempt["arm"] != arm or not label.startswith(case + "."):
            continue
        if row.get("status") != "completed":
            continue
        if (
            label in seen
            or label != f"{row['case']}.{row['trial']}"
            or row["arm"] != arm
            or row["displacement"] is not None
            or not math.isfinite(row["energy_hartree"])
            or not math.isfinite(row["endpoint_seconds"])
            or row["endpoint_seconds"] <= 0
        ):
            raise ValueError(
                f"Invalid completed identity or scalar fields for {arm}/{label}"
            )
        force = row["force_hartree_per_bohr"]
        if (
            not force
            or len(force) != CASES[case]
            or any(len(xyz) != 3 or not all(map(math.isfinite, xyz)) for xyz in force)
        ):
            raise ValueError(f"Invalid complete Cartesian force for {arm}/{label}")
        seen.add(label)
        accepted.append(row)
    return accepted


def compare(native: Record, reference: Record) -> Record:
    diffs = [
        x - y
        for left, right in zip(
            native["force_hartree_per_bohr"],
            reference["force_hartree_per_bohr"],
            strict=True,
        )
        for x, y in zip(left, right, strict=True)
    ]
    return {
        "native_trial": native["trial"],
        "reference_trial": reference["trial"],
        "energy": abs(native["energy_hartree"] - reference["energy_hartree"]),
        "max_force": max(map(abs, diffs)),
        "rms_force": math.sqrt(statistics.mean(x * x for x in diffs)),
    }


def summarize(attempts: list[Record]) -> Record:
    for attempt in attempts:
        raw = (json.dumps(attempt["record"], indent=2, allow_nan=False) + "\n").encode()
        if hashlib.sha256(raw).hexdigest() != attempt["record_sha256"]:
            raise ValueError(
                f"Raw record checksum differs for {attempt['arm']}/{attempt['label']}"
            )
    molecules = {}
    for case in CASES:
        selected = {arm: completed(attempts, case, arm) for arm in ARMS}
        timings = {
            arm: {
                "samples": values,
                "median": statistics.median(values) if len(values) >= 2 else None,
            }
            for arm, rows in selected.items()
            for values in [[row["endpoint_seconds"] for row in rows]]
        }
        references = {}
        for arm in ARMS[1:]:
            pairs = [
                compare(native, reference)
                for native, reference in itertools.product(
                    selected["native-conv"], selected[arm]
                )
            ]
            worst = (
                {key: max(pair[key] for pair in pairs) for key in LIMITS}
                if pairs
                else {}
            )
            references[arm] = {
                "pairs": pairs,
                "maxima": worst,
                "status": (
                    "outside_tolerance"
                    if any(value > LIMITS[key] for key, value in worst.items())
                    else "passed"
                    if len(selected["native-conv"]) >= 2 and pairs
                    else "incomplete"
                ),
            }
        outcomes = {entry["status"] for entry in references.values()}
        verdict = (
            "outside_tolerance"
            if "outside_tolerance" in outcomes
            else "passed"
            if "passed" in outcomes
            else "incomplete"
        )
        molecules[case] = {
            "status": verdict,
            "timing_seconds": timings,
            "comparisons": references,
        }
    return {"limits": LIMITS, "cases": molecules}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        metavar="PATH",
        type=Path,
        nargs="+",
        default=[ROOT / f"records-{case}.json" for case in CASES],
        help="Read one combined record file or multiple case files.",
    )
    args = parser.parse_args()
    attempts = []
    for path in args.records:
        attempts.extend(json.loads(path.read_text())["attempts"])
    print(json.dumps(summarize(attempts), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

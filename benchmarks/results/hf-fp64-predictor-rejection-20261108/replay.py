"""Replay bounded HF96 handoff experiments without hiding negative arms."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
support = runpy.run_path(
    str(ROOT.parent / "hf-screened-handoff-rejections-20261108" / "replay_support.py")
)
load_observations = support["load_observations"]


def require(condition: bool, message: str) -> None:
    """Authenticate retained inputs even with Python optimization enabled."""
    if not condition:
        raise ValueError(message)


def replay(directory: Path = ROOT) -> dict:
    """Recompute every independent gate and matched warm, not only medians."""
    cohort = json.loads((directory / "cohort.json").read_text())
    files = load_observations(directory, cohort)
    build_entry = cohort["build_log"]
    build_log = files[build_entry["member"]].encode()
    require(len(build_log) == build_entry["uncompressed_bytes"], "incomplete build log")
    require(
        hashlib.sha256(build_log).hexdigest() == build_entry["uncompressed_sha256"],
        "build log changed",
    )
    reference = json.loads(files["reference/oracle.json"])
    geometry = json.loads(files["reference/geometry.json"])
    require(len(geometry) == len(reference["forces"]) == 96, "reference shape")
    summary = {
        "decision": "rejected",
        "default_promotion_qualified": False,
        "cards": {},
    }
    for card, prefix in cohort["cards"].items():
        require(files[prefix + "/exit-code.txt"].strip() == "0", "native job failed")
        arms = []
        for ordinal, arm in enumerate(
            ("control", "candidate", "candidate", "control"), 1
        ):
            record = json.loads(files[f"{prefix}/hf96-{ordinal}-{arm}.json"])
            require(record["geometry_bohr"] == geometry, "geometry mismatch")
            for name in ("source_identity", "library_sha256"):
                require(record["native_build"][name] == cohort[name], name)
            require(record["xy_control"] is None, "nonautomatic XY")
            require(
                record["cold_screening_control"]
                == ("1" if arm == "candidate" else None),
                "arm",
            )
            require(
                record["effective_controls"] == cohort["public_controls"],
                "public controls",
            )
            samples = record["samples"]
            require(len(samples) == 4, "incomplete cold/warm population")
            errors = []
            for repeat, sample in enumerate(samples):
                require(sample["converged"], "unconverged endpoint")
                require(sample["phase"] == ("warm" if repeat else "cold"), "phase")
                require(
                    math.isfinite(sample["seconds"]) and sample["seconds"] > 0, "time"
                )
                require(len(sample["forces"]) == 96, "force atom count")
                require(
                    sample["iterations"]
                    == sample["screened_stage_iterations"]
                    + sample["strict_target_iterations"],
                    "stage work",
                )
                differences = []
                for actual, expected in zip(
                    sample["forces"], reference["forces"], strict=True
                ):
                    require(len(actual) == len(expected) == 3, "Cartesian shape")
                    differences.extend(
                        abs(left - right)
                        for left, right in zip(actual, expected, strict=True)
                    )
                energy = abs(sample["energy"] - reference["energy"])
                require(
                    math.isfinite(energy)
                    and all(math.isfinite(value) for value in differences),
                    "nonfinite observable",
                )
                errors.append(
                    {
                        "energy": energy,
                        "force": max(differences),
                        "passed": energy < 1e-8 and max(differences) < 1e-7,
                    }
                )
            cold = samples[0]
            arms.append(
                {
                    "arm": arm,
                    "cold_seconds": cold["seconds"],
                    "iterations": cold["iterations"],
                    "focks": cold["fock_builds"],
                    "screened_iterations": cold["screened_stage_iterations"],
                    "strict_iterations": cold["strict_target_iterations"],
                    "initial_guess": cold["initial_guess"],
                    "warm_seconds": [sample["seconds"] for sample in samples[1:]],
                    "independent_errors": errors,
                }
            )
        cold_ratio = statistics.mean(
            arm["cold_seconds"] for arm in arms if arm["arm"] == "candidate"
        ) / statistics.mean(
            arm["cold_seconds"] for arm in arms if arm["arm"] == "control"
        )
        warm_ratios = [
            candidate / control
            for candidate_arm, control_arm in ((arms[1], arms[0]), (arms[2], arms[3]))
            for candidate, control in zip(
                candidate_arm["warm_seconds"], control_arm["warm_seconds"], strict=True
            )
        ]
        summary["cards"][card] = {
            "arms": arms,
            "candidate_cold_ratio": cold_ratio,
            "paired_warm_ratios": warm_ratios,
            "every_warm_within_3_percent": all(value <= 1.03 for value in warm_ratios),
            "all_independent_passed": all(
                error["passed"] for arm in arms for error in arm["independent_errors"]
            ),
            "credible_cold_win": cold_ratio <= 0.98,
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(support["compact_summary"](replay()), indent=2, allow_nan=False))

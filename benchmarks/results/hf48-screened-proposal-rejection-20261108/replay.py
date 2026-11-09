"""Replay the rejected HF48 expansion, keeping scientific failures visible."""

from __future__ import annotations

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
    """Fail closed, including with Python optimization enabled."""
    if not condition:
        raise ValueError(message)


def replay(directory: Path = ROOT) -> dict:
    """Keep negative timings and every failed independent gate in the result."""
    cohort = json.loads((directory / "cohort.json").read_text())
    files = load_observations(directory, cohort)
    reference = json.loads(files[cohort["reference"]])
    summary = {
        "decision": cohort["decision"],
        "current_head_qualification": False,
        "cards": {},
    }
    for card, prefix in cohort["cards"].items():
        require(files[prefix + "/exit-code.txt"].strip() == "0", "failed native job")
        arms = []
        for ordinal, arm in enumerate(
            ("control", "candidate", "candidate", "control"), start=1
        ):
            record = json.loads(files[f"{prefix}/hf48-{ordinal}-{arm}.json"])
            require(
                record["geometry_bohr"] == reference["geometry_bohr"],
                "geometry mismatch",
            )
            require(
                record["native_build"]["source_identity"] == cohort["source_identity"],
                "source",
            )
            require(
                record["native_build"]["library_sha256"] == cohort["library_sha256"],
                "ELF",
            )
            require(record["xy_control"] is None, "nonautomatic XY")
            require(
                record["cold_screening_control"]
                == ("1" if arm == "candidate" else None),
                "arm",
            )
            samples = record["samples"]
            require(len(samples) == 4, "incomplete endpoint population")
            errors = []
            for repeat, sample in enumerate(samples):
                require(sample["converged"], "unconverged endpoint")
                require(sample["phase"] == ("warm" if repeat else "cold"), "phase")
                require(
                    math.isfinite(sample["seconds"]) and sample["seconds"] > 0, "time"
                )
                require(
                    len(sample["forces"]) == len(reference["forces"]) == 48,
                    "force shape",
                )
                differences = []
                for actual, target in zip(
                    sample["forces"], reference["forces"], strict=True
                ):
                    require(len(actual) == len(target) == 3, "Cartesian shape")
                    differences.extend(
                        abs(left - right)
                        for left, right in zip(actual, target, strict=True)
                    )
                energy = abs(sample["energy"] - reference["energy"])
                require(
                    math.isfinite(energy)
                    and all(math.isfinite(value) for value in differences),
                    "nonfinite",
                )
                errors.append(
                    {
                        "energy": energy,
                        "force": max(differences),
                        "passed": energy < 1e-8 and max(differences) < 1e-7,
                    }
                )
            arms.append(
                {
                    "arm": arm,
                    "cold_seconds": samples[0]["seconds"],
                    "cold_iterations": samples[0]["iterations"],
                    "cold_focks": samples[0]["fock_builds"],
                    "warm_seconds": [sample["seconds"] for sample in samples[1:]],
                    "independent_errors": errors,
                }
            )
        warm_ratios = [
            candidate / control
            for candidate_arm, control_arm in ((arms[1], arms[0]), (arms[2], arms[3]))
            for candidate, control in zip(
                candidate_arm["warm_seconds"], control_arm["warm_seconds"], strict=True
            )
        ]
        cold_ratio = statistics.mean(
            arm["cold_seconds"] for arm in arms if arm["arm"] == "candidate"
        ) / statistics.mean(
            arm["cold_seconds"] for arm in arms if arm["arm"] == "control"
        )
        summary["cards"][card] = {
            "arms": arms,
            "candidate_cold_ratio": cold_ratio,
            "paired_warm_ratios": warm_ratios,
            "all_warms_within_3_percent": all(value <= 1.03 for value in warm_ratios),
            "credible_cold_win": cold_ratio <= 0.98,
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(support["compact_summary"](replay()), indent=2, allow_nan=False))

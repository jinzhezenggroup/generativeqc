"""Replay immutable observations without turning a retained failed gate green."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import runpy
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
load_observations = runpy.run_path(
    str(ROOT.parent / "hf-screened-handoff-rejections-20261108" / "replay_support.py")
)["load_observations"]


def require(condition: bool, message: str) -> None:
    """Fail closed even when the interpreter runs with optimization enabled."""
    if not condition:
        raise ValueError(message)


def load(directory: Path = ROOT) -> tuple[dict, dict]:
    """Authenticate exact retained UTF-8 bytes, including unsuccessful jobs."""
    cohort = json.loads((directory / "cohort.json").read_text())
    return cohort, load_observations(directory, cohort)


def observable_error(actual: dict, target: dict) -> dict:
    """Use unchanged absolute E/F gates, never native-to-native parity alone."""
    actual_forces, expected_forces = actual["forces"], target["forces"]
    require(len(actual_forces) == len(expected_forces) > 0, "wrong force shape")
    differences = []
    for actual_row, expected_row in zip(actual_forces, expected_forces, strict=True):
        require(len(actual_row) == len(expected_row) == 3, "wrong Cartesian shape")
        differences.extend(
            abs(left - right)
            for left, right in zip(actual_row, expected_row, strict=True)
        )
    energy = abs(actual["energy"] - target["energy"])
    require(
        math.isfinite(energy) and all(math.isfinite(value) for value in differences),
        "nonfinite independent observable",
    )
    force = max(differences)
    return {
        "energy": energy,
        "force": force,
        "passed": bool(actual["converged"] and energy < 1e-8 and force < 1e-7),
    }


def replay(directory: Path = ROOT) -> dict:
    """Report performance, parity, strict requests and default failures separately."""
    cohort, capsule = load(directory)

    def record(name: str) -> dict:
        return json.loads(capsule[name])

    references = {
        size: record(name) for size, name in cohort["reference_files"].items()
    }
    summary = {
        "source_identity": cohort["source_identity"],
        "cards": {},
        "independent_gates": {"absolute_energy": 1e-8, "maximum_force": 1e-7},
        "paired_warm_ratio_ceiling": 1.03,
    }
    promotion = True
    for card in ("a100", "5090"):
        cases = {}
        for atoms in (96, 48):
            populations = []
            for ordinal, arm in enumerate(
                ("control", "candidate", "candidate", "control"), 1
            ):
                name = f"final-{card}/hf-{atoms}-{ordinal}-{arm}.json"
                data = record(name)
                require(
                    data["status"] == "complete" and len(data["samples"]) == 4,
                    "incomplete population",
                )
                require(
                    data["method"] == "hf"
                    and data["mode"] == "direct"
                    and data["arm"] == "standard"
                    and data["atoms"] == atoms,
                    "changed scientific endpoint",
                )
                require(
                    data["native_build"]["source_identity"] == cohort["source_identity"]
                    and data["native_build"]["library_sha256"]
                    == cohort["library_sha256"],
                    "wrong measured image",
                )
                for sample in data["samples"]:
                    require(
                        sample["converged"]
                        and math.isfinite(sample["seconds"])
                        and sample["seconds"] > 0,
                        "incomplete timed result",
                    )
                    require(
                        (
                            sample["settings"]["energy_tolerance"],
                            sample["settings"]["density_tolerance"],
                            sample["settings"]["screening_tolerance"],
                        )
                        == (1e-9, 1e-5, 1e-12),
                        "strict-request solve relabeled as default",
                    )
                populations.append(data)
            original = populations[0]
            if "geometry_bohr" in references[str(atoms)]:
                require(
                    original["geometry_bohr"]
                    == references[str(atoms)]["geometry_bohr"],
                    "changed independent geometry",
                )
            for data in populations:
                for key in (
                    "geometry_bohr",
                    "orbital_basis",
                    "basis_representation",
                    "charge",
                    "spin",
                    "protocol",
                ):
                    require(data[key] == original[key], "changed paired input: " + key)
                for sample in data["samples"]:
                    require(
                        sample["settings"] == original["samples"][0]["settings"],
                        "changed paired controls",
                    )
            samples = [sample for data in populations for sample in data["samples"]]
            parity = [
                observable_error(left, right)
                for left, right in itertools.combinations(samples, 2)
            ]
            require(len(parity) == 120, "missing native sample pairs")
            paired = []
            for control_index, candidate_index in ((0, 1), (3, 2)):
                ratios = [
                    candidate["seconds"] / control["seconds"]
                    for control, candidate in zip(
                        populations[control_index]["samples"],
                        populations[candidate_index]["samples"],
                        strict=True,
                    )
                ]
                paired.append(
                    {
                        "candidate_over_control": ratios,
                        "each_warm_within_3pct": all(
                            value <= 1.03 for value in ratios[1:]
                        ),
                    }
                )
            independent = [
                observable_error(sample, references[str(atoms)]) for sample in samples
            ]
            case = {
                "cold_median_seconds": {
                    "control": statistics.median(
                        populations[index]["samples"][0]["seconds"] for index in (0, 3)
                    ),
                    "candidate": statistics.median(
                        populations[index]["samples"][0]["seconds"] for index in (1, 2)
                    ),
                },
                "paired_timing": paired,
                "all_120_native_pairs_pass": all(item["passed"] for item in parity),
                "independent_default_observations": independent,
                "target_iterations": [
                    data["samples"][0]["iterations"] for data in populations
                ],
                "target_fock_builds": [
                    data["samples"][0]["fock_builds"] for data in populations
                ],
                "initial_guess_fock_builds": [
                    data["samples"][0]["initial_guess"]["preliminary_fock_builds"]
                    for data in populations
                ],
                "quartet_work_counters_valid": [
                    data["samples"][0]["incremental_direct_jk"][
                        "quartet_work_counters_valid"
                    ]
                    for data in populations
                ],
            }
            promotion &= case["all_120_native_pairs_pass"] and all(
                item["passed"] for item in independent
            )
            promotion &= all(item["each_warm_within_3pct"] for item in paired)
            cases[str(atoms)] = case
        strict = {}
        for arm in ("control", "candidate"):
            data = record(f"final-{card}/hf48-tight-{arm}.json")
            require(
                len(data["samples"]) == 4
                and (
                    data["energy_tolerance"],
                    data["density_tolerance"],
                    data["screening_tolerance"],
                )
                == (1e-12, 1e-10, 1e-14),
                "changed strict diagnostic",
            )
            strict[arm] = [
                observable_error(sample, references["48"]) for sample in data["samples"]
            ]
            promotion &= all(item["passed"] for item in strict[arm])
        moved_target = record("history-qualification-a100/extra-reference.json")[
            "observations"
        ][1]
        extra = record(f"final-{card}/extra-native.json")["observations"]
        require(
            [item["name"] for item in extra] == ["hf96-moved", "hf96-restored"],
            "missing displaced/restored endpoint",
        )
        restored = []
        for item, target in zip(extra, (moved_target, references["96"]), strict=True):
            error = observable_error(item, target)
            restored.append(
                {
                    "name": item["name"],
                    **error,
                    "iterations": item["iterations"],
                    "fock_builds": item["fock_builds"],
                }
            )
            promotion &= error["passed"]
        summary["cards"][card] = {
            "default_cases": cases,
            "strict_request_hf48": strict,
            "moved_and_restored_hf96": restored,
        }
    summary["independent_default_promotion_qualified"] = bool(promotion)
    return summary


def main() -> None:
    """A successful replay may truthfully report a failed promotion gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-promotion", action="store_true")
    arguments = parser.parse_args()
    summary = replay()
    print(json.dumps(summary, indent=2, allow_nan=False))
    if (
        arguments.require_promotion
        and not summary["independent_default_promotion_qualified"]
    ):
        raise SystemExit(
            "Independent default promotion remains blocked; retained failures are not passes"
        )


if __name__ == "__main__":
    main()

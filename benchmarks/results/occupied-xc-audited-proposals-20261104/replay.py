"""Replay complete occupied-XC endpoints, including rejected scaling cohorts."""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
from pathlib import Path

from benchmarks.occupied_xc_proposals import check_vectors, load, replay
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT.parent / "readme-final-20261022"


def scaling(cohort: dict) -> dict:
    """Recheck dense references and every paired vector before deriving costs."""
    records = cohort["records"]
    expected = {f"r{repeat}-{selector}" for repeat in range(3) for selector in (0, 1)}
    if set(records) != expected:
        raise ValueError("incomplete scaling population")
    oracles = load(REFERENCES / "oracles.json.xz")
    reuse = load(REFERENCES / "reference-reuse.json")
    for name, record in records.items():
        if (
            record["status"] != "complete"
            or record["source_revision"] != cohort["source_commit"]
            or record["native_build"]["library_sha256"] != cohort["library_sha256"]
            or (record["method"], record["atoms"], record["mode"], record["arm"])
            != ("r2scan-3c", 96, "direct", "standard")
            or len(record["samples"]) != 2
        ):
            raise ValueError(f"invalid scaling source or endpoint: {name}")
        qualified = qualify([record], oracles, reference_reuse=reuse)[0]
        if qualified["status"] != "qualified":
            raise ValueError(f"independent scaling gate failed: {name}")
        for sample in record["samples"]:
            if sample["fock_builds"] < sample["iterations"]:
                raise ValueError("incomplete Fock work")
    result = {}
    for index, phase in enumerate(("cold", "warm")):
        populations = {selector: [] for selector in (0, 1)}
        work = {selector: [] for selector in (0, 1)}
        for repeat in range(3):
            baseline, proposal = (
                records[f"r{repeat}-{selector}"]["samples"][index]
                for selector in (0, 1)
            )
            check_vectors(proposal, baseline)
            for selector, sample in enumerate((baseline, proposal)):
                populations[selector].append(sample["seconds"])
                work[selector].append((sample["iterations"], sample["fock_builds"]))
        medians = {
            selector: statistics.median(values)
            for selector, values in populations.items()
        }
        deviations = {
            selector: statistics.median(
                abs(value - medians[selector]) for value in values
            )
            for selector, values in populations.items()
        }
        gain = 1 - medians[1] / medians[0]
        floor = max(0.02, 2 * sum(deviations[key] / medians[key] for key in (0, 1)))
        result[phase] = {
            "seconds": populations,
            "medians": medians,
            "speedup": medians[0] / medians[1],
            "improvement_fraction": gain,
            "variability_floor": floor,
            "profitable": gain > floor,
            "regresses": -gain > floor,
            "iterations_fock_builds": work,
        }
    return result


def main() -> None:
    """Keep gates effective under Python -O and reject corrupted evidence."""
    capsule = load(ROOT / "qualification.json.xz")
    for name, expected in capsule["reference_sha256"].items():
        if hashlib.sha256((REFERENCES / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"reference bytes changed: {name}")
    summaries = {
        name: scaling(cohort) for name, cohort in capsule["scaling_cohorts"].items()
    }
    for corruption in ("energy", "force", "source", "missing"):
        cohort = copy.deepcopy(capsule["scaling_cohorts"]["final"])
        first = cohort["records"]["r0-0"]["samples"][0]
        if corruption == "energy":
            first["energy"] += 1
        elif corruption == "force":
            first["forces"][0][0] = float("nan")
        elif corruption == "source":
            cohort["source_commit"] = "unknown"
        else:
            cohort["records"].pop("r2-1")
        try:
            scaling(cohort)
        except (ValueError, KeyError):
            continue
        raise ValueError(f"corrupted scaling evidence accepted: {corruption}")
    summaries["small-final"] = replay(load(ROOT / "endpoints-final.json.xz"))
    summaries["rejected-history-reset"] = replay(
        load(ROOT / "rejected-reset-history.json.xz")
    )
    summaries["two-panel-small"] = replay(
        load(ROOT / "two-panel-small-endpoints.json.xz")
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

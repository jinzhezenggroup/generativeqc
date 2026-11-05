"""Replay source-bound default admission, independent E+F and complete costs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

from benchmarks.occupied_xc_proposals import check_vectors, load, replay
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT.parent / "readme-final-20261022"


def scaling(capsule: dict) -> dict:
    """Keep absent controls distinct and reuse the original dense-oracle gates."""
    records = capsule["scaling_records"]
    selectors = ("0", "unset")
    expected = {
        f"r{repeat}-{selector}" for repeat in range(3) for selector in selectors
    }
    if set(records) != expected:
        raise ValueError("incomplete scaling population")
    oracles = load(REFERENCES / "oracles.json.xz")
    reuse = load(REFERENCES / "reference-reuse.json")
    for name, record in records.items():
        control = "0" if name.endswith("-0") else None
        if (
            "occupied_xc_env" not in record
            or record["occupied_xc_env"] != control
            or record["status"] != "complete"
            or record["source_revision"] != capsule["source_commit"]
            or record["native_build"]["library_sha256"] != capsule["library_sha256"]
            or record["native_build"]["source_identity"] != capsule["source_identity"]
            or (record["method"], record["atoms"], record["mode"], record["arm"])
            != ("r2scan-3c", 96, "direct", "standard")
            or [sample["phase"] for sample in record["samples"]] != ["cold", "warm"]
        ):
            raise ValueError(f"invalid default control, source or endpoint: {name}")
        qualified = qualify([record], oracles, reference_reuse=reuse)[0]
        if qualified["status"] != "qualified":
            raise ValueError(f"independent scaling gate failed: {name}")
        for sample in record["samples"]:
            if (
                not math.isfinite(sample["seconds"])
                or sample["seconds"] <= 0
                or sample["iterations"] < 1
                or sample["fock_builds"] < sample["iterations"]
            ):
                raise ValueError("invalid endpoint duration or work counts")
    result = {}
    for index, phase in enumerate(("cold", "warm")):
        populations = {selector: [] for selector in selectors}
        work = {selector: [] for selector in selectors}
        for repeat in range(3):
            baseline, proposal = (
                records[f"r{repeat}-{selector}"]["samples"][index]
                for selector in selectors
            )
            check_vectors(proposal, baseline)
            if baseline["settings"] != proposal["settings"]:
                raise ValueError("paired grid or solver settings changed")
            for selector, sample in zip(selectors, (baseline, proposal), strict=True):
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
        gain = 1 - medians["unset"] / medians["0"]
        floor = max(
            0.02,
            2 * sum(deviations[selector] / medians[selector] for selector in selectors),
        )
        result[phase] = {
            "seconds": populations,
            "medians": medians,
            "speedup": medians["0"] / medians["unset"],
            "improvement_fraction": gain,
            "variability_floor": floor,
            "profitable": gain > floor,
            "regresses": -gain > floor,
            "iterations_fock_builds": work,
        }
    return result


def main() -> None:
    """Recheck physical gates and corrupt controls, also under optimized Python."""
    capsule = load(ROOT / "qualification.json.xz")
    if capsule["schema"] != "generativeqc.occupied-xc-default-proposals.v1":
        raise ValueError("unsupported default qualification")
    for name, expected in capsule["reference_sha256"].items():
        if hashlib.sha256((REFERENCES / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"reference bytes changed: {name}")
    small = load(ROOT / "endpoints.json.xz")
    if (
        small["source_commit"] != capsule["source_commit"]
        or small["library_sha256"] != capsule["library_sha256"]
        or small.get("proposal_selector") != "unset"
    ):
        raise ValueError("small endpoint control or image mismatch")
    summary = {"small": replay(small), "scaling": scaling(capsule)}
    if (
        not summary["scaling"]["cold"]["profitable"]
        or summary["scaling"]["warm"]["regresses"]
    ):
        raise ValueError("default promotion fails the predeclared endpoint cost floor")
    for corruption in ("energy", "force", "source", "control", "missing"):
        changed = copy.deepcopy(capsule)
        record = changed["scaling_records"]["r0-unset"]
        if corruption == "energy":
            record["samples"][0]["energy"] += 1
        elif corruption == "force":
            record["samples"][0]["forces"][0][0] = float("nan")
        elif corruption == "source":
            changed["source_commit"] = "unknown"
        elif corruption == "control":
            record["occupied_xc_env"] = "1"
        else:
            changed["scaling_records"].pop("r2-unset")
        try:
            scaling(changed)
        except (ValueError, KeyError):
            continue
        raise ValueError(f"corrupted default evidence accepted: {corruption}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

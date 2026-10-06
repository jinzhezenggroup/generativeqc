"""Validate retained paired endpoints; report raw times and actual work only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

EXPECTED_BINARIES = {
    "control": "daf5d5b0f3bdca15bdf0e56243f0163455767333c2fa8ff6b9285f4a561ce2af",
    "candidate": "cca2d24221b6c040a32907b70cb77590526a6dd891da1ba14352e9b8b4472098",
}
PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def load_native(root: Path, method: str, atoms: int, arm: str) -> tuple[dict, str]:
    """Reject partial journals, swapped binary labels and unmatched protocols."""
    path = root / f"{method}-{atoms}-{arm}.json"
    if method == "hf":
        path = root / f"{method}-{atoms}-{arm}" / "results.json"
    payload = json.loads(path.read_text())
    identity = payload["identity"] if method == "hf" else payload
    if method != "hf" and payload["status"] != "measured":
        raise ValueError(f"incomplete native journal: {path}")
    binary = identity["native_build"]["library_sha256"]
    if binary != EXPECTED_BINARIES[arm]:
        raise ValueError(f"unexpected {arm} binary: {binary}")
    rows = payload["records"]
    expected = sorted(
        (phase, repeat)
        for phase, count in PHASE_COUNTS.items()
        for repeat in range(count)
    )
    actual = sorted((row["phase"], row["repeat"]) for row in rows)
    if actual != expected:
        raise ValueError(f"incomplete or duplicate endpoint inventory: {path}")
    for row in rows:
        if not row["gate"] or not row["converged"] or row["status"] != 0:
            raise ValueError(f"failed independent numerical gate: {path}")
        if len(row["forces"]) != atoms or any(
            len(vector) != 3 for vector in row["forces"]
        ):
            raise ValueError(f"unexpected force shape: {path}")
        values = [row["energy"], row["seconds"], row["complete_seconds"]]
        values.extend(value for vector in row["forces"] for value in vector)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"nonfinite endpoint: {path}")
        if row["seconds"] <= 0 or row["iterations"] < 1:
            raise ValueError(f"invalid endpoint time or iteration count: {path}")
        if row["energy_error"] > 1e-8 or row["force_error"] > 1e-7:
            raise ValueError(f"independent endpoint tolerance exceeded: {path}")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: Path) -> dict:
    """Preserve branches and missing telemetry, without iteration-normalized times."""
    if (root / "job.exit").read_text().strip() != "0":
        raise ValueError("campaign did not terminate successfully")
    summary = {
        "scope": "paired independent clean HF/PBE0 endpoints, no profiler timing",
        "iteration_normalization": False,
        "hf_native_iteration_history": "not exported by the retained native API",
        "cells": [],
    }
    for method in ("hf", "pbe0"):
        for atoms in (48, 96):
            arms = {}
            for arm in EXPECTED_BINARIES:
                payload, digest = load_native(root, method, atoms, arm)
                identity = payload["identity"] if method == "hf" else payload
                arms[arm] = {
                    "payload": payload,
                    "protocol": identity["protocol"],
                    "reference_sha256": identity["reference_sha256"],
                    "endpoint_sha256": digest,
                }
            if arms["control"]["protocol"] != arms["candidate"]["protocol"]:
                raise ValueError("paired native scientific protocols differ")
            if (
                arms["control"]["reference_sha256"]
                != arms["candidate"]["reference_sha256"]
            ):
                raise ValueError(
                    "paired native arms did not use one independent oracle"
                )
            cell = {"method": method, "atoms": atoms, "phases": {}}
            for phase in PHASE_COUNTS:
                entry = {}
                for arm, content in arms.items():
                    rows = sorted(
                        (
                            row
                            for row in content["payload"]["records"]
                            if row["phase"] == phase
                        ),
                        key=lambda row: row["repeat"],
                    )
                    entry[arm] = {
                        "complete_seconds": [row["complete_seconds"] for row in rows],
                        "median_complete_seconds": statistics.median(
                            row["complete_seconds"] for row in rows
                        ),
                        "iterations": [row["iterations"] for row in rows],
                        "fock_builds": [row["fock_builds"] for row in rows],
                        "maximum_energy_error": max(
                            row["energy_error"] for row in rows
                        ),
                        "maximum_force_error": max(row["force_error"] for row in rows),
                    }
                entry["control_over_candidate_raw_speedup"] = (
                    entry["control"]["median_complete_seconds"]
                    / entry["candidate"]["median_complete_seconds"]
                )
                entry["matching_actual_iterations"] = (
                    entry["control"]["iterations"] == entry["candidate"]["iterations"]
                )
                entry["matching_actual_fock_builds"] = (
                    None
                    if any(
                        value is None
                        for arm in EXPECTED_BINARIES
                        for value in entry[arm]["fock_builds"]
                    )
                    else entry["control"]["fock_builds"]
                    == entry["candidate"]["fock_builds"]
                )
                cell["phases"][phase] = entry
            cell["receipts"] = {
                arm: {key: value for key, value in content.items() if key != "payload"}
                for arm, content in arms.items()
            }
            summary["cells"].append(cell)
    return summary


def main() -> None:
    """Write a verified summary separately from the immutable raw evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(
        json.dumps(summarize(arguments.campaign), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

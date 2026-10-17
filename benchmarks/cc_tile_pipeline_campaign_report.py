"""Reconcile two clean CC endpoints and one separate diagnostic run.

This report binds every retained run artifact, exact semantic work and endpoint
energy bits. Diagnostic wall is never mixed into the clean timing pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any

try:
    from benchmarks._retention import raw_output_path
    from benchmarks.cc_tile_pipeline_campaign import WORK, check_identity, check_science
except ModuleNotFoundError:
    from _retention import raw_output_path
    from cc_tile_pipeline_campaign import WORK, check_identity, check_science

SCHEMA = "generativeqc.cc-tile-campaign.v1"
REPORT_SCHEMA = "generativeqc.cc-tile-campaign-report.v1"


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_run(directory: Path) -> dict[str, Any]:
    """Rehash an immutable run directory before trusting its receipt."""
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported campaign receipt schema")
    identity = receipt.get("identity")
    check_identity(identity)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("campaign artifact receipt is missing")
    if not {"endpoint.json", "oracle.json", "native.log"} <= artifacts.keys():
        raise ValueError(
            "endpoint, independent oracle and native log artifacts required"
        )
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != {*artifacts, "receipt.json"}:
        raise ValueError("campaign directory has missing or unreceipted files")
    for name, expected in artifacts.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"campaign artifact must be a regular file: {name}")
        if path.stat().st_size != expected.get("size") or checksum(
            path
        ) != expected.get("sha256"):
            raise ValueError(f"campaign artifact mismatch: {name}")
    if (
        receipt.get("outcome") != "PASS"
        or receipt.get("binary_identity_unchanged") is not True
        or receipt.get("input_identity_unchanged") is not True
        or type(receipt.get("exit_code")) is not int
        or receipt["exit_code"] != 0
    ):
        raise ValueError("campaign endpoint did not pass its bound gates")
    if checksum(directory / "oracle.json") != identity["oracle_sha256"]:
        raise ValueError("independent oracle differs from frozen campaign identity")
    endpoint = json.loads((directory / "endpoint.json").read_text())
    oracle = json.loads((directory / "oracle.json").read_text())
    recomputed_science = check_science(endpoint, oracle)
    if recomputed_science["status"] != "PASS":
        raise ValueError("recomputed independent scientific gates failed")
    science = receipt.get("science", {})
    gates = science.get("gates", {})
    if (
        science.get("status") != "PASS"
        or gates.keys() != recomputed_science["gates"].keys()
    ):
        raise ValueError("independent scientific gates are missing or failed")
    for name, gate in gates.items():
        if gate.get("passed") is not True or any(
            type(gate.get(key)) not in (int, float)
            or not math.isfinite(gate[key])
            or gate[key] < 0
            for key in ("value", "limit")
        ):
            raise ValueError(f"invalid scientific gate: {name}")
    if science != recomputed_science:
        raise ValueError("recorded scientific gates differ from endpoint/oracle replay")
    wall = receipt.get("process_wall_ns")
    if type(wall) is not int or wall <= 0:
        raise ValueError("positive endpoint process wall is required")
    work = receipt.get("work")
    if (
        not isinstance(work, dict)
        or work.keys() != set(WORK)
        or any(type(value) is not int or value < 0 for value in work.values())
    ):
        raise ValueError("complete integer semantic work is required")
    if work != {key: endpoint[key] for key in WORK}:
        raise ValueError("semantic work differs from retained endpoint output")
    return {"directory": str(directory), "receipt": receipt, "endpoint": endpoint}


def compare_runs(directories: list[Path]) -> dict[str, Any]:
    """Require exactly two clean samples and one separately profiled sample."""
    if len(directories) != 3:
        raise ValueError("exactly two clean runs and one diagnostic run are required")
    if len({directory.resolve() for directory in directories}) != 3:
        raise ValueError("three distinct run directories are required")
    runs = [read_run(directory) for directory in directories]
    clean = [run for run in runs if run["receipt"].get("diagnostic") is False]
    diagnostic = [run for run in runs if run["receipt"].get("diagnostic") is True]
    if len(clean) != 2 or len(diagnostic) != 1:
        raise ValueError("require two clean receipts and one diagnostic receipt")
    identity = clean[0]["receipt"].get("identity")
    if not isinstance(identity, dict) or any(
        run["receipt"].get("identity") != identity for run in runs
    ):
        raise ValueError("campaign run identity mismatch")
    work = clean[0]["receipt"]["work"]
    if any(run["receipt"]["work"] != work for run in runs):
        raise ValueError("semantic work changed across clean/diagnostic runs")
    energy_keys = ("total_energy", "triples_energy")
    energy_bits = {
        key: [struct.pack("d", run["endpoint"][key]).hex() for run in runs]
        for key in energy_keys
    }
    if any(len(set(values)) != 1 for values in energy_bits.values()):
        raise ValueError("endpoint energy bits changed across the campaign")
    clean_wall = [run["receipt"]["process_wall_ns"] for run in clean]
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "identity": identity,
        "clean_process_wall_ns": clean_wall,
        "clean_process_wall_median_ns": statistics.median(clean_wall),
        "diagnostic_process_wall_ns": diagnostic[0]["receipt"]["process_wall_ns"],
        "diagnostic_wall_used_for_timing": False,
        "semantic_work": work,
        "energy_bits": {key: values[0] for key, values in energy_bits.items()},
        "science": [run["receipt"]["science"] for run in runs],
        "runs": [
            {
                "directory": run["directory"],
                "diagnostic": run["receipt"]["diagnostic"],
                "receipt_sha256": checksum(Path(run["directory"]) / "receipt.json"),
            }
            for run in runs
        ],
        "interpretation": (
            "paired clean endpoint observations plus one separately instrumented "
            "diagnostic; no speedup or pipeline promotion is inferred"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    report = compare_runs(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

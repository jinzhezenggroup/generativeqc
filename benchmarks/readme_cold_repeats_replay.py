"""Replay a portable numeric projection without the acquisition filesystem.

This is a declared projection, not a reconstruction of original process JSON.
Original archive/member identities stay in its provenance; all scientific sample
fields, full force vectors, timings and work counters are retained unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_cold_repeats import aggregate
from benchmarks.readme_cold_repeats_report import (
    accuracy,
    diagnostics,
    render,
    sha,
    verify_build,
)
from benchmarks.readme_defaults import save

SCHEMA = "generativeqc.readme-cold-repeats-portable.v1"


def identity(value: object) -> str:
    """Preserve JSON numeric values exactly; key order is not an identity."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def numerical_identity(checks: list[dict]) -> str:
    """Compare all recomputed gates apart from the declared receipt projection."""
    return identity(
        [{k: v for k, v in row.items() if k != "receipt_sha256"} for row in checks]
    )


def replay(payload: dict, oracles: list[dict]) -> dict:
    """Reject incomplete matrices and changed numerical results before plotting."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported portable cold-repeat projection")
    records = payload["records"]
    execution = payload.get("execution_binding")
    if (
        execution is not None
        and execution["build_receipt_sha256"]
        != payload["originals"]["build_receipt_sha256"]
    ):
        raise ValueError("projected execution names another original build")
    verify_build(records, payload["build_binding"], execution)
    rows = aggregate(records)
    diagnostic = diagnostics(rows)
    checks = accuracy(records, oracles, payload["reference_reuse"])
    actual = {
        "records": identity(records),
        "populations": identity(rows),
        "diagnostics": identity(diagnostic),
        "numerical": numerical_identity(checks),
    }
    if actual != payload["expected_replay_sha256"]:
        raise ValueError("portable replay differs from authenticated original analysis")
    return {
        "schema": SCHEMA,
        "label": payload["label"],
        "projection": payload["projection"],
        "originals": payload["originals"],
        "populations": rows,
        "diagnostics": diagnostic,
        "accuracy": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    payload = json.loads(lzma.decompress(args.evidence.read_bytes()))
    if sha(args.oracles) != payload["originals"]["oracles_sha256"]:
        raise ValueError("independent oracle bytes differ from frozen reference")
    result = replay(payload, json.loads(lzma.decompress(args.oracles.read_bytes())))
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", result)
    render(result["populations"], result["label"], args.output / "results-hfdft")


if __name__ == "__main__":
    main()

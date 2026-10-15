"""Retain rejected same-binary observations without publishing qualified timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

from tools.render_readme_dft import collect_point


def digest(path: Path) -> str:
    """Bind retained raw documents and generated archives to their actual bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Recheck numerical gates and keep the stale native identity disqualifying."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.campaign / "source-freeze.json").read_text())
    raw = args.campaign / "remote"
    points = []
    for arm in ("control", "candidate"):
        directory = raw / "validation-measurements" / arm / "none" / "r2scan-3c"
        for atoms in (12, 48):
            point = collect_point(directory, atoms)
            document = json.loads((directory / str(atoms) / "native.json").read_text())
            native = point["engines"]["native"]
            observed = document["native_build"]["probe"]["source_identity"]
            expected = frozen["native_source_identity"]
            consumer_match = (
                document["source_file_sha256"] == frozen["consumer_source_file_sha256"]
            )
            point.update(
                arm=arm,
                publication_status="diagnostic_only_not_promoted",
                source_binding={
                    "status": "unqualified",
                    "native_identity_matches": observed == expected,
                    "consumer_hashes_match": consumer_match,
                    "advertised_native_identity": observed,
                    "frozen_native_identity": expected,
                    "library_sha256": document["native_build"]["library_sha256"],
                },
            )
            if native["status"] != "measured" or len(native["records"]) != 12:
                raise ValueError(f"incomplete diagnostic population: {arm}/{atoms}")
            if observed == expected or not consumer_match:
                raise ValueError(f"unexpected source binding: {arm}/{atoms}")
            points.append(point)
    attempts = []
    for population in ("pilot-measurements", "triage-measurements"):
        for path in sorted((raw / population).rglob("native.outcome")):
            attempts.append(
                {
                    "path": str(path.relative_to(raw)),
                    "outcome": json.loads(path.read_text()),
                    "documents": {
                        name: {
                            "sha256": digest(path.with_name(name)),
                            "content": path.with_name(name).read_text(),
                        }
                        for name in ("native.json", "native.log", "reference.json")
                        if path.with_name(name).is_file()
                    },
                }
            )
    payload = {
        "schema": "generativeqc.rejected-order-six.v1",
        "publication_status": "diagnostic_only_not_promoted",
        "frozen_source": frozen,
        "points": points,
        "failed_attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        lzma.compress((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())
    )


if __name__ == "__main__":
    main()

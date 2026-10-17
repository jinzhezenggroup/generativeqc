"""Check retained observations, not missing force arrays or promotion gates.

The original packet is already public in immutable Git history. It was removed
from the live results tree after concurrent evidence exhausted its aggregate
budget. Read that existing object without network access or writing a second
archive; shallow clones can instead supply an explicit local packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import statistics
import subprocess
from pathlib import Path
from typing import Any

from tools.generativeqc_validation.xc_diagonal_evidence import REFERENCE

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_COMMIT = "f40055ecdb7763175f59fe8108f87805c3f3f193"
PACKET_MEMBER = "benchmarks/results/direct-relative-force-20261017/measurements.json.xz"
PACKET_SHA256 = "b8ec7d7031d50f9fa0a0cafbc3b39ab9931133439d2303bb377628f9fe21b7a9"


def load_packet(path: Path | None = None) -> dict[str, Any]:
    """Read an explicit packet or authenticate the previously published object.

    Missing historical objects are an actionable error, never permission to
    download evidence, replace observations, or silently skip validation.
    """
    if path is not None:
        payload = path.read_bytes()
    else:
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{EVIDENCE_COMMIT}:{PACKET_MEMBER}"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"Retained evidence object {EVIDENCE_COMMIT} is unavailable; "
                "fetch that existing commit or use --packet /path/to/measurements.json.xz"
            ) from error
        payload = result.stdout
        if hashlib.sha256(payload).hexdigest() != PACKET_SHA256:
            raise ValueError("Retained evidence object failed its integrity check")
    return json.loads(lzma.decompress(payload))


def summarize(packet: dict[str, Any]) -> None:
    """Check protocol identities, retained gates, samples and exclusive ledgers."""
    identities = {row["native_identity"] for row in packet["bindings"].values()}
    for expected, binding in packet["protocols"].items():
        protocol = REFERENCE["load"](binding["pinned_oracle_member"])["protocol"]
        actual = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert actual == expected
    for name, population in packet["populations"].items():
        assert population["source_identity"] in identities
        assert population["maximum_energy_error"] <= 1e-8
        assert population["maximum_force_error"] <= 1e-7
        assert population["samples_per_phase"] == {
            "cold": 1,
            "warm": 5,
            "moved": 1,
            "moved-warm": 5,
        }
        if "rows" in population:
            assert len(population["rows"]) == 12
            for phase, expected in population["complete_seconds"].items():
                rows = [
                    row
                    for label, row in zip(
                        packet["phases"], population["rows"], strict=True
                    )
                    if label == phase
                ]
                assert statistics.median(row[0] for row in rows) == expected
                assert all(row[0] > 0 and row[1] > 0 and row[2] > 0 for row in rows)
        print(name, population["complete_seconds"])
    for profile in packet["profiles"].values():
        row = profile["warm"]
        if "exclusive_host_seconds" in row:
            assert (
                abs(
                    sum(row["exclusive_host_seconds"].values())
                    - sum(row["exclusive_device_window_seconds"].values())
                )
                < 1e-7
            )
            assert (
                abs(
                    sum(row["complete_wall_ledger"].values())
                    - row["record"]["complete_seconds"]
                )
                < 1e-7
            )
    print(
        "Retained observations and protocol identities pass; no offline force replay."
    )


def main(argv: list[str] | None = None) -> None:
    """Validate historical observations or an explicitly selected local packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path)
    arguments = parser.parse_args(argv)
    summarize(load_packet(arguments.packet))


if __name__ == "__main__":
    main()

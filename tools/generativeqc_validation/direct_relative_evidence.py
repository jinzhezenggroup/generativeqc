"""Check compact observations, not missing force arrays or promotion gates."""

import hashlib
import json
import lzma
import statistics
from pathlib import Path

from tools.generativeqc_validation.xc_diagonal_evidence import REFERENCE

root = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/direct-relative-force-20261017"
)
packet = json.loads(lzma.decompress((root / "measurements.json.xz").read_bytes()))
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
                for label, row in zip(packet["phases"], population["rows"], strict=True)
                if label == phase
            ]
            assert statistics.median(row[0] for row in rows) == expected
            assert all(row[0] > 0 and row[1] > 0 and row[2] > 0 for row in rows)
    print(name, population["complete_seconds"])
for profile in packet["profiles"].values():
    for phase in ("warm",):
        row = profile[phase]
        if "exclusive_host_seconds" not in row:
            continue
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
print("Retained observations and protocol identities pass; no offline force replay.")

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
import math
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


PHASES = ["cold", *(["warm"] * 5), "moved", *(["moved-warm"] * 5)]
SAMPLES = {phase: PHASES.count(phase) for phase in PHASES}


def _number(value: Any, *, positive: bool = False) -> float:
    assert type(value) in (float, int) and math.isfinite(value)
    assert value > 0 if positive else value >= 0
    return value


def _digest(value: Any, length: int = 64) -> None:
    assert isinstance(value, str) and len(value) == length
    assert all(character in "0123456789abcdef" for character in value)


def _errors(row: dict[str, Any], prefix: str = "maximum_") -> None:
    assert _number(row[f"{prefix}energy_error"]) <= 1e-8
    assert _number(row[f"{prefix}force_error"]) <= 1e-7


def _population(row: dict[str, Any]) -> None:
    _errors(row)
    assert set(row["complete_seconds"]) == set(SAMPLES)
    for value in row["complete_seconds"].values():
        _number(value, positive=True)
    if "rows" in row:
        assert len(row["rows"]) == len(PHASES)
        for values in row["rows"]:
            assert len(values) == 3
            _number(values[0], positive=True)
            assert all(type(value) is int and value > 0 for value in values[1:])
        for phase, expected in row["complete_seconds"].items():
            samples = [
                values[0]
                for label, values in zip(PHASES, row["rows"], strict=True)
                if label == phase
            ]
            assert statistics.median(samples) == expected


def summarize(packet: dict[str, Any]) -> None:
    """Validate available links/scalars; raw source receipts and arrays are absent."""
    assert packet["schema"] == "direct-relative-scalar-receipts-v1"
    assert packet["columns"] == ["complete_seconds", "iterations", "fock_builds"]
    assert packet["phases"] == PHASES
    assert packet["bindings"] and packet["protocols"] and packet["populations"]
    identities = set()
    for binding in packet["bindings"].values():
        _digest(binding["native_identity"])
        _digest(binding["revision"], 40)
        assert binding["all_production_inputs_byte_equal"] is True
        assert type(binding["production_input_count"]) is int
        assert binding["production_input_count"] > 0
        identities.add(binding["native_identity"])
    protocols = {}
    for expected, binding in packet["protocols"].items():
        member = binding["pinned_oracle_member"]
        protocol = REFERENCE["load"](member)["protocol"]
        actual = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert actual == expected
        method = protocol["method"].split("/")[0].lower()
        case = f"{'hf' if method == 'rhf' else method}-{protocol['atoms']}"
        assert member == f"raw/reference-{case}.json.xz"
        assert member not in protocols
        protocols[member] = expected
    for name, population in packet["populations"].items():
        case = name.rsplit("/", 1)[1]
        assert (
            population["protocol_sha256"] == protocols[f"raw/reference-{case}.json.xz"]
        )
        assert population["source_identity"] in identities
        assert population["samples_per_phase"] == SAMPLES
        _population(population)
        print(name, population["complete_seconds"])
    for peer in packet["peers"].values():
        assert peer["columns"] == ["complete_seconds", "iterations", "scf_jk_builds"]
        assert peer["protocol_sha256"] == protocols[peer["independent_oracle"]]
        _digest(peer["protocol_source_files_sha256"])
        assert "rows" in peer
        _population(peer)
    for profile in packet["profiles"].values():
        row = profile["warm"]
        record = row["record"]
        assert record["phase"] == "warm"
        assert record["gate"] is True and record["converged"] is True
        _errors(record, "")
        for name in ("iterations", "fock_builds"):
            assert type(record[name]) is int and record[name] > 0
        _number(record["complete_seconds"], positive=True)
        _number(record["seconds"], positive=True)
        _number(record["prepare_seconds"])
        if "exclusive_host_seconds" not in row:
            continue
        for name in (
            "exclusive_host_seconds",
            "exclusive_device_window_seconds",
            "complete_wall_ledger",
        ):
            for value in row[name].values():
                _number(value)
        assert (
            abs(
                sum(row["exclusive_host_seconds"].values())
                - sum(row["exclusive_device_window_seconds"].values())
            )
            < 1e-7
        )
        assert (
            abs(sum(row["complete_wall_ledger"].values()) - record["complete_seconds"])
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

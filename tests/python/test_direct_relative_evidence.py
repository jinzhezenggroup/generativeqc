"""Keep evidence recovery explicit, read-only and independent of GPU execution."""

import copy
import hashlib
import json
import lzma
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.generativeqc_validation import direct_relative_evidence as evidence


def test_explicit_packet_does_not_require_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported source tree can validate a separately retained local packet."""
    packet = {"fixture": [1, 2, 3]}
    path = tmp_path / "observations.json.xz"
    path.write_bytes(lzma.compress(json.dumps(packet).encode()))

    def unexpected_git(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Explicit evidence must not access Git or the network")

    monkeypatch.setattr(evidence.subprocess, "run", unexpected_git)
    assert evidence.load_packet(path) == packet


@pytest.mark.parametrize("corrupt", [False, True])
def test_historical_packet_integrity(
    monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    """Authenticate retained scientific inputs before decoding observations."""
    packet = {"fixture": "immutable scientific observations"}
    payload = lzma.compress(json.dumps(packet).encode())
    monkeypatch.setattr(evidence, "PACKET_SHA256", hashlib.sha256(payload).hexdigest())
    if corrupt:
        payload += b"unexpected bytes"
    monkeypatch.setattr(
        evidence.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=payload),
    )
    if corrupt:
        with pytest.raises(ValueError, match="integrity"):
            evidence.load_packet()
    else:
        assert evidence.load_packet() == packet


def test_missing_history_requires_explicit_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shallow checkouts must not silently omit evidence or start downloads."""

    def missing_object(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(evidence.subprocess, "run", missing_object)
    with pytest.raises(RuntimeError, match="--packet"):
        evidence.load_packet()


PRIMARY = "primary-node5/candidate/r2scan-3c-96"


@pytest.fixture
def packet() -> dict[str, Any]:
    return evidence.load_packet()


def _run(packet: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    evidence.summarize(packet)


def test_retained_packet_passes(
    packet: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(packet, monkeypatch)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_case",
        "unknown_protocol",
        "wrong_source",
        "schema",
        "columns",
        "phase",
        "peer_force",
        "peer_nan",
        "peer_protocol",
        "peer_median",
        "peer_count",
        "clean_force",
        "counted_energy",
        "clean_gate",
        "counted_convergence",
        "negative_error",
        "infinite_row",
        "negative_summary",
        "native_force",
        "native_median",
    ],
)
def test_available_corruption_is_rejected(
    packet: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    packet = copy.deepcopy(packet)
    native = packet["populations"][PRIMARY]
    peer = packet["peers"]["reference-node5"]
    clean = packet["profiles"]["relative-six-clean/candidate"]["warm"]["record"]
    counted = packet["profiles"]["relative-six/candidate"]["warm"]["record"]
    if mutation == "wrong_case":
        native["protocol_sha256"] = packet["populations"][
            "primary-node5/candidate/hf-48"
        ]["protocol_sha256"]
    elif mutation == "unknown_protocol":
        native["protocol_sha256"] = "0" * 64
    elif mutation == "wrong_source":
        native["source_identity"] = "0" * 64
    elif mutation in ("schema", "columns"):
        packet[mutation] = "unknown"
    elif mutation == "phase":
        packet["phases"][0] = "warm"
    elif mutation == "peer_force":
        peer["maximum_force_error"] = 1.0
    elif mutation == "peer_nan":
        peer["maximum_energy_error"] = float("nan")
    elif mutation == "peer_protocol":
        peer["protocol_sha256"] = "0" * 64
    elif mutation == "peer_median":
        peer["complete_seconds"]["cold"] += 1
    elif mutation == "peer_count":
        peer["rows"][0][1] = True
    elif mutation == "clean_force":
        clean["force_error"] = 1.0
    elif mutation == "counted_energy":
        counted["energy_error"] = 1.0
    elif mutation == "clean_gate":
        clean["gate"] = False
    elif mutation == "counted_convergence":
        counted["converged"] = False
    elif mutation == "negative_error":
        native["maximum_force_error"] = -1.0
    elif mutation == "infinite_row":
        native["rows"][5][0] = float("inf")
    elif mutation == "negative_summary":
        packet["populations"]["secondary-first/candidate/r2scan-3c-96"][
            "complete_seconds"
        ]["warm"] = -1
    elif mutation == "native_force":
        native["maximum_force_error"] = 1.0
    elif mutation == "native_median":
        native["complete_seconds"]["warm"] += 1
    with pytest.raises((AssertionError, KeyError, ValueError)):
        _run(packet, monkeypatch)

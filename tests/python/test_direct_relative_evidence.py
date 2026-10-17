"""Keep evidence recovery explicit, read-only and independent of GPU execution."""

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

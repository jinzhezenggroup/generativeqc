"""Keep the retained negative CPU campaign offline and non-promotable."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.generativeqc_validation.publication import validate_publication

PACKET = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/required-native-policy-20261104"
)


def run_replay(packet: Path) -> subprocess.CompletedProcess[str]:
    """A deliberately unavailable library must not affect receipt-only replay."""
    return subprocess.run(
        [sys.executable, "-I", str(packet / "replay.py")],
        env={**os.environ, "GENERATIVEQC_LIBRARY": str(packet / "unavailable.so")},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_incomplete_campaign_replays_without_native_execution() -> None:
    replay = run_replay(PACKET)
    assert replay.returncode == 2, replay.stderr
    result = json.loads(replay.stdout)
    assert result["current_protocol"]["complete"] is False
    assert result["current_protocol"]["qualified"] is False
    assert len(result["current_protocol"]["missing_blocks"]) == 8
    assert result["current_protocol"]["partial_journals"] == []
    assert result["previous_protocol"]["qualified"] is False


def test_publication_envelope_does_not_promote_partial_results() -> None:
    manifest = json.loads((PACKET / "publication.json").read_text())
    files = {
        entry["path"]: (PACKET / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    assert manifest["decision"]["status"] == "inconclusive"
    evidence = json.loads(files["evidence.json"])
    assert evidence["stages"]["endpoint"]["status"] == "fail"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["settings"]["real_gpu_execution"] is False


@pytest.mark.parametrize(
    "member",
    ["samples.json.gz", "summary.json", "audit_receipts.py.txt", "replay.py"],
)
def test_changed_packet_member_fails_before_replay(tmp_path: Path, member: str) -> None:
    packet = tmp_path / "packet"
    shutil.copytree(PACKET, packet)
    changed = packet / member
    changed.write_bytes(changed.read_bytes() + b"\n")
    replay = run_replay(packet)
    assert replay.returncode == 1
    assert "publication member changed" in replay.stderr
    assert replay.stdout == ""

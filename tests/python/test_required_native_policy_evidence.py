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
from tools.generativeqc_validation.record import load_publication_record

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
    evidence = load_publication_record(PACKET)
    assert evidence["stages"]["endpoint"]["status"] == "fail"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["settings"]["real_gpu_execution"] is False


@pytest.mark.parametrize(
    "member",
    ["samples.json.xz", "summary.json.xz", "frozen-sources.json.xz", "replay.py"],
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


def test_frozen_source_extraction_preserves_bytes_and_never_overwrites(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "sources"
    command = [
        sys.executable,
        "-I",
        str(PACKET / "replay.py"),
        "--extract-sources",
        str(destination),
    ]
    extracted = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=60
    )
    assert extracted.returncode == 0, extracted.stderr
    assert (destination / "audit_receipts.py.txt").is_file()
    assert (destination / "frozen-v2/compare.py.txt").is_file()
    assert (destination / "diagnostic/packaged_cpu_probe.cpp.txt").is_file()
    previous = (destination / "audit_receipts.py.txt").read_bytes()
    repeated = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=60
    )
    assert repeated.returncode == 1
    assert "FileExistsError" in repeated.stderr
    assert (destination / "audit_receipts.py.txt").read_bytes() == previous

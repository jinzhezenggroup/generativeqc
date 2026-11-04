"""Numerical receipt replay must fail closed with or without Python optimization."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = Path("benchmarks/results/packaged-correlated-cpu-20261104")
REFERENCE = Path("tests/reference_data/cbs/h2-dz-tz.json")


def run_replay(root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    """Only replay JSON observations; a native library is deliberately unavailable."""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "python"), str(ROOT))),
        "GENERATIVEQC_LIBRARY": str(root / "unavailable.so"),
    }
    environment.pop("PYTHONOPTIMIZE", None)
    if mode == "environment":
        environment["PYTHONOPTIMIZE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            *(["-O"] if mode == "flag" else []),
            str(root / PACKET / "replay.py"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.parametrize("mode", ["normal", "flag", "environment"])
def test_numerical_receipts_replay_in_all_python_modes(mode: str) -> None:
    replay = run_replay(ROOT, mode)
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout) == {
        "numerical_blocks_passed": 12,
        "performance_qualified": False,
    }


@pytest.mark.parametrize("mode", ["normal", "flag", "environment"])
@pytest.mark.parametrize("mutation", ["reference", "numerical", "source", "promotion"])
def test_invalid_receipts_fail_closed(tmp_path: Path, mode: str, mutation: str) -> None:
    """Exercise semantic gates as well as the independent reference hash binding."""
    packet = tmp_path / PACKET
    shutil.copytree(ROOT / PACKET, packet)
    reference = tmp_path / REFERENCE
    reference.parent.mkdir(parents=True)
    shutil.copy2(ROOT / REFERENCE, reference)
    if mutation == "reference":
        value = json.loads(reference.read_text())
        value["reference"]["energy"] += 1.0
        reference.write_text(json.dumps(value))
    else:
        samples_path = packet / "samples.json.xz"
        value = json.loads(lzma.decompress(samples_path.read_bytes()))
        if mutation == "numerical":
            value["journal"][1]["energy"] += 1.0
        elif mutation == "source":
            value["provenance"]["native_library_sha256"] = "0" * 64
        else:
            value["journal"][1]["qualification"] = True
        samples_path.write_bytes(lzma.compress(json.dumps(value).encode()))
        # Rebind only storage checks in this isolated fixture, so semantic
        # failures cannot be masked by an earlier container-hash rejection.
        evidence_path = packet / "evidence.json.xz"
        evidence = json.loads(lzma.decompress(evidence_path.read_bytes()))
        for attachment in evidence["attachments"]:
            if attachment["path"] == samples_path.name:
                attachment["sha256"] = hashlib.sha256(
                    samples_path.read_bytes()
                ).hexdigest()
        evidence_path.write_bytes(lzma.compress(json.dumps(evidence).encode()))
        manifest_path = packet / "publication.json"
        manifest = json.loads(manifest_path.read_text())
        for member in manifest["files"]:
            data = (packet / member["path"]).read_bytes()
            member.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        manifest_path.write_text(json.dumps(manifest))
    replay = run_replay(tmp_path, mode)
    assert replay.returncode != 0
    message = (
        "ValueError: ('composite', 'energy'"
        if mutation == "numerical"
        else "ValueError: retained evidence gate failed"
    )
    assert message in replay.stderr
    assert replay.stdout == ""

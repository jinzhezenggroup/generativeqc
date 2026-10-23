"""Retained proposal-floor evidence keeps executable gates under Python -O."""

from __future__ import annotations

import json
import lzma
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/fast-source-floor-20261024"
)


@pytest.mark.parametrize("damage", ("population", "binary", "force", "time"))
def test_floor_replay_rejects_damaged_evidence(damage: str) -> None:
    """Check the replay's scientific and population boundaries, not source text."""
    replay = runpy.run_path(str(DIRECTORY / "reproduce.py"))["reproduce"]
    with lzma.open(DIRECTORY / "qualification.json.xz", "rt") as stream:
        payload = json.load(stream)
    records = payload["qualification"]
    name = "fast-pbe0-12-candidate-r1.json"
    if damage == "population":
        del records[name]
    elif damage == "binary":
        records[name]["native_build"]["library_sha256"] = "wrong-library"
    elif damage == "force":
        records[name]["samples"][0]["forces"][0][0] += 1.0
    else:
        records[name]["samples"][0]["seconds"] = float("nan")
    with pytest.raises(ValueError):
        replay(payload)


def test_floor_replay_optimized_numerical_gate() -> None:
    """Optimized Python must still reject an independently recomputed error."""
    script = """
import json, lzma, runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
replay = runpy.run_path(str(root / 'reproduce.py'))['reproduce']
with lzma.open(root / 'qualification.json.xz', 'rt') as stream:
    payload = json.load(stream)
replay(payload)
payload['qualification']['fast-pbe0-12-candidate-r1.json']['samples'][0]['energy'] += 1
replay(payload)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script, str(DIRECTORY)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr and "energy_per_atom" in result.stderr

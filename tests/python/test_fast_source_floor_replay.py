"""Retained proposal-floor evidence keeps executable gates under Python -O."""

from __future__ import annotations

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
@pytest.mark.parametrize("cohort", ("qualification.json.xz", "integration.json.xz"))
def test_floor_replay_rejects_damaged_evidence(damage: str, cohort: str) -> None:
    """Check the replay's scientific and population boundaries, not source text."""
    module = runpy.run_path(str(DIRECTORY / "reproduce.py"))
    replay = module["reproduce"]
    payload = module["load_receipt"](cohort)
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


@pytest.mark.parametrize("cohort", ("qualification.json.xz", "integration.json.xz"))
def test_floor_replay_optimized_numerical_gate(cohort: str) -> None:
    """Optimized Python must still reject an independently recomputed error."""
    script = """
import runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
module = runpy.run_path(str(root / 'reproduce.py'))
replay = module['reproduce']
payload = module['load_receipt'](sys.argv[2])
replay(payload)
payload['qualification']['fast-pbe0-12-candidate-r1.json']['samples'][0]['energy'] += 1
replay(payload)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script, str(DIRECTORY), cohort],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr and "energy_per_atom" in result.stderr

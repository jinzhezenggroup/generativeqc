"""Retained cold claims must reject corrupted observables, work and identities."""

from __future__ import annotations

import json
import lzma
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/combined-range-source-20261107"
)


@pytest.mark.parametrize(
    "corruption",
    [
        "valid",
        "force",
        "work",
        "identity",
        "selector",
        "vv10",
        "time",
        "missing_a100",
        "missing_5090",
        "empty_devices",
        "infinite_baseline",
        "basis",
        "charge",
        "spin",
    ],
)
@pytest.mark.parametrize("optimized", [False, True])
def test_combined_source_evidence_keeps_gates_under_optimized_python(
    tmp_path: Path, corruption: str, optimized: bool
) -> None:
    """A missing gate cannot be hidden by -O or an otherwise self-consistent archive."""
    evidence = json.loads(
        lzma.decompress((DIRECTORY / "evidence.json.xz").read_bytes())
    )
    candidate = evidence["cohorts"]["final"]["devices"]["A100"]["records"]["default"]
    sample = candidate["samples"][0]
    if corruption == "force":
        sample["forces"][0][0] += 1e-3
    elif corruption == "work":
        sample["ks_diagnostic"]["coulomb_exchange_work"]["exchange_applications"] = 32
    elif corruption == "identity":
        candidate["source_revision"] = "invalid"
    elif corruption == "selector":
        candidate["source_schedule_experiment"]["combined_range_source"] = "1"
    elif corruption == "vv10":
        sample["ks_diagnostic"]["nonlocal_work"]["pair_sweeps"] -= 1
    elif corruption == "time":
        sample["seconds"] *= 2
    elif corruption == "missing_a100":
        del evidence["cohorts"]["final"]["devices"]["A100"]
    elif corruption == "missing_5090":
        del evidence["cohorts"]["final"]["devices"]["RTX5090"]
    elif corruption == "empty_devices":
        evidence["cohorts"]["final"]["devices"].clear()
    elif corruption == "infinite_baseline":
        baseline = evidence["cohorts"]["final"]["devices"]["A100"]["records"][
            "baseline"
        ]
        baseline["samples"][0]["seconds"] = float("inf")
    elif corruption == "basis":
        candidate["orbital_basis"] = "def2-tzvp"
    elif corruption == "charge":
        candidate["charge"] = 1
    elif corruption == "spin":
        candidate["spin"] = 2
    (tmp_path / "evidence.json.xz").write_bytes(
        lzma.compress(json.dumps(evidence).encode())
    )
    command = (
        "import runpy; from pathlib import Path; "
        f"reader=runpy.run_path({str(DIRECTORY / 'replay.py')!r}); "
        f"reader['replay'](Path({str(tmp_path)!r}))"
    )
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if corruption == "valid":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "ValueError" in result.stderr

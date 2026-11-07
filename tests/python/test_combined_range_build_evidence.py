"""Composite-build claims retain numerical/work/default gates under Python -O."""

from __future__ import annotations

import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/combined-range-build-20261107"
)


@pytest.mark.parametrize(
    "evidence_name", ["evidence.json.xz", "pre-master-evidence.json.xz"]
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
        "source",
        "oracle",
        "master_source",
        "retained_receipt",
        "basis",
        "charge",
        "spin",
        "warm_protocol",
        "paired_settings",
        "paired_source_work",
        "missing_device",
        "infinite_baseline",
    ],
)
@pytest.mark.parametrize("optimized", [False, True])
def test_combined_build_evidence_keeps_optimized_python_gates(
    tmp_path: Path, corruption: str, evidence_name: str, optimized: bool
) -> None:
    """A self-consistent archive must still pass the independently specified gates."""
    evidence = json.loads(lzma.decompress((DIRECTORY / evidence_name).read_bytes()))
    population = evidence["devices"]["A100"]
    candidate = population["records"]["default"]
    sample = candidate["samples"][0]
    if corruption == "basis":
        candidate["orbital_basis"] = "def2-TZVP"
    elif corruption == "charge":
        candidate["charge"] = 1
    elif corruption == "spin":
        candidate["spin"] = 2
    elif corruption == "warm_protocol":
        candidate["protocol"]["warm"] = "cross-owner pooled density"
    elif corruption == "paired_settings":
        for record in population["records"].values():
            record["samples"][0]["settings"]["screening_tolerance"] = 1e-6
    elif corruption == "paired_source_work":
        for record in population["records"].values():
            for operator in record["samples"][0]["precision"]["operators"]:
                if operator["name"] == "coulomb_recurrence":
                    operator["count"] *= 2
    elif corruption == "missing_device":
        del evidence["devices"]["RTX5090"]
    elif corruption == "infinite_baseline":
        population["records"]["baseline"]["samples"][0]["seconds"] = float("inf")
    elif corruption == "force":
        sample["forces"][0][0] += 1e-3
    elif corruption == "work":
        sample["ks_diagnostic"]["coulomb_exchange_work"]["exchange_applications"] *= 2
    elif corruption == "identity":
        candidate["source_revision"] = "invalid"
    elif corruption == "selector":
        candidate["source_schedule_experiment"]["combined_range_build"] = "1"
    elif corruption == "vv10":
        sample["ks_diagnostic"]["nonlocal_work"]["pair_sweeps"] -= 1
    elif corruption == "time":
        sample["seconds"] *= 2
    elif corruption == "source":
        next(
            operator
            for operator in sample["precision"]["operators"]
            if operator["name"] == "coulomb_recurrence"
        )["count"] *= 2
    elif corruption == "oracle":
        population["oracle_log"] = "missing independent numerical qualification"
    elif corruption in {"master_source", "retained_receipt"}:
        previous = DIRECTORY / "pre-master-evidence.json.xz"
        evidence["master_sync"] = {
            "previous_evidence": previous.relative_to(DIRECTORY.parents[2]).as_posix(),
            "previous_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
            "integration_commit": evidence["source_receipt"]["base"],
        }
        key = (
            "integration_commit" if corruption == "master_source" else "previous_sha256"
        )
        evidence["master_sync"][key] = "invalid"
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

"""Retain the complete A100 matrix, including failures and original FP64 bytes."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/readme-a100-20261108"
)


@pytest.mark.parametrize("optimized", [False, True])
def test_all_readme_a100_endpoints_replay(optimized: bool) -> None:
    """Offline qualification must not depend on assert statements or a GPU."""
    command = [
        sys.executable,
        *(["-O"] if optimized else []),
        str(DIRECTORY / "replay.py"),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["execution_counts"] == {
        "populations": 128,
        "failed": 0,
        "complete_endpoints": 512,
    }
    assert report["totals"]["cold"]["paired"] == 64
    assert report["totals"]["warm"]["paired"] == 64


@pytest.mark.parametrize("corruption", ["checksum", "allocation", "scientific_gates"])
def test_reject_relabelled_readme_a100_receipts(corruption: str) -> None:
    """Recomputed identities alone cannot legitimize altered scientific inputs."""
    module = runpy.run_path(str(DIRECTORY / "replay.py"))
    evidence = module["load_evidence"]()
    files = evidence["files"]
    name = "raw/results-228202/direct-hf-3-standard.json"
    if corruption == "checksum":
        files[name] += "\n"
    else:
        record = json.loads(files[name])
        if corruption == "allocation":
            record["scheduler"]["LSB_JOBID"] = "wrong-allocation"
        else:
            record["protocol"]["common_observable_gates"][
                "energy_abs_per_atom_hartree"
            ] *= 2
        files[name] = json.dumps(record, indent=2) + "\n"
        digest = hashlib.sha256(files[name].encode()).hexdigest()
        evidence["sha256"][name] = digest
        executions_name = "raw/results-228202/executions.json"
        executions = json.loads(files[executions_name])
        execution = next(
            row
            for row in executions
            if row["case"]["method"] == "hf"
            and row["case"]["mode"] == "direct"
            and row["case"]["atoms"] == 3
            and row["arm"] == "standard"
        )
        execution["record_sha256"] = digest
        files[executions_name] = json.dumps(executions, indent=2) + "\n"
        evidence["sha256"][executions_name] = hashlib.sha256(
            files[executions_name].encode()
        ).hexdigest()
    with pytest.raises(ValueError):
        module["replay"](evidence=evidence)

"""Complete cold/default qualification cannot hide warm, accuracy or early loss."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/gga-indexed-gemm-20261108"
)


@pytest.mark.parametrize("optimized", [False, True])
def test_complete_gga_default_qualification_replays(optimized: bool) -> None:
    """Retain both experiment/default images and every complete population."""
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(DIRECTORY / "replay.py")],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "qualified": True,
        "populations": 70,
        "endpoints": 280,
    }


def test_early_loss_is_not_replaced_by_a_later_point_win() -> None:
    """A PBE0 DF near-tie cannot be reported as universal superiority."""
    module = runpy.run_path(str(DIRECTORY / "replay.py"))
    report = module["replay"]()
    first = next(
        row for row in report["groups"]["screen-a100"] if row["method"] == "pbe0"
    )
    assert first["timing"]["on"]["cold"] > first["timing"]["reference"]["cold"]
    final = next(
        row
        for row in report["groups"]["final-a100"]
        if row["method"] == "b3lyp" and row["mode"] == "df"
    )
    assert final["timing"]["default"]["cold"] < final["timing"]["reference"]["cold"]


@pytest.mark.parametrize(
    "corruption",
    [
        "checksum",
        "allocation",
        "accuracy",
        "work",
        "warm",
        "default_control",
        "missing_cohort",
    ],
)
def test_gga_replay_rejects_semantic_corruption(corruption: str) -> None:
    """Rehashing an altered receipt cannot manufacture scientific acceptance."""
    module = runpy.run_path(str(DIRECTORY / "replay.py"))
    evidence = module["load_evidence"]()
    if corruption == "missing_cohort":
        del evidence["groups"]["final-5090"]
    else:
        name = "raw/final-a100/pbe0-df-default." + (
            "env" if corruption == "default_control" else "json"
        )
        if corruption == "checksum":
            evidence["files"][name] += "\n"
        elif corruption == "default_control":
            evidence["files"][name] += "GENERATIVEQC_CUDA_XC_GGA_GEMM=1\n"
        else:
            record = json.loads(evidence["files"][name])
            if corruption == "allocation":
                record["scheduler"]["LSB_JOBID"] = "other-allocation"
            elif corruption == "accuracy":
                record["samples"][0]["forces"][0][0] += 0.01
            elif corruption == "work":
                record["samples"][0]["initial_guess"]["preliminary_fock_builds"] += 1
            elif corruption == "warm":
                record["samples"][1]["seconds"] *= 1.2
            evidence["files"][name] = json.dumps(record, indent=2) + "\n"
        if corruption != "checksum":
            evidence["sha256"][name] = hashlib.sha256(
                evidence["files"][name].encode()
            ).hexdigest()
    with pytest.raises(ValueError):
        module["replay"](evidence=evidence)

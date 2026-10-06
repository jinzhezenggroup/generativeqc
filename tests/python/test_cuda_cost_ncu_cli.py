"""Offline CUDA cost reports can attach retained NCU mechanism calibration."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _row(function: str, registers: int) -> str:
    return (
        f"ptxas info : Compiling entry function '{function}' for 'sm_120'\n"
        + f"ptxas info : Function properties for {function}\n"
        + "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        + f"ptxas info : Used {registers} registers, 0 bytes smem\n"
    )


def _run(tmp_path: Path, evidence: object) -> subprocess.CompletedProcess[str]:
    ptxas = tmp_path / "ptxas.log"
    ptxas.write_text(_row("force_kernel", 255), encoding="utf-8")
    ncu = tmp_path / "ncu.json"
    ncu.write_text(json.dumps(evidence), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(_ROOT / "tools/analyze_cuda_cost.py"),
            "--arch",
            "sm_120",
            "--block-threads",
            "256",
            "--ptxas",
            str(ptxas),
            "--ncu-evidence",
            str(ncu),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_ROOT / "python")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _force_evidence() -> dict[str, object]:
    return {
        "schema": "generativeqc.compiler.ncu-execution-evidence.v1",
        "evidence": {
            "architecture": "sm_120",
            "source_revision": "c5ab37ec9e7b3d38d2e06729319f9eef66510e5c",
            "kernel_identity": "force_kernel",
            "report_sha256": "0" * 64,
            "device": "retained-test-device",
            "theoretical_occupancy_fraction": 1 / 6,
            "achieved_occupancy_fraction": 1 / 6,
            "executed_threads_per_warp_instruction": 17.91,
            "issue_active_fraction": 0.0643,
            "eligible_warps_per_scheduler": 0.07,
            "warp_cycles_per_issued_instruction": 31.10,
            "barrier_cycles_per_issued_instruction": 13.67,
            "wait_cycles_per_issued_instruction": 8.23,
            "short_scoreboard_cycles_per_issued_instruction": 5.50,
            "long_scoreboard_cycles_per_issued_instruction": 1.49,
            "local_load_requests": 58_849_833_769,
            "local_store_requests": 27_175_631_760,
            "dram_busy_fraction": 0.1693,
        },
    }


def test_cli_reports_ncu_mechanism_and_static_occupancy_calibration(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, _force_evidence())

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assessment = payload["ncu_execution_assessment"]
    calibration = payload["ncu_static_calibration"]

    assert "barrier-divergence-local-state" in assessment["mechanisms"]
    assert "occupancy-bound-is-not-root-cause-proof" in assessment["mechanisms"]
    assert calibration["static_occupancy_upper_bound"] == pytest.approx(1 / 6)
    assert calibration["ncu_theoretical_occupancy_fraction"] == pytest.approx(1 / 6)
    assert calibration["absolute_error"] == pytest.approx(0.0)
    assert calibration["within_two_percentage_points"] is True
    assert payload["screening_priority"]


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"schema": "unsupported", "evidence": {}},
        {
            "schema": "generativeqc.compiler.ncu-execution-evidence.v1",
            "evidence": {"achieved_occupancy_fraction": 16.67},
        },
    ],
)
def test_cli_rejects_malformed_ncu_evidence(tmp_path: Path, evidence: object) -> None:
    result = _run(tmp_path, evidence)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr

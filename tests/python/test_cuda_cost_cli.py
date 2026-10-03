"""Offline CUDA cost reports retain complete and target-matched PTXAS evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _row(function: str, registers: int, architecture: str | None = None) -> str:
    header = (
        ""
        if architecture is None
        else f"ptxas info : Compiling entry function '{function}' for '{architecture}'\n"
    )
    return (
        header
        + f"ptxas info : Function properties for {function}\n"
        + "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        + f"ptxas info : Used {registers} registers, 0 bytes smem\n"
    )


def _run(tmp_path: Path, log: str | None) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(_ROOT / "tools/analyze_cuda_cost.py"),
        "--arch",
        "sm_120",
        "--block-threads",
        "128",
    ]
    if log is not None:
        path = tmp_path / "ptxas.log"
        path.write_text(log, encoding="utf-8")
        args.extend(("--ptxas", str(path)))
    return subprocess.run(
        args,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_ROOT / "python")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("log", "message"),
    [
        ("", "requires PTXAS resource rows"),
        (
            _row("light", 32, "sm_120")
            + "ptxas info : Function properties for heavy\n"
            + "    1024 bytes stack frame, 128 bytes spill stores, 128 bytes spill loads\n",
            "incomplete or unsupported resource rows",
        ),
        (
            _row("kernel", 32, "sm_120")
            + "ptxas info : Function properties for kernel\n"
            + "    1024 bytes stack frame, 128 bytes spill stores, 128 bytes spill loads\n",
            "incomplete or unsupported resource rows",
        ),
        (
            _row("light", 32, "sm_120")
            + "ptxas info : Compiling entry function 'heavy' for 'sm_120'\n",
            "incomplete or unsupported resource rows",
        ),
        (_row("kernel", 32, "sm_80"), "do not match requested target sm_120"),
        (
            _row("kernel", 255, "sm_80") + _row("kernel", 32, "sm_120"),
            "do not match requested target sm_120",
        ),
    ],
)
def test_cli_rejects_incomplete_or_mismatched_ptxas(
    tmp_path: Path, log: str, message: str
) -> None:
    result = _run(tmp_path, log)

    assert result.returncode != 0
    assert result.stdout == ""
    assert message in result.stderr


def test_cli_reports_complete_multi_kernel_provenance(tmp_path: Path) -> None:
    result = _run(tmp_path, _row("light", 32, "sm_120") + _row("heavy", 128, "sm_120"))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ptxas_evidence"] == {
        "architecture": "sm_120",
        "architecture_verified": True,
        "functions": ["light", "heavy"],
    }
    assert payload["evidence_stage"] == "compiled"
    assert payload["registers_per_thread"] == 128
    assert payload["occupancy_upper_bound"] == pytest.approx(1 / 3)
    assert payload["diagnostics"] == []


@pytest.mark.parametrize("first_architecture", [None, "sm_120"])
def test_cli_discloses_missing_architecture_headers(
    tmp_path: Path, first_architecture: str | None
) -> None:
    result = _run(tmp_path, _row("light", 32, first_architecture) + _row("heavy", 128))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["architecture"] == "sm_120"
    assert payload["ptxas_evidence"]["architecture"] is None
    assert payload["ptxas_evidence"]["architecture_verified"] is False
    assert payload["registers_per_thread"] == 128
    assert any("PTXAS architecture is unverified" in d for d in payload["diagnostics"])


def test_static_cli_does_not_invent_ptxas_provenance(tmp_path: Path) -> None:
    result = _run(tmp_path, None)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["evidence_stage"] == "static"
    assert payload["ptxas_evidence"] is None

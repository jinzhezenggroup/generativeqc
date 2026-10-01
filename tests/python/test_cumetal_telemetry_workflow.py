"""Keep unprofiled workload and harness failures separate from telemetry."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"


def _step(name: str) -> str:
    return (
        WORKFLOW.read_text(encoding="utf-8")
        .split(f"      - name: {name}\n", 1)[1]
        .split("      - name:", 1)[0]
        .split("\n      #", 1)[0]
    )


@pytest.mark.parametrize(
    "case", ["valid", "zero", "negative", "nan", "inf", "missing", "not-ready", "exit"]
)
def test_direct_smoke_gate_rejects_invalid_workloads(tmp_path: Path, case: str) -> None:
    body = _step("Smoke CuMetal FP32 proxy workloads")
    command = textwrap.dedent(body.split("        run: |\n", 1)[1])
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    server = tmp_path / "benchmark"
    server.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "case = os.environ['TEST_CASE']\n"
        "if case != 'not-ready': print('READY device=fixture')\n"
        "for name in ('compute', 'memory', 'gather', 'mixed'):\n"
        "    if case == 'missing' and name == 'mixed': continue\n"
        "    value = {'zero': '0', 'negative': '-1', 'nan': 'nan', 'inf': 'inf'}.get(case, '0.1')\n"
        "    print(f'OK {name} {value}')\n"
        "sys.stdin.read()\n"
        "sys.exit(4 if case == 'exit' else 0)\n",
        encoding="utf-8",
    )
    server.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", command],
        cwd=tmp_path,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GENERATIVEQC_CUMETAL_FP32_BENCH": str(server),
            "TEST_CASE": case,
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode == 0) == (case == "valid"), result.stdout + result.stderr


def test_python_harness_remains_gating_before_optional_telemetry() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    name = "Validate CuMetal benchmark harness without profiling"
    gate = _step(name)
    telemetry = _step("Run CodSpeed FP32 CuMetal proxy suite")
    assert workflow.index(name) < workflow.index(
        "Run CodSpeed FP32 CuMetal proxy suite"
    )
    assert "continue-on-error" not in gate
    assert (
        "env -u CODSPEED_ENV .venv/bin/python -m pytest benchmarks/test_cumetal_fp32_codspeed.py -q"
        in gate
    )
    assert "--codspeed" not in gate
    assert "continue-on-error: true" in telemetry
    assert "--codspeed -q" in telemetry

"""D4 qualification must retain failed diagnostics without accepting empty work."""

from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[2] / "benchmarks/results/d4-capacity-20261029"


def trace(path: Path, *, kernels: int = 1, allocations: int = 1) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS "
        "(memoryOperationType INTEGER, bytes INTEGER, memKind INTEGER, start INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (nameId INTEGER);"
        "CREATE TABLE StringIds (id INTEGER, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (copyKind INTEGER, bytes INTEGER);"
        "CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, label TEXT);"
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)",
        [(index,) for index in range(kernels)],
    )
    for index in range(allocations):
        connection.executemany(
            "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS VALUES (?, ?, ?, ?)",
            [(0, 128, 2, index * 2), (1, 128, 2, index * 2 + 1)],
        )
    connection.commit()
    connection.close()


def profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, counts: dict[str, dict[str, int]]
) -> Path:
    for role in ("baseline", "candidate"):
        trace(tmp_path / f"{role}.sqlite", **counts[role])
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_summary",
            "--baseline",
            str(tmp_path / "baseline.sqlite"),
            "--candidate",
            str(tmp_path / "candidate.sqlite"),
            "--output",
            str(output),
        ],
    )
    runpy.run_path(str(BUNDLE / "profile_summary.py"))["main"]()
    return output


@pytest.mark.parametrize("field", ["kernels", "allocations"])
@pytest.mark.parametrize("role", ["baseline", "candidate", "both"])
def test_profile_requires_observed_cuda_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, role: str
) -> None:
    counts = {
        name: {"kernels": 1, "allocations": 1} for name in ("baseline", "candidate")
    }
    for name, count in counts.items():
        if role in (name, "both"):
            count[field] = 0
    with pytest.raises(SystemExit, match="captured no CUDA kernels or allocations"):
        profile(monkeypatch, tmp_path, counts)
    # Preserve the failed trace summary as a diagnostic, including equal empties.
    report = json.loads((tmp_path / "summary.json").read_text())
    key = field if field == "kernels" else "explicit_device_allocation_count"
    assert report["baseline" if role == "both" else role][key] == 0


@pytest.mark.parametrize("candidate_kernels", [1, 2])
def test_nonempty_profile_keeps_exact_work_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, candidate_kernels: int
) -> None:
    counts = {"baseline": {"kernels": 1}, "candidate": {"kernels": candidate_kernels}}
    if candidate_kernels == 2:
        with pytest.raises(SystemExit, match="actual GPU work or storage changed"):
            profile(monkeypatch, tmp_path, counts)
    else:
        report = json.loads(profile(monkeypatch, tmp_path, counts).read_text())
        assert report["baseline"] == report["candidate"]
        assert report["baseline"]["explicit_device_allocation_count"] == 1
        assert report["baseline"]["explicit_device_final_live_bytes"] == 0


def run_script(
    tmp_path: Path,
    name: str,
    *,
    qualification: int = 0,
    profiler: int = 0,
    mutate: bool = False,
    campaign: str = "all",
) -> tuple[subprocess.CompletedProcess[str], list[dict], Path]:
    """Adapt only deployment paths; exercise the actual shell control flow."""
    run_root, base_root = tmp_path / "run", tmp_path / "base"
    out, base_out = run_root / "artifacts", base_root / "artifacts"
    out.mkdir(parents=True)
    base_out.mkdir(parents=True)
    for directory in (out, base_out):
        (directory / "libgenerativeqc.so.0").write_bytes(b"frozen library fixture")
    for executable in (
        "sdk-cuda",
        "generativeqc_d4_production_tests",
        "generativeqc_d4_ragged_tests",
        "generativeqc_d4_eeq_cuda_tests",
        "generativeqc_d4_reference_cuda_tests",
        "generativeqc_d4_schedule_cuda_tests",
    ):
        path = out / executable
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    frozen = out / "frozen.txt"
    frozen.write_text("frozen input\n")
    (out / "input-sha256.txt").write_text(
        f"{hashlib.sha256(frozen.read_bytes()).hexdigest()}  {frozen}\n"
    )
    events = tmp_path / "events.jsonl"
    commands = tmp_path / "commands"
    commands.mkdir()
    stub = f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with Path({str(events)!r}).open('a') as stream:
    stream.write(json.dumps({{'command': name, 'args': args}}) + '\\n')
if name == 'python' and args and args[0].endswith('/qualify.py'):
    if os.environ['D4_TEST_MUTATE'] == '1':
        Path({str(frozen)!r}).write_text('changed input\\n')
    output = args[args.index('--output') + 1]
    campaign = 'budget' if 'budget' in Path(output).name else 'default'
    if os.environ['D4_TEST_CAMPAIGN'] in ('all', campaign):
        sys.exit(int(os.environ['D4_TEST_QUALIFICATION']))
if (name == 'nsys' and args[0] == 'profile') or (name == 'python' and args[:2] == ['-m', 'cProfile']):
    sys.exit(int(os.environ['D4_TEST_PROFILER']))
"""
    for command in ("python", "sccache", "nvidia-smi", "compute-sanitizer", "nsys"):
        path = commands / command
        path.write_text(stub)
        path.chmod(0o755)
    script = (BUNDLE / name).read_text()
    for variable, path in (
        ("run_root", run_root),
        ("base_root", base_root),
        ("reference_root", tmp_path / "reference"),
    ):
        script = re.sub(
            rf"^{variable}=.*$",
            f"{variable}={shlex.quote(str(path))}",
            script,
            flags=re.MULTILINE,
        )
    script = re.sub(
        r"^export PATH=.*$",
        f"export PATH={shlex.quote(str(commands))}:$PATH",
        script,
        flags=re.MULTILINE,
    )
    runner = tmp_path / name
    runner.write_text(script)
    result = subprocess.run(
        ["bash", str(runner)],
        env={
            **os.environ,
            "SLURM_JOB_ID": "fixture",
            "SLURMD_NODENAME": "fixture",
            "CUDA_VISIBLE_DEVICES": "0",
            "D4_TEST_QUALIFICATION": str(qualification),
            "D4_TEST_PROFILER": str(profiler),
            "D4_TEST_MUTATE": str(int(mutate)),
            "D4_TEST_CAMPAIGN": campaign,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    records = [json.loads(line) for line in events.read_text().splitlines()]
    return result, records, out


@pytest.mark.parametrize(
    "qualification,profiler,mutate,expected",
    [
        (0, 0, False, 0),
        (23, 0, False, 23),
        (0, 31, False, 31),
        (0, 0, True, 1),
        (23, 0, True, 23),
    ],
)
def test_image_diagnostic_always_checks_frozen_inputs(
    tmp_path: Path, qualification: int, profiler: int, mutate: bool, expected: int
) -> None:
    result, _, out = run_script(
        tmp_path,
        "profile-image-identity.sh",
        qualification=qualification,
        profiler=profiler,
        mutate=mutate,
    )
    assert result.returncode == expected, result.stderr
    receipt = (out / "integrity-after.log").read_text()
    assert ("FAILED" if mutate else "OK") in receipt


@pytest.mark.parametrize("campaign", ["default", "budget"])
def test_failed_timing_keeps_both_campaign_profiles(
    tmp_path: Path, campaign: str
) -> None:
    result, events, out = run_script(
        tmp_path, "run-gpu.sh", qualification=23, campaign=campaign
    )
    assert result.returncode == 23, result.stderr
    profiles = [
        row
        for row in events
        if row["command"] == "nsys" and row["args"][0] == "profile"
    ]
    summaries = [
        row
        for row in events
        if row["command"] == "python" and row["args"][0].endswith("/profile_summary.py")
    ]
    assert len(profiles) == 4 and len(summaries) == 2
    assert "OK" in (out / "integrity-after.log").read_text()


@pytest.mark.parametrize(
    "qualification,profiler,mutate,expected",
    [
        (0, 0, False, 0),
        (0, 31, False, 31),
        (23, 31, False, 23),
        (0, 0, True, 1),
        (23, 0, True, 23),
    ],
)
def test_runner_preserves_failures_and_final_integrity(
    tmp_path: Path, qualification: int, profiler: int, mutate: bool, expected: int
) -> None:
    result, _, out = run_script(
        tmp_path,
        "run-gpu.sh",
        qualification=qualification,
        profiler=profiler,
        mutate=mutate,
    )
    assert result.returncode == expected, result.stderr
    assert ("FAILED" if mutate else "OK") in (out / "integrity-after.log").read_text()

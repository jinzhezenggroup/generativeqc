"""Protect the finite cold CUDA build and failure-only, allowlisted diagnostics."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_ci_prose_workflow import assert_full_coverage_step, selected

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()


def _job(workflow: str) -> str:
    return re.split(
        r"\n  [A-Za-z0-9_-]+:\n",
        workflow.split("\n  cuda-compile:\n", 1)[1],
        maxsplit=1,
    )[0]


def _step(job: str, name: str) -> str:
    return job.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def _minutes(section: str) -> int:
    match = re.search(r"^\s*timeout-minutes: (\d+)$", section, re.MULTILINE)
    assert match, "execution must have a finite timeout"
    return int(match[1])


def _assert_cold_budget(workflow: str) -> None:
    job = _job(workflow)
    header = job.split("    steps:\n", 1)[0]
    build = _step(job, "Compile production CUDA and runtime CTest")
    prerequisite = _step(job, "Compile production CUDA prerequisite archive")
    # Both serial build phases must fit without consuming setup/probe/cleanup time.
    assert _minutes(prerequisite) == 10
    assert _minutes(build) == 30
    assert _minutes(prerequisite) + _minutes(build) + 10 <= _minutes(header) <= 50
    assert "continue-on-error:" not in header
    gates = (
        (
            "Configure production CUDA compile check",
            "-DGENERATIVEQC_COMPILER_CACHE=ccache",
        ),
        (
            "Compile production CUDA prerequisite archive",
            "--target generativeqc_direct_native --parallel 3",
        ),
        (
            "Compile production CUDA and runtime CTest",
            "--target generativeqc_cuda_runtime_tests --parallel 3",
        ),
        (
            "Compile opted-in AO radial kernels with release resources",
            "test_cuda_ao_radial_reuse.py::test_emitted_ao_cuda_compiles_with_specialized_resources",
        ),
        (
            "Check Fock and proposal preflight without a GPU",
            "ctest --test-dir build/cuda-release-sm120",
        ),
    )
    positions = []
    for name, required_work in gates:
        section = _step(job, name)
        assert required_work in section
        assert_full_coverage_step(section)
        assert "continue-on-error:" not in section
        positions.append(job.index(f"      - name: {name}\n"))
    assert positions == sorted(positions)
    radial = _step(job, gates[3][0])
    assert (
        "test_packaging.py::test_wheel_grid_compiler_links_without_checkout_or_device"
        in radial
    )
    assert "actions/cache/save@" not in job

    report = _step(job, "Summarize compiler cache")
    assert "if: always()" in report
    assert _minutes(report) == 1
    assert "--verbose --build-dir build/cuda-release-sm120" in report
    receipt = _step(job, "Preserve failed fast CUDA build diagnostics")
    for failed, cancelled in ((False, False), (True, False), (False, True)):
        assert selected(
            receipt,
            {
                "success()": not failed and not cancelled,
                "failure()": failed,
                "cancelled()": cancelled,
            },
        ) == (failed or cancelled)
    assert "continue-on-error: true" in receipt
    assert _minutes(receipt) == 2
    assert "include-hidden-files: true" in receipt
    assert "retention-days: 14" in receipt
    paths = re.search(r"          path: \|\n((?:            .+\n)+)", receipt)
    assert paths is not None
    assert set(paths[1].split()) == {
        "build/cuda-release-sm120/.ninja_log",
        "build/cuda-fast-compile-cache.txt",
    }
    assert job.index("      - name: Summarize compiler cache\n") < job.index(
        "      - name: Preserve failed fast CUDA build diagnostics\n"
    )


def test_cold_budget_keeps_qualification_and_failure_diagnostics() -> None:
    _assert_cold_budget(WORKFLOW)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("timeout-minutes: 50", "timeout-minutes: 35"),
        ("timeout-minutes: 50", "timeout-minutes: 40"),
        ("timeout-minutes: 50", "timeout-minutes: 45"),
        ("timeout-minutes: 50", "timeout-minutes: 60"),
        ("timeout-minutes: 10", "timeout-minutes: 0"),
        ("timeout-minutes: 10", "# timeout omitted"),
        ("timeout-minutes: 30", "timeout-minutes: 0"),
        ("timeout-minutes: 30", "# timeout omitted"),
        ("timeout-minutes: 30", "timeout-minutes: 30\n        continue-on-error: true"),
        ("timeout-minutes: 30", "timeout-minutes: 30\n        if: false"),
        ("--target generativeqc_cuda_runtime_tests --parallel 3", "--target help"),
        ("ctest --test-dir build/cuda-release-sm120", "echo skipped"),
        ("if: always()", "if: success()"),
        ("failure() || cancelled()", "success()"),
        ("build/cuda-release-sm120/.ninja_log", "build/cuda-release-sm120/**"),
        (
            "build/cuda-fast-compile-cache.txt",
            "build/cuda-release-sm120/CMakeCache.txt",
        ),
        ("--verbose --build-dir build/cuda-release-sm120", "--verbose"),
        ("retention-days: 14", "retention-days: 90"),
        ("uses: actions/cache/restore@", "uses: actions/cache/save@"),
    ],
)
def test_budget_diagnostic_and_gate_controls_reject_regressions(
    before: str, after: str
) -> None:
    job = _job(WORKFLOW)
    assert before in job
    broken = WORKFLOW.replace(job, job.replace(before, after))
    with pytest.raises(AssertionError):
        _assert_cold_budget(broken)


def test_cold_budget_rejects_probe_before_production_build() -> None:
    job = _job(WORKFLOW)
    start = job.index("      - name: Check Fock and proposal preflight without a GPU\n")
    end = job.index("      - name: Show CUDA ccache statistics\n", start)
    probe = job[start:end]
    changed = job[:start] + job[end:]
    position = changed.index(
        "      - name: Compile production CUDA and runtime CTest\n"
    )
    changed = changed[:position] + probe + changed[position:]
    with pytest.raises(AssertionError):
        _assert_cold_budget(WORKFLOW.replace(job, changed))


@pytest.mark.parametrize("configured", [False, True])
def test_diagnostic_step_runs_after_partial_setup_without_dumping_config(
    tmp_path: Path, configured: bool
) -> None:
    (tmp_path / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
    build = tmp_path / "build/cuda-release-sm120"
    if configured:
        build.mkdir(parents=True)
        (build / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            "GENERATIVEQC_CUDA_FAST_COMPILE:BOOL=ON\n"
            "SECRET_VALUE:STRING=must-not-be-uploaded\n"
        )
    script = textwrap.dedent(
        _step(_job(WORKFLOW), "Summarize compiler cache").split("        run: |\n", 1)[
            1
        ]
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to(sys.executable)
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "UNRELATED_SECRET": "must-not-be-uploaded",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = (tmp_path / "build/cuda-fast-compile-cache.txt").read_text()
    assert "Compiler cache: CUDA 12.9 compile sm120" in receipt
    assert "must-not-be-uploaded" not in receipt
    assert "SECRET_VALUE" not in receipt and "UNRELATED_SECRET" not in receipt
    assert ("CMAKE_BUILD_TYPE=Release" in receipt) is configured
    assert "CMake cache unavailable:" in receipt or configured

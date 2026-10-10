"""Keep reusable master caches independent of downstream test outcomes."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _job(name: str) -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    section = workflow.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [A-Za-z0-9_-]+:\n", section, maxsplit=1)[0]


def _step(job: str, name: str) -> str:
    return _job(job).split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def test_master_cpu_seed_covers_exact_pr_and_queue_cache_modes() -> None:
    seed = _job("master-cpu-cache")
    header = seed.split("\n    steps:\n", 1)[0]
    assert (
        "if: github.event_name == 'push' && github.ref == 'refs/heads/master'" in header
    )
    assert "compiler: [gcc, clang]" in header
    assert "mode: [plain]" in header
    assert "include:\n          - compiler: gcc\n            mode: coverage" in header
    assert "timeout-minutes: ${{ matrix.mode == 'coverage' && 10 || 8 }}" in header
    assert "fail-fast: false" in header
    restore = _step("master-cpu-cache", "Restore CPU ccache")
    consumer = _step("cpu", "Restore ccache")
    seed_inputs = re.findall(r"hashFiles\((.+)\)", restore)[0]
    consumer_inputs = re.findall(r"hashFiles\((.+)\)", consumer)[0]
    assert seed_inputs == consumer_inputs
    assert "ccache-cpu-v3-${{ matrix.compiler }}-${{ matrix.mode }}-" in restore
    assert "restore-keys:" in restore
    build = _step("master-cpu-cache", "Build CPU cache seed")
    configure = _step("cpu", "Configure")
    for flags in (
        "-fprofile-arcs -ftest-coverage -fprofile-update=atomic",
        "--coverage",
    ):
        assert flags in build and flags in configure
    for option in (
        '-DCMAKE_CXX_FLAGS="${COVERAGE_COMPILE_FLAGS}"',
        '-DCMAKE_EXE_LINKER_FLAGS="${COVERAGE_LINK_FLAGS}"',
        '-DCMAKE_SHARED_LINKER_FLAGS="${COVERAGE_LINK_FLAGS}"',
        "-DGENERATIVEQC_COMPILER_CACHE=ccache",
        "-DGENERATIVEQC_ENABLE_CUDA=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ):
        assert option in build and option in configure
    assert "cmake --build build --parallel" in build
    assert "ctest" not in seed and "lcov" not in seed
    save = _step("master-cpu-cache", "Save CPU ccache")
    assert "key: ${{ steps.master_cpu_ccache.outputs.cache-primary-key }}" in save
    assert "always()" not in save


def test_python_test_timeout_leaves_finite_setup_and_save_headroom() -> None:
    job = _job("python")
    tests = _step("python", "Run Python tests with coverage")
    save = _step("python", "Save native Python probes")
    pattern = (
        r"timeout-minutes: \$\{\{ \(github.event_name == 'schedule' \|\| "
        r"github.event_name == 'workflow_dispatch'\) && (\d+) \|\| (\d+) \}\}"
    )
    job_limits = re.search(pattern, job.split("\n    steps:\n", 1)[0])
    test_limits = re.search(pattern, tests)
    assert job_limits and test_limits
    assert tuple(map(int, test_limits.groups())) == (90, 30)
    assert tuple(map(int, job_limits.groups())) == (100, 40)
    assert "timeout-minutes: 2" in save
    assert "id: python_tests" in tests
    assert "continue-on-error:" not in tests
    assert "-m pytest" in tests and '"${test_targets[@]}"' in tests


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_partial_probe_snapshots_do_not_freeze_the_complete_key(outcome: str) -> None:
    save = _step("python", "Save native Python probes")
    expression = re.search(
        r"key: \$\{\{ steps.python_tests.outcome == 'success' && "
        r"steps.python_probe_ccache.outputs.cache-primary-key \|\| "
        r"format\('([^']+)', steps.python_probe_ccache.outputs.cache-primary-key, "
        r"github.run_id, github.run_attempt\) \}\}",
        save,
    )
    assert expression
    primary = "ccache-python-probes-v2-core-b-source-hash"
    actual = (
        primary if outcome == "success" else expression.group(1).format(primary, 123, 2)
    )
    assert actual == (primary if outcome == "success" else primary + "-partial-123-2")
    assert actual.startswith(primary)


def test_benchmark_compiler_cache_is_saved_before_endpoint_failures() -> None:
    job = _job("cpu-benchmark")
    assert (
        job.index("name: Restore CPU benchmark ccache")
        < job.index("name: Build CPU benchmark library")
        < job.index("name: Save CPU benchmark ccache after build")
        < job.index("name: Install benchmark dependencies")
        < job.index("name: Run CPU CodSpeed endpoint suite")
    )
    restore = _step("cpu-benchmark", "Restore CPU benchmark ccache")
    save = _step("cpu-benchmark", "Save CPU benchmark ccache after build")
    assert "uses: actions/cache/restore@" in restore
    assert "id: benchmark_ccache" in restore
    assert "uses: actions/cache/save@" in save
    assert "key: ${{ steps.benchmark_ccache.outputs.cache-primary-key }}" in save
    assert "steps.benchmark_ccache.outputs.cache-hit != 'true'" in save
    assert "env.GENERATIVEQC_CODSPEED_RUN != '0'" in save
    assert "always()" not in save and "failure()" not in save
    assert "continue-on-error:" not in _step(
        "cpu-benchmark", "Build CPU benchmark library"
    )
    assert "ccache-cpu-benchmark-v2-" in restore


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_codspeed_non_pr_events_initialize_all_setup_guards(event: str) -> None:
    header = _job("cpu-benchmark").split("\n    steps:\n", 1)[0]
    assert 'GENERATIVEQC_CODSPEED_RUN: "1"' in header
    selector = _step("cpu-benchmark", "Select change-aware PR CodSpeed coverage")
    assert "if: github.event_name == 'pull_request'" in selector
    assert event != "pull_request"
    for name in (
        "Install uv and Python 3.11",
        "Install ccache 4.14",
        "Restore CPU benchmark ccache",
        "Reset CPU benchmark ccache statistics",
        "Build CPU benchmark library",
        "Show CPU benchmark ccache statistics",
        "Install benchmark dependencies",
    ):
        assert "if: env.GENERATIVEQC_CODSPEED_RUN != '0'" in _step(
            "cpu-benchmark", name
        )
    run = _step("cpu-benchmark", "Run CPU CodSpeed endpoint suite")
    assert "github.event_name != 'pull_request'" in run
    assert "steps.baseline.outputs.qualified == 'true'" in run


@pytest.mark.parametrize(
    ("changed", "expected_run", "expected_extra"),
    [
        ("docs/index.md\n", "0", ""),
        ("tests/python/test_foo.py\n", "0", ""),
        ("manifests/maintenance/record.json\n", "0", ""),
        ("src/scf/rhf.cpp\n", "1", "wb97mv"),
        ("CMakeLists.txt\n", "1", ""),
        (".github/workflows/ci.yml\n", "1", "wb97mv"),
    ],
)
def test_pr_codspeed_selector_still_overrides_the_default(
    tmp_path: Path, changed: str, expected_run: str, expected_extra: str
) -> None:
    selector = _step("cpu-benchmark", "Select change-aware PR CodSpeed coverage")
    script = selector.split("        run: |\n", 1)[1]
    script = "\n".join(line[10:] for line in script.splitlines())
    script = script.replace(
        "${{ github.event.pull_request.base.sha }}...${{ github.event.pull_request.head.sha }}",
        "base...head",
    )
    git = tmp_path / "git"
    git.write_text('#!/bin/sh\nprintf "%s" "$CHANGED_FILES"\n')
    git.chmod(0o755)
    output = tmp_path / "environment"
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_ENV": str(output),
            "CHANGED_FILES": changed,
            "GENERATIVEQC_CODSPEED_RUN": "1",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    updates = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert updates["GENERATIVEQC_CODSPEED_RUN"] == expected_run
    assert updates["GENERATIVEQC_CODSPEED_EXTRA_CASES"] == expected_extra

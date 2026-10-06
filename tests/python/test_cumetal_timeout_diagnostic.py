"""A live timeout observation must not weaken scientific endpoint acceptance."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/diagnose_cumetal_timeout.py"
RUNNER = ROOT / ".github/scripts/run_cumetal_cuda_pytests.py"
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"


def _diagnostic() -> dict:
    return runpy.run_path(str(SCRIPT), run_name="cumetal_timeout_test")


@pytest.mark.parametrize("timeout", (False, True))
def test_live_native_output_survives_diagnostic_process_exit(
    tmp_path: Path, timeout: bool
) -> None:
    observe = _diagnostic()["observe"]
    monotonic = observe.__globals__["time"].monotonic
    command = [
        sys.executable,
        "-c",
        (
            "import os, time; os.write(2, b'[cumetal-reg] emit kernel fixture\\n'); "
            f"time.sleep({30 if timeout else 0})"
        ),
    ]
    result = observe(command, tmp_path, dict(os.environ), monotonic() + 1.5)
    assert result["timed_out"] == timeout
    assert (result["returncode"] < 0) if timeout else (result["returncode"] == 0)
    log = (tmp_path / "endpoint-live.txt").read_text()
    assert "t+" in log
    assert "[cumetal-reg] emit kernel fixture" in log
    assert "cannot qualify" in result["note"]
    assert (tmp_path / "process-00.txt").exists()


@pytest.mark.parametrize("modified_provider", (False, True))
def test_diagnostic_reuses_cache_and_exact_contract_without_qualifying_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modified_provider: bool
) -> None:
    diagnose = _diagnostic()["diagnose"]
    cache = tmp_path / "gate-cache/registration-jit"
    cache.mkdir(parents=True)
    cached = cache / "existing.metallib"
    cached.write_bytes(b"pre-existing cache content")
    library = tmp_path / "library"
    library.write_bytes(b"fixture native library")
    compiler = tmp_path / "bin/cumetalc"
    compiler.parent.mkdir()
    compiler.write_bytes(b"fixture compiler")
    env = {
        "CUMETAL_PREFIX": str(tmp_path),
        "CUMETAL_SOURCE": str(tmp_path / "source"),
        "CUMETAL_COMMIT": "fixture-pin",
        "CUMETAL_PTX_BACKEND": "cumetal-ir",
        "CUMETAL_FP64_MODE": "fast48",
        "CUMETAL_CACHE_DIR": str(cache.parent),
        "GENERATIVEQC_LIBRARY": str(library),
    }
    calls = []

    def git(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        assert command[0] == "git"
        return subprocess.CompletedProcess(
            command,
            0,
            "fixture-pin\n"
            if "HEAD" in command
            else " M source"
            if modified_provider
            else "",
        )

    def observe(command: list, output: Path, child_env: dict, deadline: float) -> dict:
        calls.append(command)
        assert child_env["CUMETAL_CACHE_DIR"] == env["CUMETAL_CACHE_DIR"]
        assert child_env["CUMETAL_PTX_BACKEND"] == "cumetal-ir"
        assert child_env["CUMETAL_FP64_MODE"] == "fast48"
        assert child_env["CUMETAL_DEBUG_REGISTRATION"] == "1"
        assert child_env["CUMETAL_DIAGNOSTIC_PHASES"] == "1"
        assert "-s" in command and "-u" in command
        assert str(output / "diagnostic-only.xml") in command[-1]
        (cache / "new.metallib").write_bytes(b"new cache entry")
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(diagnose.__globals__["platform"], "platform", lambda: "fixture")
    monkeypatch.setattr(subprocess, "run", git)
    monkeypatch.setitem(diagnose.__globals__, "observe", observe)
    output = tmp_path / "evidence"
    assert diagnose(output, env) == int(modified_provider)
    report = json.loads((output / "diagnostic.json").read_text())
    assert "not QC acceptance" in report["purpose"]
    assert "partially warmed" in report["purpose"]
    assert len(report["cache_before"]) == 1
    assert cached.read_bytes() == b"pre-existing cache content"
    if not modified_provider:
        assert len(calls) == 1
        assert len(report["cache_after"]) == 2
        assert report["cache_before"][0] == report["cache_after"][0]
    else:
        assert calls == []


def test_diagnostic_refuses_stale_evidence(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        _diagnostic()["diagnose"](tmp_path, {})


@pytest.mark.parametrize("timeout,node", ((False, 0), (True, 0), (True, 1)))
def test_only_real_rhf_gate_timeout_sets_workflow_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool, node: int
) -> None:
    runner = runpy.run_path(str(RUNNER), run_name="cumetal_gate_test")
    main = runner["main"]
    runtime = main.__globals__
    monkeypatch.setitem(runtime, "MODE", "gate")
    monkeypatch.setitem(
        runtime, "selected_nodeids", lambda: [runner["GATE_NODEIDS"][node]]
    )
    monkeypatch.setitem(runtime, "Path", lambda _: tmp_path / "endpoint.xml")
    monkeypatch.setitem(runtime, "stream_process", lambda *_: (1, "", timeout))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github-output"))
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    output = tmp_path / "github-output"
    assert output.exists() == (timeout and node == 0)
    if output.exists():
        assert output.read_text() == "rhf_timed_out=true\n"


def test_workflow_timeout_diagnostic_is_one_shot_separate_and_same_cache() -> None:
    workflow = WORKFLOW.read_text()
    gate = workflow.split("- name: Run bounded CuMetal QC endpoint gate", 1)[1].split(
        "\n      - name:", 1
    )[0]
    diagnostic = workflow.split("- name: Diagnose one CuMetal RHF timeout", 1)[1].split(
        "\n      - name:", 1
    )[0]
    for condition in (
        "failure()",
        "github.event_name == 'pull_request'",
        "github.event.pull_request.number == 1997",
        "github.event.action == 'synchronize'",
        "github.event.before == 'de9eebb88be4d7ded840a35541daff2ce43a7a44'",
        "github.run_attempt == 1",
        "steps.qc_gate.outcome == 'failure'",
        "steps.qc_gate.outputs.rhf_timed_out == 'true'",
    ):
        assert condition in diagnostic
    assert "continue-on-error: true" in diagnostic
    assert "timeout-minutes: 4" in diagnostic
    assert "continue-on-error" not in gate
    assert 'CUMETAL_CUDA_TEST_TIMEOUT_SECONDS: "90"' in gate
    assert 'CUMETAL_CUDA_SUITE_BUDGET_SECONDS: "300"' in gate
    for key in ("CUMETAL_PTX_BACKEND", "CUMETAL_FP64_MODE", "CUMETAL_CACHE_DIR"):
        gate_value = next(
            line.strip() for line in gate.splitlines() if f"{key}:" in line
        )
        assert gate_value in diagnostic
    artifact = workflow.split(
        "- name: Preserve one-shot CuMetal timeout diagnostic", 1
    )[1].split("\n      - name:", 1)[0]
    assert "always()" in artifact and "retention-days: 1" in artifact
    assert "name: cumetal-timeout-diagnostic-only" in artifact


def test_master_cache_saves_remain_read_only_on_merge_groups_and_precede_gate() -> None:
    workflow = WORKFLOW.read_text()
    for name, restore, predecessor in (
        ("Save CuMetal compiler cache", "cumetal_ccache", "Build pinned CuMetal"),
        (
            "Save CuMetal toolchain",
            "cache-cumetal",
            "Generate a CMake CUDA toolkit backed by CuMetal",
        ),
        (
            "Save GenerativeQC ccache",
            "generativeqc_ccache",
            "Build CUDA runtime test target",
        ),
    ):
        step = workflow.split(f"- name: {name}", 1)[1].split("\n      - name:", 1)[0]
        assert "github.event_name != 'merge_group'" in step
        assert "always()" not in step and "failure()" not in step
        assert "uses: actions/cache/save@" in step
        assert f"steps.{restore}.outputs.cache-primary-key" in step
        assert workflow.index(f"- name: {predecessor}") < workflow.index(
            f"- name: {name}"
        )
        assert workflow.index(f"- name: {name}") < workflow.index(
            "- name: Run bounded CuMetal QC endpoint gate"
        )
    assert "uses: actions/cache@" not in workflow
    assert "\n  push:" not in workflow
    assert "\n  cumetal-benchmark:" not in workflow
    assert "cancel-in-progress: false" in workflow

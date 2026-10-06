"""Keep CuMetal CI honest about runtime coverage and bounded in cost."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / ".github/scripts/run_cumetal_cuda_pytests.py"
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"


def _runner() -> dict[str, object]:
    return runpy.run_path(str(RUNNER), run_name="generativeqc_cumetal_ci_test")


def test_routine_cumetal_gate_spans_scientific_cuda_owners() -> None:
    namespace = _runner()
    gate = tuple(namespace["GATE_NODEIDS"])
    assert (
        "tests/python/test_cuda_runtime.py::test_cuda_minimal_rhf_matches_cpu_reference"
        in gate
    )
    assert (
        "tests/python/test_cuda_runtime.py::test_cuda_minimal_uhf_matches_cpu_reference"
        in gate
    )
    assert (
        "tests/python/test_cuda_runtime.py::test_cuda_minimal_density_fitting_matches_cpu_reference"
        in gate
    )
    assert (
        "tests/python/test_cuda_runtime.py::test_cuda_minimal_pbe_rks_matches_cpu_reference"
        in gate
    )


def test_cumetal_qualification_excludes_known_nvidia_only_response_gate() -> None:
    namespace = _runner()
    qualification = tuple(namespace["QUALIFICATION_NODEIDS"])
    assert set(namespace["GATE_NODEIDS"]).issubset(qualification)
    assert len(qualification) > len(namespace["GATE_NODEIDS"])
    assert not any(
        "resident_rhf_response_matches_host_operator" in item for item in qualification
    )


def test_cumetal_runner_requires_per_group_gpu_provenance_and_time_budget() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"device=apple_gpu" not in output' in source
    assert '"launch_success=true" not in output' in source
    assert "MISSING APPLE-GPU PROVENANCE" in source
    assert "CUMETAL_CUDA_SUITE_BUDGET_SECONDS" in source
    assert "SUITE BUDGET EXHAUSTED" in source


def test_cumetal_workflow_keeps_routine_and_qualification_budgets_separate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "Run bounded CuMetal QC endpoint gate" in workflow
    assert "CUMETAL_CUDA_TEST_MODE: gate" in workflow
    assert 'CUMETAL_CUDA_SUITE_BUDGET_SECONDS: "300"' in workflow
    assert "Run bounded CuMetal QC qualification on the Apple GPU" in workflow
    assert "CUMETAL_CUDA_TEST_MODE: full" in workflow
    assert 'CUMETAL_CUDA_SUITE_BUDGET_SECONDS: "2400"' in workflow
    assert 'GENERATIVEQC_DFT_CUDA_TEST: "1"' in workflow
    assert "Report NVIDIA-only resident-response test coverage" not in workflow


@pytest.mark.parametrize("trace_every_endpoint", (False, True))
def test_junit_provenance_isolated_for_each_native_endpoint(
    tmp_path: Path, trace_every_endpoint: bool
) -> None:
    """Exercise real pytest FD capture, including parameterized native output."""
    test_file = tmp_path / "test_native_endpoint.py"
    test_file.write_text(
        "import os\n"
        "import pytest\n"
        "@pytest.mark.parametrize('endpoint', ('lda-rks', 'pbe-rks', 'lda-uks', 'pbe-uks'))\n"
        "def test_endpoint(endpoint):\n"
        f"    if endpoint == 'lda-rks' or {trace_every_endpoint!r}:\n"
        "        os.write(2, b'device=apple_gpu launch_success=true\\n')\n",
        encoding="utf-8",
    )
    junit = tmp_path / "endpoint.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "--capture=fd",
            "-o",
            "junit_logging=all",
            "-o",
            "junit_log_passing_tests=true",
            f"--junitxml={junit}",
            "-q",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases, skipped, missing = _runner()["junit_status"](junit)
    assert cases == 4
    assert skipped == []
    assert missing == (
        []
        if trace_every_endpoint
        else [
            "test_endpoint[pbe-rks]",
            "test_endpoint[lda-uks]",
            "test_endpoint[pbe-uks]",
        ]
    )


def test_junit_missing_or_malformed_reports_fail_closed(tmp_path: Path) -> None:
    status = _runner()["junit_status"]
    junit = tmp_path / "endpoint.xml"
    assert status(junit) == (0, [], [])
    junit.write_text("<testsuite><testcase", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        status(junit)
    junit.write_text("<testsuite />", encoding="utf-8")
    assert status(junit) == (0, [], [])


@pytest.mark.parametrize("missing_provenance", (False, True))
def test_runner_rejects_shared_process_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_provenance: bool
) -> None:
    namespace = _runner()
    main = namespace["main"]
    runtime = main.__globals__
    calls = []
    monkeypatch.setitem(runtime, "selected_nodeids", lambda: ["test_group"])
    monkeypatch.setitem(runtime, "Path", lambda _: tmp_path / "endpoint.xml")

    def run(command: list[str], timeout: int) -> tuple[int, str, bool]:
        calls.append((command, timeout))
        suite = ET.Element("testsuite")
        for name in ("first", "second"):
            case = ET.SubElement(suite, "testcase", name=name)
            if name == "first" or not missing_provenance:
                ET.SubElement(
                    case, "system-err"
                ).text = "device=apple_gpu launch_success=true"
        ET.ElementTree(suite).write(tmp_path / "endpoint.xml")
        # A successful process-wide dispatch must not mask an untraced sibling.
        return 0, "device=apple_gpu launch_success=true", False

    monkeypatch.setitem(runtime, "stream_process", run)
    if missing_provenance:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
    else:
        main()
    command, timeout = calls[0]
    assert "--capture=fd" in command
    assert "junit_logging=all" in command
    assert "junit_log_passing_tests=true" in command
    assert "-ra" in command
    assert "-s" not in command
    assert 0 < timeout <= runtime["TIMEOUT_SECONDS"]


def test_runner_retains_native_cuda_skip_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mandatory endpoint's skip must expose the actual runtime failure."""
    namespace = _runner()
    main = namespace["main"]
    runtime = main.__globals__
    monkeypatch.setitem(runtime, "selected_nodeids", lambda: ["test_group"])
    monkeypatch.setitem(runtime, "Path", lambda _: tmp_path / "endpoint.xml")
    reason = "CUDA device unavailable: native kernel registration failed"

    def run(command: list[str], timeout: int) -> tuple[int, str, bool]:
        suite = ET.Element("testsuite")
        case = ET.SubElement(suite, "testcase", name="test_rhf")
        ET.SubElement(case, "skipped", message=reason)
        ET.SubElement(case, "system-err").text = "native runtime diagnostics"
        ET.ElementTree(suite).write(tmp_path / "endpoint.xml")
        return 0, "test_rhf SKIPPED", False

    monkeypatch.setitem(runtime, "stream_process", run)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert (
        f"SKIPPED: test_group (1/1 cases): test_rhf: {reason}"
        in capsys.readouterr().err
    )
    assert (tmp_path / "endpoint.xml").exists()


def test_cumetal_workflow_preserves_endpoint_diagnostics_after_qualification() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for name in (
        "Run bounded CuMetal QC endpoint gate",
        "Run bounded CuMetal QC qualification on the Apple GPU",
    ):
        endpoint_step = workflow.split(f"- name: {name}", 1)[1]
        endpoint_step = endpoint_step.split("\n      - name:", 1)[0]
        assert 'CUMETAL_DEBUG_REGISTRATION: "1"' in endpoint_step
    step = workflow.split("- name: Preserve CuMetal QC endpoint diagnostics", 1)[1]
    step = step.split("\n  cumetal-benchmark:", 1)[0]
    assert "if: always()" in step
    assert "path: /tmp/generativeqc-cuda-test-*.xml" in step
    assert workflow.index("Preserve CuMetal QC endpoint diagnostics") > workflow.index(
        "Run bounded CuMetal QC qualification on the Apple GPU"
    )


def test_cumetal_qc_toolchain_matches_ptx_deployment_target() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runtime_job = workflow.split("\n  cuda-tests:", 1)[1].split(
        "\n  cumetal-benchmark:", 1
    )[0]
    assert "runs-on: macos-26" in runtime_job
    assert "CUMETAL_TOOLCHAIN_ID: macos26-xcode26.3-sdk26.2" in runtime_job
    assert 'CUMETAL_XCODE_VERSION: "26.3"' in runtime_job
    assert 'CUMETAL_MACOS_SDK_VERSION: "26.2"' in runtime_job
    assert (
        "DEVELOPER_DIR: /Applications/Xcode_26.3.app/Contents/Developer" in runtime_job
    )
    for step in runtime_job.split("      - name: ")[1:]:
        name = step.splitlines()[0]
        if name in {
            "Cache CuMetal toolchain",
            "Restore CuMetal toolchain",
            "Restore CuMetal ccache",
            "Restore GenerativeQC ccache",
        }:
            key = step.split("          key:", 1)[1].splitlines()[0]
            assert "${{ env.CUMETAL_TOOLCHAIN_ID }}" in key
        if name in {"Cache CuMetal toolchain", "Restore CuMetal toolchain"}:
            assert "restore-keys:" not in step
        if name.startswith("Run bounded CuMetal QC"):
            assert (
                "CUMETAL_CACHE_DIR: ${{ runner.temp }}/cumetal-qc-jit-${{ env.CUMETAL_TOOLCHAIN_ID }}"
                in step
            )


@pytest.mark.parametrize("failure", (None, "missing-xcode", "xcode", "sdk", "os"))
def test_cumetal_toolchain_preflight_fails_closed(
    tmp_path: Path, failure: str | None
) -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Verify Apple Silicon and CuMetal toolchain", 1)[1]
    script = step.split("        run: |\n", 1)[1].split("\n      - ", 1)[0]
    script = "\n".join(line[10:] for line in script.splitlines())
    developer = tmp_path / "Xcode_26.3.app/Contents/Developer"
    if failure != "missing-xcode":
        developer.mkdir(parents=True)
    commands = {
        "uname": "echo arm64",
        "sw_vers": f"echo {'15.7.9' if failure == 'os' else '26.6.2'}",
        "xcodebuild": f"echo 'Xcode {'26.2' if failure == 'xcode' else '26.3'}'",
        "xcrun": (
            'if [ "$*" = "--sdk macosx --show-sdk-version" ]; then\n'
            f"echo {'26.1' if failure == 'sdk' else '26.2'}\n"
            "else echo 'mock Metal toolchain'; fi"
        ),
    }
    for name, body in commands.items():
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DEVELOPER_DIR": str(developer),
            "CUMETAL_XCODE_VERSION": "26.3",
            "CUMETAL_MACOS_SDK_VERSION": "26.2",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert (result.returncode == 0) == (failure is None), result.stdout + result.stderr

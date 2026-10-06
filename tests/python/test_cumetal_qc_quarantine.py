"""A pinned upstream exception must never turn other QC failures into passes."""

from __future__ import annotations

import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / ".github/scripts/run_cumetal_cuda_pytests.py"
MANIFEST = ROOT / "manifests/cumetal_qc_quarantine.json"
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"
PIN = "e87f368060cf09a45b148b4f8892470c74093ebd"
QUARANTINE_ID = "upstream-ptx-e87f368"
EXPECTED_NODES = (
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_rhf_matches_cpu_reference",
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_uhf_matches_cpu_reference",
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_density_fitting_matches_cpu_reference",
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_pbe_rks_matches_cpu_reference",
    "tests/python/test_dft_cuda.py::test_native_cuda_dft_matches_independently_converged_cpu_endpoint",
    "tests/python/test_calculator.py::test_cuda_energy_only_output_selection_omits_forces",
    "tests/python/test_batch.py::test_cuda_real_spherical_batch_reuses_fixed_topology_plan",
    "tests/python/test_batch.py::test_cuda_bounded_direct_streaming_matches_exact_replay",
    "tests/python/test_batch.py::test_cuda_uhf_ragged_batch_warm_start_and_failure_isolation",
    "tests/python/test_calculator.py::test_cuda_spherical_def2_svp_water_matches_pyscf",
    "tests/python/test_calculator.py::test_cuda_def2_tzvp_water_uses_graph_native_eigensolver",
    "tests/python/test_calculator.py::test_cartesian_d_f_cuda_matches_pyscf_libcint_reference",
    "tests/python/test_calculator.py::test_cuda_uhf_direct_jk_matches_pyscf_and_force_finite_difference",
    "tests/python/test_dft_cuda.py::test_native_cuda_dft_ragged_batch_replay_and_failure_isolation",
)


def _runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str = "gate"
) -> dict[str, object]:
    for variable, value in {
        "CUMETAL_CUDA_TEST_MODE": mode,
        "CUMETAL_CUDA_QUARANTINE": QUARANTINE_ID,
        "CUMETAL_COMMIT": PIN,
        "CUMETAL_PTX_BACKEND": "cumetal-ir",
        "CUMETAL_FP64_MODE": "fast48",
        "CUMETAL_ROOT": "/tmp/cumetal-install",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "GITHUB_OUTPUT": str(tmp_path / "outputs.txt"),
    }.items():
        monkeypatch.setenv(variable, value)
    namespace = runpy.run_path(str(RUNNER), run_name="test_cumetal_quarantine")
    runtime = namespace["main"].__globals__
    monkeypatch.setitem(runtime, "Path", lambda path: tmp_path / Path(path).name)
    return runtime


@pytest.mark.parametrize("mode,count", (("gate", 4), ("full", 14)))
def test_only_exact_pinned_groups_are_reported_as_unqualified_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    count: int,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, mode=mode)
    manifest = json.loads(MANIFEST.read_text())
    assert tuple(manifest["nodeids"]) == EXPECTED_NODES
    assert manifest["provider_commit"] == PIN
    assert manifest["id"] == QUARANTINE_ID

    def unexpected_execution(*args: object) -> None:
        pytest.fail("A quarantined group must be reported as skipped, never executed")

    monkeypatch.setitem(runtime, "stream_process", unexpected_execution)
    runtime["main"]()
    reports = sorted(tmp_path.glob("generativeqc-cuda-test-*.xml"))
    assert len(reports) == count
    names = set()
    for report in reports:
        suite = ET.parse(report).getroot()
        assert suite.get("skipped") == "1"
        case = suite.find("testcase")
        names.add(case.get("name"))
        assert manifest["reason"] in case.find("skipped").get("message")
        assert case.find("system-err") is None
        assert case.find("system-out") is None
    assert names == set(EXPECTED_NODES[:count])
    summary = (tmp_path / "summary.md").read_text()
    assert "QC NOT QUALIFIED" in summary
    assert manifest["reason"] in summary
    assert all(f"- SKIPPED: {nodeid}" in summary for nodeid in names)
    assert "Executed 0 CuMetal CUDA endpoint groups" in capsys.readouterr().out
    assert not (tmp_path / "outputs.txt").exists()


@pytest.mark.parametrize(
    "variable,value",
    (
        ("CUMETAL_CUDA_QUARANTINE", "anything"),
        ("CUMETAL_COMMIT", "new-provider-pin"),
        ("CUMETAL_COMMIT", None),
        ("CUMETAL_PTX_BACKEND", "legacy"),
        ("CUMETAL_PTX_BACKEND", None),
        ("CUMETAL_FP64_MODE", "fp32"),
        ("CUMETAL_FP64_MODE", None),
        ("CUMETAL_ROOT", None),
        ("CUMETAL_ROOT", ""),
        ("CUMETAL_CUDA_TEST_MODE", "anything"),
    ),
)
def test_contract_change_fails_before_any_skip_or_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str, value: str | None
) -> None:
    runtime = _runtime(monkeypatch, tmp_path)
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)
    if variable == "CUMETAL_CUDA_TEST_MODE":
        monkeypatch.setitem(runtime, "MODE", value)
    with pytest.raises(SystemExit):
        runtime["main"]()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "nodes",
    ([], "all", ["*"], ["test_nvidia_only"], [EXPECTED_NODES[0]] * 2, [None]),
)
def test_malformed_or_out_of_scope_manifest_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, nodes: object
) -> None:
    runtime = _runtime(monkeypatch, tmp_path)
    manifest = json.loads(MANIFEST.read_text())
    manifest["nodeids"] = nodes
    modified_manifest = tmp_path / "invalid.json"
    modified_manifest.write_text(json.dumps(manifest))
    monkeypatch.setitem(runtime, "QUARANTINE_MANIFEST", modified_manifest)
    with pytest.raises(SystemExit, match="invalid exact CuMetal"):
        runtime["main"]()
    assert not list(tmp_path.glob("*.xml"))


@pytest.mark.parametrize("quarantine_enabled", (False, True))
@pytest.mark.parametrize(
    "outcome",
    (
        "pass",
        "skip",
        "exit",
        "missing-trace",
        "empty",
        "missing",
        "malformed",
        "timeout",
        "budget",
    ),
)
def test_strict_acceptance_survives_for_every_executed_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    quarantine_enabled: bool,
    outcome: str,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path)
    if quarantine_enabled:
        # Even a parameter spelling beneath an allowed group is not an exact
        # match. Only the allowlisted selection itself may be skipped.
        target = EXPECTED_NODES[0] + "[new-endpoint]"
        selected = [EXPECTED_NODES[0], target]
    else:
        monkeypatch.delenv("CUMETAL_CUDA_QUARANTINE")
        target = EXPECTED_NODES[0]
        selected = [target]
    monkeypatch.setitem(runtime, "selected_nodeids", lambda: selected)
    calls = []

    def run(command: list[str], timeout: int) -> tuple[int, str, bool]:
        calls.append(command[3])
        assert command[3] == target
        assert 0 < timeout <= runtime["TIMEOUT_SECONDS"]
        junit = Path(command[-1].split("=", 1)[1])
        if outcome == "malformed":
            junit.write_text("<testsuite><testcase")
        elif outcome != "missing":
            suite = ET.Element("testsuite")
            if outcome != "empty":
                case = ET.SubElement(suite, "testcase", name="actual_endpoint")
                if outcome == "skip":
                    ET.SubElement(
                        case, "skipped", message="new failure outside quarantine"
                    )
                if outcome != "missing-trace":
                    ET.SubElement(
                        case, "system-err"
                    ).text = "device=apple_gpu launch_success=true"
            ET.ElementTree(suite).write(junit)
        return (2 if outcome == "exit" else 0), "", outcome == "timeout"

    monkeypatch.setitem(runtime, "stream_process", run)
    if outcome == "budget":
        times = iter((0, runtime["SUITE_BUDGET_SECONDS"] + 1))
        monkeypatch.setattr(runtime["time"], "monotonic", lambda: next(times))
    if outcome == "pass":
        runtime["main"]()
        assert "Executed 1 CuMetal CUDA endpoint groups" in capsys.readouterr().out
    elif outcome == "malformed":
        with pytest.raises(ET.ParseError):
            runtime["main"]()
    else:
        with pytest.raises(SystemExit) as error:
            runtime["main"]()
        assert error.value.code == 1
        assert target in capsys.readouterr().err
    assert calls == ([] if outcome == "budget" else [target])


def test_quarantine_opt_in_is_scoped_to_both_qc_steps_only() -> None:
    workflow = WORKFLOW.read_text()
    steps = workflow.split("      - name: ")[1:]
    enabled_steps = {
        step.splitlines()[0]
        for step in steps
        if f"CUMETAL_CUDA_QUARANTINE: {QUARANTINE_ID}" in step
    }
    assert enabled_steps == {
        "Run bounded CuMetal QC endpoint gate",
        "Run bounded CuMetal QC qualification on the Apple GPU",
    }
    for name in (
        "Configure GenerativeQC with CUDA tests enabled",
        "Build CUDA runtime test target",
        "Prepare CuMetal execution image for the normal CTest",
        "Run CUDA runtime CTest on the Apple GPU",
        *enabled_steps,
    ):
        step = next(step for step in steps if step.startswith(name + "\n"))
        assert "continue-on-error:" not in step
        assert "if: false" not in step
    runtime = next(
        step
        for step in steps
        if step.startswith("Run CUDA runtime CTest on the Apple GPU\n")
    )
    assert "set -o pipefail" in runtime
    assert "grep -q 'device=apple_gpu'" in runtime
    assert "grep -q 'launch_success=true'" in runtime
    assert "grep -q 'PASS: CUDA runtime contracts'" in runtime
    assert "-DGENERATIVEQC_CUDA_PROVIDER=cumetal" in workflow
    assert f"CUMETAL_COMMIT: {PIN}" in workflow

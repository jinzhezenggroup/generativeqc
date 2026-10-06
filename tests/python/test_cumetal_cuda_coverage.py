"""Keep CuMetal CI honest about runtime coverage and bounded in cost."""

from __future__ import annotations

import runpy
from pathlib import Path

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
        "resident_rhf_response_matches_host_operator" in item
        for item in qualification
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

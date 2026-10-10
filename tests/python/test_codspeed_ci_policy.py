"""Guard identical bounded PR/master CodSpeed selection and the full tier."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_codspeed_pr_tier_matches_master_and_stays_change_aware() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  cpu-benchmark:\n", 1)[1].split(
        "\n  upload-coverage:\n", 1
    )[0]
    assert "if: github.event_name != 'merge_group'" in job
    assert (
        "GENERATIVEQC_CODSPEED_TIER: "
        "${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'push') && 'pr' || 'full' }}"
    ) in job
    assert (
        "GENERATIVEQC_CODSPEED_EXTRA_CASES: "
        "${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'push') && 'wb97mv' || '' }}"
    ) in job
    assert "Select change-aware PR CodSpeed coverage" in job
    # A PR path selector must not mutate the fixed job-level selection.
    assert "GENERATIVEQC_CODSPEED_EXTRA_CASES=" not in job
    assert "GENERATIVEQC_CODSPEED_RUN=0" in job
    assert "Qualify PR CodSpeed comparison" in job
    assert "steps.baseline.outputs.qualified == 'true'" in job
    assert "Record successful master CodSpeed baseline" in job
    assert "codspeed-cpu-baseline-${{ github.sha }}" in job
    assert "github.event_name == 'push'" in job
    assert "manifests/maintenance/" in job
    assert "cpu-benchmark" not in workflow.split("\n  pass:\n", 1)[1]

    benchmark = (ROOT / "benchmarks/test_cpu_codspeed.py").read_text(encoding="utf-8")
    assert benchmark.count("pr_fast=True") == 2
    assert '"water-rhf-sto3g"' in benchmark
    assert '"formaldehyde-rhf-def2-svp"' in benchmark
    assert '"water-pbe-sto3g"' in benchmark
    assert '"water-wb97mv-smallgrid-sto3g"' in benchmark
    assert 'pr_extra="wb97mv"' in benchmark
    assert "GENERATIVEQC_CODSPEED_EXTRA_CASES" in benchmark
    assert "grid_shape=(12, 4, 8)" in benchmark
    assert "test_cpu_pbe_force_walltime" in benchmark
    assert 'properties=("energy", "forces")' in benchmark
    assert "test_cpu_rhf_changed_geometry_pair_walltime" in benchmark
    assert "prepare_batch([_WATER], warm_start=True)" in benchmark


def test_shared_pr_tier_preserves_master_collection_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Collect the real benchmark definitions without loading a native library
    # or executing any endpoint. Only these three imported names are needed.
    package = ModuleType("generativeqc")
    for name in ("Calculator", "GridSpec", "KsOptions"):
        setattr(package, name, object)
    monkeypatch.setitem(sys.modules, "generativeqc", package)
    monkeypatch.setenv("GENERATIVEQC_CODSPEED_TIER", "pr")
    monkeypatch.setenv("GENERATIVEQC_CODSPEED_EXTRA_CASES", "wb97mv")
    namespace = runpy.run_path(str(ROOT / "benchmarks/test_cpu_codspeed.py"))
    cases = namespace["_active_cases"]()
    assert [case.name for case in cases] == [
        "water-rhf-sto3g",
        "water-pbe-sto3g",
        "water-wb97mv-smallgrid-sto3g",
    ]
    assert [name for name in namespace if name.startswith("test_")] == [
        "test_cpu_warm_endpoint_walltime",
        "test_cpu_pbe_force_walltime",
        "test_cpu_rhf_changed_geometry_pair_walltime",
    ]
    # The two unparameterized endpoints make five PR-tier benchmarks. Full
    # scheduled/manual coverage additionally retains the larger RHF case.
    monkeypatch.setenv("GENERATIVEQC_CODSPEED_TIER", "full")
    monkeypatch.setenv("GENERATIVEQC_CODSPEED_EXTRA_CASES", "")
    assert [case.name for case in namespace["_active_cases"]()] == [
        *[case.name for case in cases],
        "formaldehyde-rhf-def2-svp",
    ]


def test_qualifier_authenticates_tested_merge_not_stale_payload_base() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    step = workflow.split("      - name: Qualify PR CodSpeed comparison\n", 1)[1]
    step = step.split("      - name:", 1)[0]
    assert "CODSPEED_MERGE_SHA: ${{ github.sha }}" in step
    assert "CODSPEED_HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in step
    assert "CODSPEED_BASE_REF: ${{ github.event.pull_request.base.ref }}" in step
    assert '--merge-sha "$CODSPEED_MERGE_SHA"' in step
    assert '--head-sha "$CODSPEED_HEAD_SHA"' in step
    assert '--base-ref "$CODSPEED_BASE_REF"' in step
    assert "github.event.pull_request.base.sha" not in step
    assert "--base-sha" not in step

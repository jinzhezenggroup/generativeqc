"""Keep merge-queue CI from spending runners on orphaned synthetic commits."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _job(source: str, name: str) -> str:
    section = source.split(f"\n  {name}:\n", 1)[1]
    match = re.search(r"\n  [A-Za-z0-9_-]+:\n", section)
    return section if match is None else section[: match.start()]


def test_required_merge_group_jobs_use_the_liveness_gate() -> None:
    expected = {
        "ci.yml": ("cpu", "cuda-compile", "cuda-resources", "python"),
        "cumetal-cuda.yml": ("cuda-tests",),
    }
    for filename, jobs in expected.items():
        source = (WORKFLOWS / filename).read_text(encoding="utf-8")
        gate = _job(source, "merge_queue_liveness")
        assert "/git/ref/${ref}" in gate
        assert "continue-on-error: true" in gate
        assert "running CI fail-open" in gate
        assert "active=false" in gate
        for job in jobs:
            section = _job(source, job)
            job_header = section.split("\n    steps:\n", 1)[0]
            assert "needs: merge_queue_liveness" in job_header
            assert (
                "if: needs.merge_queue_liveness.outputs.active != 'false'" in job_header
            )
            assert "always()" not in job_header


def test_merge_group_concurrency_preserves_running_same_ref_runs() -> None:
    for filename in ("ci.yml", "cumetal-cuda.yml"):
        source = (WORKFLOWS / filename).read_text(encoding="utf-8")
        concurrency = source.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        assert "github.event_name == 'merge_group'" in concurrency
        assert "&& github.ref" in concurrency
        assert "cancel-in-progress: false" in concurrency


def test_pr_concurrency_preserves_running_and_replaces_only_pending_runs() -> None:
    for filename in (
        "ci.yml",
        "cumetal-cuda.yml",
        "pre-commit.yml",
        "pr-overlap.yml",
        "wheels.yml",
    ):
        source = (WORKFLOWS / filename).read_text(encoding="utf-8")
        concurrency = source.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        assert "github.event.pull_request.number" in concurrency
        assert "cancel-in-progress: false" in concurrency


def test_ci_aggregate_accepts_only_explicitly_confirmed_orphans() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    section = _job(source, "pass")
    assert "merge_queue_liveness" in section
    assert "Accept an orphaned merge-group run" in section
    assert "needs.merge_queue_liveness.outputs.active == 'false'" in section
    assert "needs.merge_queue_liveness.outputs.active != 'false'" in section
    assert (
        "if: ${{ !cancelled() && !(github.event_name == 'push' && github.ref == 'refs/heads/master') }}"
        in section
    )


def test_dequeue_cleanup_cancels_only_runs_with_deleted_queue_refs() -> None:
    source = (WORKFLOWS / "merge-queue-cleanup.yml").read_text(encoding="utf-8")
    assert "pull_request_target:" in source
    assert "types: [dequeued]" in source
    assert "actions: write" in source
    assert "-f event=merge_group" in source
    assert "gh-readonly-queue/" in source
    assert "/git/ref/heads/${head_branch}" in source
    assert "404)" in source
    assert "/actions/runs/${run_id}/cancel" in source
    assert "Could not verify" in source


@pytest.mark.parametrize(
    ("event", "ref", "expected"),
    [
        ("pull_request", "refs/pull/1/merge", True),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-1", True),
        ("schedule", "refs/heads/master", True),
        ("workflow_dispatch", "refs/heads/master", True),
        ("push", "refs/heads/master", False),
        ("push", "refs/heads/feature", True),
    ],
)
@pytest.mark.parametrize("cancelled", [True, False])
def test_aggregate_gates_required_events_and_skips_only_master_pushes(
    event: str, ref: str, expected: bool, cancelled: bool
) -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    expression = _job(source, "pass").split("    if: ${{ ", 1)[1].split(" }}", 1)[0]
    expression = (
        expression.replace("!cancelled()", '"$CANCELLED" == false')
        .replace("github.event_name", '"$EVENT"')
        .replace("github.ref", '"$REF"')
        .replace("!(", "! (")
    )
    result = subprocess.run(
        ["bash", "-c", f"if [[ {expression} ]]; then printf run; else printf skip; fi"],
        env={
            **os.environ,
            "CANCELLED": str(cancelled).lower(),
            "EVENT": event,
            "REF": ref,
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout == ("run" if expected and not cancelled else "skip")


def test_wheels_preserve_scoped_pr_and_periodic_qualification() -> None:
    source = (WORKFLOWS / "wheels.yml").read_text(encoding="utf-8")
    triggers = source.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
    assert set(re.findall(r"^  ([a-z_]+):", triggers, re.MULTILINE)) == {
        "pull_request",
        "push",
        "schedule",
        "workflow_dispatch",
    }
    assert '- cron: "17 4 * * 1"' in triggers
    pull_request = triggers.split("  pull_request:\n", 1)[1].split(
        "  workflow_dispatch:", 1
    )[0]
    assert set(re.findall(r'      - "([^"]+)"', pull_request)) == {
        ".github/workflows/wheels.yml",
        "cmake/GenerativeQCCudaImplib.cmake",
        "cmake/3rdparty/implib/**",
        "pyproject.toml",
        "python/ci/prepare-wheel-build.sh",
        "python/ci/repair-wheel.sh",
        "src/runtime/nvidia_host_api/**",
        "tools/generate_cuda_implib.py",
        "tools/link_cuda_implib.py",
    }


def test_wheels_refresh_default_branch_cache_on_every_master_push() -> None:
    source = (WORKFLOWS / "wheels.yml").read_text(encoding="utf-8")
    triggers = source.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
    push = triggers.split("  push:\n", 1)[1]
    push = re.split(r"^  [a-z_]+:", push, maxsplit=1, flags=re.MULTILINE)[0]
    # No paths filter: source/compiler changes must also refresh master wheels.
    assert push.strip() == "branches: [master]"


def test_wheels_coalesce_only_pending_runs_in_the_matching_scope() -> None:
    source = (WORKFLOWS / "wheels.yml").read_text(encoding="utf-8")
    concurrency = source.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
    assert (
        "group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' "
        "&& github.event.pull_request.number || github.ref }}" in concurrency
    )
    assert "cancel-in-progress: false" in concurrency

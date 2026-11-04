"""Keep merge-queue CI from spending runners on orphaned synthetic commits."""

import os
import re
import subprocess
from pathlib import Path

import pytest
from test_ci_prose_workflow import expression

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PR_WORKFLOWS = (
    "array-api",
    "cumetal-cuda",
    "docs",
    "pre-commit",
    "pr-overlap",
    "work-audit",
)


def _policy(workflow: str, context: dict[str, str]) -> tuple[str, bool]:
    source = (WORKFLOWS / f"{workflow}.yml").read_text()
    match = re.search(r"^concurrency:\n((?:[ \t]+[^\n]*\n)+)", source, re.MULTILINE)
    assert match, f"Missing concurrency policy for {workflow}"
    fields = dict(re.findall(r"^  ([\w-]+): (.+)$", match[1], re.MULTILINE))
    group = re.sub(
        r"\$\{\{.*?\}\}",
        lambda value: str(expression(value[0], context)),
        fields["group"],
    )
    cancel = fields["cancel-in-progress"]
    return group, cancel == "true" or (
        cancel != "false" and bool(expression(cancel, context))
    )


def _context(event: str, run_id: str, *, pull: str = "17") -> dict[str, str]:
    return {
        "github.workflow": "qualification",
        "github.event_name": event,
        "github.event.pull_request.number": pull if event == "pull_request" else "",
        "github.ref": "refs/pull/17/merge"
        if event == "pull_request"
        else "refs/heads/master",
        "github.run_id": run_id,
    }


@pytest.mark.parametrize("workflow", PR_WORKFLOWS)
def test_only_revisions_of_the_same_pr_share_a_preemptible_group(workflow: str) -> None:
    first = _policy(workflow, _context("pull_request", "101"))
    replacement = _policy(workflow, _context("pull_request", "102"))
    unrelated = _policy(workflow, _context("pull_request", "103", pull="18"))
    assert first == replacement
    assert first[0] != unrelated[0]
    assert first[1] and replacement[1] and unrelated[1]


@pytest.mark.parametrize("workflow", ("array-api", "cumetal-cuda", "docs"))
@pytest.mark.parametrize(
    "event", ("push", "merge_group", "workflow_dispatch", "schedule")
)
def test_non_pr_qualification_never_preempts_a_running_check(
    workflow: str, event: str
) -> None:
    group, cancel = _policy(workflow, _context(event, "101"))
    assert not cancel
    assert group != _policy(workflow, _context("pull_request", "101"))[0]


@pytest.mark.parametrize("event", ("push", "merge_group"))
def test_array_api_keeps_pending_non_pr_checks_independent(event: str) -> None:
    # cancel-in-progress=false alone still replaces an older pending run.
    first = _context(event, "101")
    second = _context(event, "102")
    assert _policy("array-api", first)[0] != _policy("array-api", second)[0]


@pytest.mark.parametrize("workflow", ("cumetal-cuda", "docs"))
def test_existing_merge_queue_ref_groups_remain_nonpreemptible(workflow: str) -> None:
    first = _context("merge_group", "101")
    first["github.ref"] = "refs/heads/gh-readonly-queue/master/pr-17-first"
    second = {**first, "github.run_id": "102"}
    different = {
        **second,
        "github.ref": "refs/heads/gh-readonly-queue/master/pr-18-second",
    }
    assert _policy(workflow, first) == _policy(workflow, second)
    assert _policy(workflow, first)[0] != _policy(workflow, different)[0]
    assert not _policy(workflow, first)[1]


def _job(source: str, name: str) -> str:
    section = source.split(f"\n  {name}:\n", 1)[1]
    match = re.search(r"\n  [A-Za-z0-9_-]+:\n", section)
    return section if match is None else section[: match.start()]


def test_required_merge_group_jobs_use_the_liveness_gate() -> None:
    expected = {
        "ci.yml": (
            "cpu",
            "cuda-compile",
            "cuda-resources",
            "cuda-direct-aot",
            "python",
        ),
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
        # Both workflows only cancel superseded PR revisions, never running
        # merge-group qualification for an existing synthetic queue ref.
        assert (
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}"
            in concurrency
        )


def test_pr_concurrency_cancels_only_superseded_nonrelease_checks() -> None:
    # The wheel build intentionally finishes its in-flight package/cache work.
    # All other PR-scoped checks may preempt stale heads while their merge
    # groups and master cache producers retain their protected run identities.
    expected = {
        "ci.yml": "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
        "cumetal-cuda.yml": (
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}"
        ),
        "docs.yml": "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
        "array-api.yml": (
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}"
        ),
        "pre-commit.yml": "cancel-in-progress: true",
        "pr-overlap.yml": "cancel-in-progress: true",
        "work-audit.yml": "cancel-in-progress: true",
        "wheels.yml": "cancel-in-progress: false",
    }
    for filename, policy in expected.items():
        source = (WORKFLOWS / filename).read_text(encoding="utf-8")
        concurrency = source.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        assert "github.event.pull_request.number" in concurrency
        assert policy in concurrency


def test_ci_aggregate_accepts_only_explicitly_confirmed_orphans() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    section = _job(source, "pass")
    assert "merge_queue_liveness" in section
    assert "Accept an orphaned merge-group run" in section
    assert "needs.merge_queue_liveness.outputs.active == 'false'" in section
    assert "needs.merge_queue_liveness.outputs.active != 'false'" in section
    assert "if: ${{ !cancelled() && github.event_name != 'push' }}" in section


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
        ("workflow_dispatch", "refs/tags/v0.1.0", True),
        # Push runs build master caches/versioned artifacts; their required
        # CPU/Python jobs are skipped, so the correctness aggregate skips too.
        ("push", "refs/heads/master", False),
        ("push", "refs/heads/feature", False),
        ("push", "refs/tags/v0.1.0", False),
    ],
)
@pytest.mark.parametrize("cancelled", [True, False])
def test_aggregate_gates_required_events_and_skips_push_runs(
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
        "CMakeLists.txt",
        ".github/workflows/wheels.yml",
        "cmake/GenerativeQCCudaImplib.cmake",
        "cmake/cuda_wheel_payload.ld",
        "cmake/3rdparty/implib/**",
        "pyproject.toml",
        "src/cli/wheel_launcher.cpp",
        "python/generativeqc/tune_cli.py",
        "python/ci/prepare-wheel-build.sh",
        ".github/actions/restore-wheel-ccache/**",
        "tools/ci_cpu_ccache_snapshot.py",
        "tools/ci_select_wheel_matrix.py",
        "python/ci/repair-wheel.sh",
        "src/runtime/nvidia_host_api/**",
        "tools/generate_cuda_implib.py",
        "tools/link_cuda_implib.py",
        "tools/generate_nvidia_host_api.py",
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

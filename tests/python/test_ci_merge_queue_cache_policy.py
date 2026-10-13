"""Merge-queue cache provenance, finite transport, and isolated fallback."""

import ast
import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
DOWNLOAD = "Restore trusted master CUDA ccache snapshot"
ACTIVATE = "Activate verified trusted CUDA ccache snapshot"
DISCARD = "Discard incomplete trusted CUDA cache download"
FALLBACK = "Restore shared Linux CUDA ccache"


def _step(name: str, workflow: str = WORKFLOW) -> str:
    marker = f"      - name: {name}\n"
    start = workflow.index(marker)
    end = workflow.find("      - name: ", start + len(marker))
    return workflow[start:] if end < 0 else workflow[start:end]


def _field(section: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}: (.+)$", section, re.MULTILINE)
    assert match, f"Missing {key}"
    return match[1]


def _condition(step: str, context: dict[str, str]) -> bool:
    expression = _field(step, "if")
    for key, value in context.items():
        expression = expression.replace(key, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")

    def evaluate(node: ast.AST) -> str | bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool)):
            return node.value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            equal = evaluate(node.left) == evaluate(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return equal
            if isinstance(node.ops[0], ast.NotEq):
                return not equal
        if isinstance(node, ast.BoolOp):
            values = [bool(evaluate(value)) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        raise AssertionError(f"Unsupported condition: {ast.dump(node)}")

    return bool(evaluate(ast.parse(expression, mode="eval").body))


def test_trusted_cuda_snapshot_is_scoped_to_successful_master_ci_pushes() -> None:
    discovery = _step("Locate trusted master CUDA ccache snapshot")
    assert "/actions/workflows/ci.yml/runs" in discovery
    assert "-f branch=master -f event=push -f status=success" in discovery
    assert '.event == "push"' in discovery
    assert '.head_branch == "master"' in discovery
    assert '.conclusion == "success"' in discovery
    assert "/actions/runs/${candidate_run_id}/artifacts" in discovery
    assert "/actions/artifacts" not in discovery


def test_trusted_cuda_snapshot_failures_fall_back_instead_of_gating_ci() -> None:
    discovery = _step("Locate trusted master CUDA ccache snapshot")
    download = _step("Restore trusted master CUDA ccache snapshot")
    fallback = _step("Restore shared Linux CUDA ccache")
    publish = _step("Publish trusted master CUDA ccache snapshot")

    assert 'if runs_payload="$(gh api --method GET' in discovery
    assert 'if ! artifacts_payload="$(gh api --method GET' in discovery
    assert "falling back to actions/cache or a cold compile" in discovery
    assert "continue-on-error: true" in download
    assert "steps.trusted_cuda_ccache_activate.outcome != 'success'" in fallback
    assert "continue-on-error: true" in publish


def _assert_isolated_transport(workflow: str) -> None:
    download = _step(DOWNLOAD, workflow)
    activate = _step(ACTIVATE, workflow)
    discard = _step(DISCARD, workflow)
    fallback = _step(FALLBACK, workflow)
    assert 0 < int(_field(download, "timeout-minutes")) <= 2
    assert 0 < int(_field(activate, "timeout-minutes")) <= 1
    assert 0 < int(_field(discard, "timeout-minutes")) <= 1
    for optional in (download, activate, discard):
        assert _field(optional, "continue-on-error") == "true"
    assert _field(download, "digest-mismatch") == "error"
    assert (
        _field(download, "run-id") == "${{ steps.trusted_cuda_ccache.outputs.run_id }}"
    )
    assert _field(download, "repository") == "${{ github.repository }}"
    assert _field(download, "github-token") == "${{ github.token }}"
    staging = _field(download, "path")
    assert staging == _field(activate, "TRUSTED_CUDA_CCACHE_DIR")
    assert staging == _field(discard, "TRUSTED_CUDA_CCACHE_DIR")
    assert "${{ runner.temp }}" in staging
    assert "${{ github.run_id }}" in staging
    assert "${{ github.run_attempt }}" in staging
    assert _field(fallback, "path") == "~/.cache/ccache"
    assert _field(fallback, "path") != staging

    for event in ("merge_group", "pull_request", "push", "workflow_dispatch"):
        for run_id in ("", "12345"):
            for outcome in ("success", "failure", "cancelled", "skipped", ""):
                context = {
                    "github.event_name": event,
                    "steps.trusted_cuda_ccache.outputs.run_id": run_id,
                    "steps.trusted_cuda_ccache_download.outcome": outcome,
                    "steps.trusted_cuda_ccache_activate.outcome": outcome,
                }
                assert _condition(download, context) == (
                    event == "merge_group" and bool(run_id)
                )
                assert _condition(activate, context) == (
                    event == "merge_group" and outcome == "success"
                )
                assert _condition(fallback, context) == (
                    event != "merge_group" or outcome != "success"
                )
                assert _condition(discard, context) == (
                    event == "merge_group" and outcome in ("failure", "cancelled")
                )

    # Transport may choose a cache, never suppress or soften required work.
    for name in (
        "Reset ccache statistics",
        "Configure production CUDA compile check",
        "Compile production CUDA and runtime CTest",
        "Compile opted-in AO radial kernels with release resources",
        "Check Fock and proposal preflight without a GPU",
    ):
        cuda_job = workflow.split("\n  cuda-compile:\n", 1)[1]
        required = _step(name, cuda_job)
        assert "if:" not in required and "continue-on-error:" not in required
    assert (
        workflow.index(DOWNLOAD)
        < workflow.index(ACTIVATE)
        < workflow.index(DISCARD)
        < workflow.index(FALLBACK)
    )
    assert workflow.index(FALLBACK) < workflow.index(
        "Reset ccache statistics", workflow.index(FALLBACK)
    )


def test_trusted_restore_has_a_deadline_and_activates_only_verified_staging() -> None:
    _assert_isolated_transport(WORKFLOW)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("timeout-minutes: 2", "timeout-minutes: 35"),
        ("digest-mismatch: error", "digest-mismatch: warn"),
        ("continue-on-error: true", "continue-on-error: false"),
        (".outcome == 'success'", ".conclusion == 'success'"),
        (".outcome != 'success'", ".outcome == 'success'"),
        ("${{ runner.temp }}/trusted-cuda-ccache-", "~/.cache/ccache/"),
        ("run-id: ${{ steps.trusted_cuda_ccache.outputs.run_id }}", "run-id: 1"),
    ],
    ids=[
        "deadline",
        "digest",
        "non-gating",
        "outcome",
        "fallback",
        "isolation",
        "provenance",
    ],
)
def test_restore_policy_rejects_unsafe_transport_mutations(
    before: str, after: str
) -> None:
    start = WORKFLOW.index(f"      - name: {DOWNLOAD}\n")
    end = WORKFLOW.index("      - name: Reset ccache statistics\n", start)
    section = WORKFLOW[start:end]
    assert before in section
    broken = WORKFLOW[:start] + section.replace(before, after) + WORKFLOW[end:]
    with pytest.raises(AssertionError):
        _assert_isolated_transport(broken)


@pytest.mark.parametrize(
    ("outcome", "downloaded"),
    [
        ("success", b"completed and verified snapshot"),
        ("failure", b"interrupted extraction"),
        ("failure", b"completed extraction with rejected digest"),
        ("cancelled", b"cancelled extraction"),
        ("skipped", b"untouched unrelated staging"),
        ("", b"untouched preexisting staging"),
    ],
    ids=["success", "timeout", "checksum-failure", "cancelled", "skipped", "not-run"],
)
@pytest.mark.parametrize(
    "staging_exists", [True, False], ids=["partial-or-complete", "missing"]
)
def test_actual_activation_shell_never_selects_failed_or_missing_staging(
    tmp_path: Path, outcome: str, downloaded: bytes, staging_exists: bool
) -> None:
    home = tmp_path / "home"
    normal = home / ".cache/ccache"
    normal.mkdir(parents=True)
    existing = normal / "existing-cache-entry"
    existing.write_bytes(b"retain this unrelated cache")
    runner_temp = tmp_path / "runner temp"
    staging = runner_temp / "trusted-cuda-ccache-1-2"
    if staging_exists:
        staging.mkdir(parents=True)
        (staging / "cache-entry").write_bytes(downloaded)
    environment_file = tmp_path / "github-env"
    environment_file.touch()
    activate = _step(ACTIVATE)
    context = {
        "github.event_name": "merge_group",
        "steps.trusted_cuda_ccache_download.outcome": outcome,
    }
    activation_outcome = "skipped"
    if _condition(activate, context):
        script = textwrap.dedent(activate.split("        run: |\n", 1)[1])
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            env={
                **os.environ,
                "HOME": str(home),
                "GITHUB_ENV": str(environment_file),
                "TRUSTED_CUDA_CCACHE_DIR": str(staging),
            },
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        activation_outcome = "success" if result.returncode == 0 else "failure"

    selected = outcome == "success" and staging_exists
    assert environment_file.read_text() == (
        f"CCACHE_DIR={staging}\n" if selected else ""
    )
    context["steps.trusted_cuda_ccache_activate.outcome"] = activation_outcome
    assert _condition(_step(FALLBACK), context) is not selected
    discard = _step(DISCARD)
    discarded = _condition(discard, context)
    if discarded:
        result = subprocess.run(
            [
                "bash",
                "-e",
                "-o",
                "pipefail",
                "-c",
                textwrap.dedent(discard.split("        run: |\n", 1)[1]),
            ],
            env={
                **os.environ,
                "RUNNER_TEMP": str(runner_temp),
                "GITHUB_RUN_ID": "1",
                "GITHUB_RUN_ATTEMPT": "2",
                "TRUSTED_CUDA_CCACHE_DIR": str(staging),
                "CCACHE_DIR": str(staging) if selected else "",
            },
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    # Interrupted/mismatched extraction stays outside the normal fallback cache.
    assert list(normal.iterdir()) == [existing]
    assert existing.read_bytes() == b"retain this unrelated cache"
    assert staging.exists() == (staging_exists and not discarded)
    if staging.exists():
        assert (staging / "cache-entry").read_bytes() == downloaded


@pytest.mark.parametrize(
    "case",
    [
        "empty-path",
        "normal-cache",
        "selected-cache",
        "missing-temp",
        "missing-run",
        "missing-attempt",
    ],
)
def test_discard_shell_refuses_unexpected_or_selected_cache_paths(
    tmp_path: Path, case: str
) -> None:
    runner_temp = tmp_path / "runner-temp"
    staging = runner_temp / "trusted-cuda-ccache-1-2"
    normal = tmp_path / "home/.cache/ccache"
    for directory in (staging, normal):
        directory.mkdir(parents=True)
        (directory / "keep").write_bytes(b"must survive refused cleanup")
    env = {
        **os.environ,
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "2",
        "TRUSTED_CUDA_CCACHE_DIR": str(staging),
        "CCACHE_DIR": "",
    }
    key, value = {
        "empty-path": ("TRUSTED_CUDA_CCACHE_DIR", ""),
        "normal-cache": ("TRUSTED_CUDA_CCACHE_DIR", str(normal)),
        "selected-cache": ("CCACHE_DIR", str(staging)),
        "missing-temp": ("RUNNER_TEMP", ""),
        "missing-run": ("GITHUB_RUN_ID", ""),
        "missing-attempt": ("GITHUB_RUN_ATTEMPT", ""),
    }[case]
    env[key] = value
    script = textwrap.dedent(_step(DISCARD).split("        run: |\n", 1)[1])
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    for directory in (staging, normal):
        assert (directory / "keep").read_bytes() == b"must survive refused cleanup"

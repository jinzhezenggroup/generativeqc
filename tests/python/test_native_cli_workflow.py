"""Behavioral controls for native CLI archive and required aggregate admission."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.package_native_cli import package_native_cli

ROOT = Path(__file__).resolve().parents[2]


def _job(name: str) -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    body = workflow.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-zA-Z0-9_-]+:\n", body, maxsplit=1)[0]


def _evaluate(node: ast.AST) -> str | bool:
    """Interpret string equality and short-circuit operators without executing code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool)):
        return node.value
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
    ):
        equal = _evaluate(node.left) == _evaluate(node.comparators[0])
        return equal if isinstance(node.ops[0], ast.Eq) else not equal
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        result = _evaluate(node.values[0])
        for value in node.values[1:]:
            if (isinstance(node.op, ast.And) and result) or (
                isinstance(node.op, ast.Or) and not result
            ):
                result = _evaluate(value)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "startsWith"
        and len(node.args) == 2
        and not node.keywords
    ):
        value, prefix = (_evaluate(arg) for arg in node.args)
        assert isinstance(value, str) and isinstance(prefix, str)
        return value.startswith(prefix)
    raise AssertionError(f"Unsupported workflow expression: {ast.dump(node)}")


@pytest.mark.parametrize("tier", ["skip", "fast", "release"])
@pytest.mark.parametrize(
    ("event", "ref"),
    [
        ("push", "refs/heads/master"),
        ("push", "refs/heads/feature"),
        ("push", "refs/tags/v1.0.0"),
        ("pull_request", "refs/pull/1/merge"),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-1"),
        ("workflow_dispatch", "refs/heads/master"),
        ("schedule", "refs/heads/master"),
    ],
)
@pytest.mark.parametrize(
    "step_name",
    [
        "Restore shared Linux CUDA ccache",
        "Seed fast CUDA CI objects in the shared cache",
        "Build monolithic C++ CUDA CLI (release or fast PR)",
        "Verify full automatic CUDA XC catalog is compiled for distribution",
        "Smoke-test release-mode native command",
    ],
)
def test_native_cli_tier_admits_all_selected_build_stages(
    tier: str, event: str, ref: str, step_name: str
) -> None:
    step = (
        _job("native-cli-artifact")
        .split(f"      - name: {step_name}\n", 1)[1]
        .split("\n      - ", 1)[0]
    )
    condition = re.search(r"^        if: (.+)$", step, re.MULTILINE)
    assert condition is not None
    expression = (
        condition.group(1)
        .replace("steps.cli_scope.outputs.tier", repr(tier))
        .replace("github.event_name", repr(event))
        .replace("github.ref", repr(ref))
        .replace("&&", " and ")
        .replace("||", " or ")
    )
    expected = tier != "skip"
    if step_name == "Seed fast CUDA CI objects in the shared cache":
        # Cache production is master-only; consumer build/test stages still run
        # for every selected fast/release tier on PR and merge-queue events.
        expected = expected and event == "push" and ref == "refs/heads/master"
    assert _evaluate(ast.parse(expression, mode="eval").body) == expected


def _dependencies(name: str) -> set[str]:
    raw = re.search(r"^    needs: (.+)$", _job(name), re.MULTILINE)
    assert raw is not None
    return {item.strip() for item in raw[1].strip("[]").split(",")}


def _condition(name: str) -> str:
    # Read the scalar job condition, including YAML's folded continuation lines.
    raw = re.search(r"^    if: (.+)((?:\n      .+)*)", _job(name), re.MULTILINE)
    assert raw is not None
    return " ".join(raw[2].split()) if raw[1] == ">-" else raw[1]


def _expression(value: str, context: dict[str, str | bool]) -> str | bool:
    value = value.removeprefix("${{ ").removesuffix(" }}")
    for name in sorted(context, key=len, reverse=True):
        value = value.replace(name, repr(context[name]))
    value = value.replace("&&", " and ").replace("||", " or ")
    value = re.sub(r"!(?!=)", " not ", value).strip()
    return _evaluate(ast.parse(value, mode="eval").body)


def _schedule(
    name: str,
    event: str,
    ref: str,
    results: dict[str, str],
    *,
    active: str = "true",
    cancelled: bool = False,
) -> str:
    """Execute the workflow's dependency/condition contract, not native builds."""
    needed = _dependencies(name)
    terminal = {"success", "failure", "skipped", "cancelled"}
    if any(results.get(dependency) not in terminal for dependency in needed):
        return "waiting"
    condition = _condition(name)
    context = {
        "github.event_name": event,
        "github.ref": ref,
        "needs.merge_queue_liveness.outputs.active": active,
        "cancelled()": cancelled,
        **{f"needs.{key}.result": value for key, value in results.items()},
    }
    # GitHub adds success() to job ifs without a status-check function. Thus a
    # failed/skipped/cancelled dependency cannot be bypassed by a true event if.
    if "cancelled()" not in condition and (
        cancelled or any(results[dependency] != "success" for dependency in needed)
    ):
        return "skipped"
    return "ready" if _expression(condition, context) else "skipped"


def _allowed_skips(event: str) -> set[str]:
    allowed = re.search(r"allowed-skips: (.+)", _job("pass"))
    assert allowed is not None
    result = _expression(allowed[1], {"github.event_name": event})
    assert isinstance(result, str)
    return {name.strip() for name in result.split(",") if name.strip()}


def _aggregate(event: str, results: dict[str, str]) -> bool:
    # Execute the inputs to pinned alls-green: only explicitly permitted skips
    # are accepted. Failure/cancellation never inherits an allowed-skip exception.
    assert "jobs: ${{ toJSON(needs) }}" in _job("pass")
    assert "allowed-failures" not in _job("pass")
    allowed = _allowed_skips(event)
    return all(
        results.get(name) == "success"
        or (results.get(name) == "skipped" and name in allowed)
        for name in _dependencies("pass")
    )


EVENTS = [
    pytest.param(
        "pull_request",
        "refs/pull/1/merge",
        "native-cli-qualification",
        id="same-repo-pr",
    ),
    pytest.param(
        "pull_request", "refs/pull/2/merge", "native-cli-qualification", id="fork-pr"
    ),
    pytest.param(
        "merge_group",
        "refs/heads/gh-readonly-queue/master/pr-1",
        "native-cli-qualification",
        id="merge-group",
    ),
    pytest.param("push", "refs/heads/master", "native-cli-artifact", id="master-push"),
    pytest.param("push", "refs/tags/v1.0.0", "native-cli-artifact", id="version-tag"),
    pytest.param(
        "workflow_dispatch", "refs/heads/master", "native-cli-artifact", id="manual"
    ),
    pytest.param("schedule", "refs/heads/master", "", id="schedule"),
    pytest.param("push", "refs/heads/feature", "", id="unselected-push"),
]
CLI_JOBS = {"native-cli-artifact", "native-cli-qualification"}


@pytest.mark.parametrize(("event", "ref", "selected"), EVENTS)
def test_event_selects_exactly_one_cli_branch(
    event: str, ref: str, selected: str
) -> None:
    results = {"merge_queue_liveness": "success", "cuda-compile": "success"}
    ready = {
        name for name in CLI_JOBS if _schedule(name, event, ref, results) == "ready"
    }
    assert ready == ({selected} if selected else set())


@pytest.mark.parametrize("event", ["pull_request", "merge_group"])
def test_cli_and_cuda_are_ready_together_after_liveness(event: str) -> None:
    ref = "refs/heads/gh-readonly-queue/master/pr-1"
    for name in ("cuda-compile", "native-cli-qualification"):
        assert _schedule(name, event, ref, {}) == "waiting"
        results = {"merge_queue_liveness": "success", "cuda-compile": "in_progress"}
        assert _schedule(name, event, ref, results) == "ready"
        results["cuda-compile"] = "failure"
        # A subsequent verified sibling failure is handled by the existing
        # same-attempt guard/aggregate, not a sequential scheduling dependency.
        assert _schedule(name, event, ref, results) == "ready"


@pytest.mark.parametrize(
    "result", ["in_progress", "failure", "cancelled", "skipped", "success"]
)
@pytest.mark.parametrize(
    "event,ref",
    [
        ("push", "refs/heads/master"),
        ("push", "refs/tags/v1.0.0"),
        ("workflow_dispatch", "refs/heads/master"),
    ],
)
def test_publish_branch_requires_successful_cuda_predecessor(
    event: str, ref: str, result: str
) -> None:
    actual = _schedule("native-cli-artifact", event, ref, {"cuda-compile": result})
    assert actual == (
        "ready"
        if result == "success"
        else "waiting"
        if result == "in_progress"
        else "skipped"
    )


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
@pytest.mark.parametrize("event", ["pull_request", "merge_group"])
def test_liveness_failure_cannot_start_parallel_cli(event: str, result: str) -> None:
    assert (
        _schedule(
            "native-cli-qualification",
            event,
            "refs/pull/1/merge",
            {"merge_queue_liveness": result},
        )
        == "skipped"
    )


@pytest.mark.parametrize(
    "active,expected", [("false", "skipped"), ("", "ready"), ("true", "ready")]
)
def test_only_confirmed_orphan_skips_parallel_cli(active: str, expected: str) -> None:
    assert (
        _schedule(
            "native-cli-qualification",
            "merge_group",
            "refs/heads/gh-readonly-queue/master/pr-1",
            {"merge_queue_liveness": "success"},
            active=active,
        )
        == expected
    )


@pytest.mark.parametrize(("event", "ref", "selected"), EVENTS)
def test_cancelled_workflow_does_not_start_cli(
    event: str, ref: str, selected: str
) -> None:
    results = {"merge_queue_liveness": "success", "cuda-compile": "success"}
    assert all(
        _schedule(name, event, ref, results, cancelled=True) == "skipped"
        for name in CLI_JOBS
    )


@pytest.mark.parametrize(
    "event", ["schedule", "pull_request", "merge_group", "workflow_dispatch"]
)
@pytest.mark.parametrize("result", ["success", "skipped", "failure", "cancelled"])
@pytest.mark.parametrize(
    "required_job",
    [
        "native-cli-artifact",
        "native-cli-qualification",
        "cuda-compile",
        "cuda-direct-aot",
    ],
)
def test_aggregate_only_admits_unused_event_branch_skip(
    event: str, result: str, required_job: str
) -> None:
    expected_skips = (
        CLI_JOBS
        if event == "schedule"
        else {"native-cli-artifact"}
        if event in {"pull_request", "merge_group"}
        else {"native-cli-qualification"}
    )
    assert _allowed_skips(event) == expected_skips
    results = {name: "success" for name in _dependencies("pass")}
    results.update({name: "skipped" for name in expected_skips})
    results[required_job] = result
    assert _aggregate(event, results) == (
        result == "success" or (result == "skipped" and required_job in expected_skips)
    )


@pytest.mark.parametrize("event", ["pull_request", "merge_group", "workflow_dispatch"])
def test_skipping_both_cli_branches_never_passes(event: str) -> None:
    results = {name: "success" for name in _dependencies("pass")}
    results.update({name: "skipped" for name in CLI_JOBS})
    assert not _aggregate(event, results)


@pytest.mark.parametrize("event", ["pull_request", "merge_group", "workflow_dispatch"])
def test_aggregate_runs_and_rejects_failed_or_skipped_selected_job(event: str) -> None:
    selected = (
        "native-cli-qualification"
        if event != "workflow_dispatch"
        else "native-cli-artifact"
    )
    for result in ("failure", "cancelled", "skipped"):
        results = {name: "success" for name in _dependencies("pass")}
        results.update({name: "skipped" for name in _allowed_skips(event)})
        results[selected] = result
        assert _schedule("pass", event, "refs/heads/master", results) == "ready"
        assert not _aggregate(event, results)


@pytest.mark.parametrize(("event", "ref", "selected"), EVENTS)
@pytest.mark.parametrize("publish", ["true", "false", ""])
def test_master_publication_uses_only_ordered_cuda_output(
    event: str, ref: str, selected: str, publish: str
) -> None:
    producer = _job("native-cli-artifact")
    output = re.search(r"PUBLISH_CUDA_CACHE_SNAPSHOT: (.+)", producer)
    assert output is not None
    value = _expression(
        output[1], {"needs.cuda-compile.outputs.publish_cache_snapshot": publish}
    )
    qualification = _job("native-cli-qualification")
    assert re.search(r"PUBLISH_CUDA_CACHE_SNAPSHOT: ['\"]false['\"]", qualification)
    step = producer.split(
        "      - name: Publish trusted master CUDA ccache snapshot\n", 1
    )[1].split("\n      - ", 1)[0]
    condition = re.search(r"^        if: (.+)$", step, re.MULTILINE)
    assert condition is not None
    for name in CLI_JOBS:
        enabled = _expression(
            condition[1],
            {
                "github.event_name": event,
                "github.ref": ref,
                "env.PUBLISH_CUDA_CACHE_SNAPSHOT": value
                if name == "native-cli-artifact"
                else "false",
            },
        )
        assert enabled == (
            name == "native-cli-artifact"
            and event == "push"
            and ref == "refs/heads/master"
            and publish == "true"
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Distribution targets Linux ELF")
@pytest.mark.parametrize("mismatch", [False, True])
def test_archive_verification_uses_installed_bytes(
    tmp_path: Path, mismatch: bool
) -> None:
    prefix = tmp_path / "build/native-cli-prefix"
    (prefix / "bin").mkdir(parents=True)
    installed = prefix / "bin/generativeqc"
    shutil.copyfile("/bin/true", installed)
    installed.chmod(0o755)
    build_cli = tmp_path / "build/cuda-release-cli/generativeqc"
    build_cli.parent.mkdir()
    # Model the distinct ELF bytes caused by CMake's install-time RUNPATH edit.
    shutil.copyfile("/bin/false", build_cli)
    build_cli.chmod(0o755)
    source = tmp_path / "source"
    (source / "LICENSES").mkdir(parents=True)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        (source / name).write_text("fixture\n")
    package_prefix = prefix
    if mismatch:
        package_prefix = tmp_path / "different-prefix"
        (package_prefix / "bin").mkdir(parents=True)
        shutil.copyfile(build_cli, package_prefix / "bin/generativeqc")
    package_native_cli(
        prefix=package_prefix,
        source_root=source,
        output_dir=tmp_path / "dist",
        version="0.1.0",
        commit="a" * 40,
        release=False,
        source_date_epoch=1,
    )
    step = (
        _job("native-cli-artifact")
        .split("      - name: Verify packaged executable and report byte sizes\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_SHA": "a" * 40,
            "GITHUB_STEP_SUMMARY": str(summary),
            "TMPDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if mismatch:
        assert result.returncode != 0
        assert not summary.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert str(installed.stat().st_size) in summary.read_text()

"""Probe overlays retain ref-scoped reuse without publishing shared master caches."""

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON_SHARDS = (
    "core-a",
    "core-b",
    "core-c",
    "uccsd-forces",
    "runtime-heavy",
    "posthf",
    "compiler-heavy",
    "ecp-forces",
)


def _python_job() -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    return workflow.split("\n  python:\n", 1)[1].split("\n  cpu-benchmark:\n", 1)[0]


def _steps() -> dict[str, str]:
    parts = re.split(r"^      - name: (.+)\n", _python_job(), flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _condition(step: str, values: dict[str, str]) -> bool:
    condition = re.search(r"^\s+if: (.+)$", step, flags=re.MULTILINE)
    # These probes evaluate an otherwise eligible step after successful setup.
    # A step without an explicit condition uses Actions' default success().
    if condition is None:
        return True
    expression = condition.group(1)
    expression = expression.replace("always()", "true")
    for index, source in enumerate(values):
        expression = expression.replace(source, f'"$VALUE_{index}"')
    result = subprocess.run(
        ["bash", "-c", f"if [[ {expression} ]]; then printf run; else printf skip; fi"],
        env={
            **os.environ,
            **{f"VALUE_{index}": value for index, value in enumerate(values.values())},
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout == "run"


def test_probe_overlay_paths_and_namespaces_remain_disjoint_from_master() -> None:
    steps = _steps()
    restore = steps["Restore native Python probes"]
    save = steps["Save native Python probes"]
    inspect = steps["Inspect native Python probe cache"]
    location = "${{ runner.temp }}/ccache-python-probes"
    for step in (restore, save):
        assert f"path: {location}" in step
        assert "path: ~/.cache/ccache" not in step
    key = next(line.strip() for line in restore.splitlines() if "key:" in line)
    assert key.startswith(
        "key: ccache-python-probes-v2-${{ matrix.shard }}-${{ hashFiles("
    )
    assert (
        "restore-keys: |\n            ccache-python-probes-v2-${{ matrix.shard }}-"
        in restore
    )
    assert {"tests/python/**", "tests/native/**", "benchmarks/**"} <= set(
        re.findall(r"'([^']+)'", key)
    )
    for name in (
        "Reset native probe ccache statistics",
        "Prepare bounded native diagnostics",
        "Run Python tests with coverage",
        "Inspect native Python probe cache",
        "Show post-test ccache statistics",
    ):
        assert f"CCACHE_DIR: {location}" in steps[name]
    for name in (
        "Prepare bounded native diagnostics",
        "Run Python tests with coverage",
        "Inspect native Python probe cache",
        "Show post-test ccache statistics",
    ):
        assert "CCACHE_MAXSIZE: 256M" in steps[name]
    assert "timeout-minutes: 2" in save
    assert "continue-on-error: true" in save
    assert "always()" in save
    assert "timeout-minutes: 1" in inspect
    assert "continue-on-error: true" in inspect
    assert "always()" in inspect
    assert "continue-on-error:" not in steps["Run Python tests with coverage"]
    shared = steps["Restore shared Linux CPU ccache (Python native)"]
    assert "path: ~/.cache/ccache" in shared
    assert "ccache-linux-cpu-v4-" in shared
    assert "ccache-python-probes-" not in shared
    # Neither trusted producer imports the ref-scoped overlay or its fallback.
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    for job in ("master-cpu-cache", "native-cli-artifact", "cuda-compile"):
        section = workflow.split(f"\n  {job}:\n", 1)[1]
        section = re.split(r"\n  [A-Za-z0-9_-]+:\n", section, maxsplit=1)[0]
        assert "ccache-python-probes-" not in section
        assert "python_probe_ccache" not in section


def test_every_python_matrix_member_has_probe_cache_coverage() -> None:
    matrix = re.search(r"^        shard: \[(.+)\]$", _python_job(), re.MULTILINE)
    assert matrix
    assert {part.strip() for part in matrix.group(1).split(",")} == set(PYTHON_SHARDS)


@pytest.mark.parametrize("shard", PYTHON_SHARDS)
@pytest.mark.parametrize(
    "event", ["pull_request", "merge_group", "schedule", "workflow_dispatch"]
)
@pytest.mark.parametrize(
    (
        "hit",
        "primary",
        "restore_outcome",
        "test_outcome",
        "contents_outcome",
        "has_entries",
        "can_inspect",
        "can_save",
    ),
    [
        ("false", "probe-key", "success", "success", "success", "true", True, True),
        ("false", "probe-key", "success", "failure", "success", "true", True, True),
        ("false", "probe-key", "success", "cancelled", "success", "true", True, True),
        ("true", "probe-key", "success", "success", "success", "true", False, False),
        ("", "probe-key", "success", "failure", "success", "true", True, True),
        ("false", "", "success", "success", "success", "true", False, False),
        ("false", "probe-key", "failure", "success", "success", "true", False, False),
        ("false", "probe-key", "skipped", "success", "success", "true", False, False),
        ("false", "probe-key", "success", "skipped", "success", "true", False, False),
        ("false", "probe-key", "success", "", "success", "true", False, False),
        ("false", "probe-key", "success", "success", "success", "false", True, False),
        ("false", "probe-key", "success", "failure", "failure", "true", True, False),
        ("false", "probe-key", "success", "success", "cancelled", "true", True, False),
        ("false", "probe-key", "success", "success", "skipped", "", True, False),
        ("false", "probe-key", "success", "failure", "success", "", True, False),
    ],
)
def test_probe_overlay_preserves_ref_scoping_and_complete_key_precedence(
    shard: str,
    event: str,
    hit: str,
    primary: str,
    restore_outcome: str,
    test_outcome: str,
    contents_outcome: str,
    has_entries: str,
    can_inspect: bool,
    can_save: bool,
) -> None:
    steps = _steps()
    assert _condition(steps["Restore native Python probes"], {"matrix.shard": shard})
    values = {
        "github.event_name": event,
        "steps.python_probe_ccache.outcome": restore_outcome,
        "steps.python_probe_ccache.outputs.cache-hit": hit,
        "steps.python_probe_ccache.outputs.cache-primary-key": primary,
        "steps.python_tests.outcome": test_outcome,
        "steps.python_probe_contents.outcome": contents_outcome,
        "steps.python_probe_contents.outputs.has-entries": has_entries,
    }
    assert _condition(steps["Inspect native Python probe cache"], values) == (
        event != "merge_group" and can_inspect
    )
    assert _condition(steps["Save native Python probes"], values) == (
        event != "merge_group" and can_save
    )


@pytest.mark.parametrize(
    ("event", "ref", "eligible"),
    [
        ("pull_request", "refs/pull/42/merge", True),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-42", True),
        ("schedule", "refs/heads/master", True),
        ("workflow_dispatch", "refs/heads/master", True),
        ("workflow_dispatch", "refs/heads/feature", True),
        ("workflow_dispatch", "refs/tags/v1.0", True),
        ("push", "refs/heads/master", False),
        ("push", "refs/heads/feature", False),
        ("push", "refs/tags/v1.0", False),
    ],
)
@pytest.mark.parametrize("active", ["true", "false", ""])
def test_probe_expansion_does_not_start_python_for_pushes_or_orphaned_queues(
    event: str, ref: str, eligible: bool, active: str
) -> None:
    header = _python_job().split("\n    steps:\n", 1)[0]
    assert _condition(
        header,
        {
            "github.event_name": event,
            "github.ref": ref,
            "needs.merge_queue_liveness.outputs.active": active,
        },
    ) == (eligible and active != "false")


@pytest.mark.parametrize(
    ("stats", "status", "succeeds", "output"),
    [
        ('{"files_in_cache": 0}', 0, True, "has-entries=false\n"),
        ('{"files_in_cache": 2}', 0, True, "has-entries=true\n"),
        # Resetting call counters must not hide entries restored from a prefix.
        ('{"files_in_cache": 2, "cache_miss": 0}', 0, True, "has-entries=true\n"),
        ("{}", 0, False, ""),
        ("malformed", 0, False, ""),
        ('{"files_in_cache": -1}', 0, False, ""),
        ('{"files_in_cache": "2"}', 0, False, ""),
        ('{"files_in_cache": true}', 0, False, ""),
        ('{"files_in_cache": null}', 0, False, ""),
        ('{"files_in_cache": 2.5}', 0, False, ""),
        ("", 1, False, ""),
        # A failed producer can emit valid bytes; its outcome still blocks save.
        ('{"files_in_cache": 2}', 1, False, "has-entries=true\n"),
    ],
)
def test_probe_inventory_rejects_empty_or_unreliable_statistics(
    tmp_path: Path, stats: str, status: int, succeeds: bool, output: str
) -> None:
    inspect = _steps()["Inspect native Python probe cache"]
    script = textwrap.dedent(inspect.split("        run: |\n", 1)[1])
    shell = re.search(r"^        shell: (.+)$", inspect, re.MULTILINE)
    assert shell is None or shell.group(1) == "bash"
    # Actions' unspecified Linux shell lacks pipefail; explicit bash has it.
    # Mirror the declared runner shell so a lost pipefail fails the error case.
    command = (
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail"]
        if shell
        else ["bash", "-e"]
    )
    cache = tmp_path / "ccache"
    cache.write_text(
        "#!/bin/sh\n"
        'test "$1" = --print-stats && test "$2" = --format=json || exit 2\n'
        'printf "%s" "$CACHE_STATS"\n'
        'exit "$CACHE_STATUS"\n'
    )
    cache.chmod(0o755)
    outputs = tmp_path / "outputs"
    result = subprocess.run(
        [*command, "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CACHE_STATS": stats,
            "CACHE_STATUS": str(status),
            "GITHUB_OUTPUT": str(outputs),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == succeeds
    assert outputs.read_text() == output


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_partial_probe_keys_cannot_freeze_the_complete_snapshot(outcome: str) -> None:
    save = _steps()["Save native Python probes"]
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
    # The unchanged exact key remains the first lookup; partials are recoverable
    # only via the shard prefix and cannot overwrite a complete cache generation.
    assert actual.startswith(primary)

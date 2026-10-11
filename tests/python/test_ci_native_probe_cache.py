"""Probe overlays retain ref-scoped reuse without publishing shared master caches."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _steps() -> dict[str, str]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    python = workflow.split("\n  python:\n", 1)[1].split("\n  cpu-benchmark:\n", 1)[0]
    parts = re.split(r"^      - name: (.+)\n", python, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _condition(step: str, values: dict[str, str]) -> bool:
    expression = step.split("        if: ", 1)[1].splitlines()[0]
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
        "Show post-test ccache statistics",
    ):
        assert f"CCACHE_DIR: {location}" in steps[name]
    for name in (
        "Prepare bounded native diagnostics",
        "Run Python tests with coverage",
        "Show post-test ccache statistics",
    ):
        assert "CCACHE_MAXSIZE: 256M" in steps[name]
    assert "timeout-minutes: 2" in save
    assert "continue-on-error: true" in save
    assert "always()" in save
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


@pytest.mark.parametrize(
    "shard",
    [
        "core-a",
        "core-b",
        "core-c",
        "runtime-heavy",
        "compiler-heavy",
        "posthf",
        "ecp-forces",
    ],
)
@pytest.mark.parametrize(
    "event", ["pull_request", "merge_group", "schedule", "workflow_dispatch"]
)
@pytest.mark.parametrize(
    ("hit", "primary", "restore_outcome", "test_outcome", "can_save"),
    [
        ("false", "probe-key", "success", "success", True),
        ("false", "probe-key", "success", "failure", True),
        ("false", "probe-key", "success", "cancelled", True),
        ("true", "probe-key", "success", "success", False),
        ("", "probe-key", "success", "failure", True),
        ("false", "", "success", "success", False),
        ("false", "probe-key", "failure", "success", False),
        ("false", "probe-key", "success", "skipped", False),
        ("false", "probe-key", "success", "", False),
    ],
)
def test_probe_overlay_preserves_ref_scoping_and_complete_key_precedence(
    shard: str,
    event: str,
    hit: str,
    primary: str,
    restore_outcome: str,
    test_outcome: str,
    can_save: bool,
) -> None:
    steps = _steps()
    consumes = shard in {
        "core-a",
        "core-b",
        "core-c",
        "runtime-heavy",
        "compiler-heavy",
    }
    assert (
        _condition(steps["Restore native Python probes"], {"matrix.shard": shard})
        == consumes
    )
    outcome = restore_outcome if consumes else "skipped"
    assert _condition(
        steps["Save native Python probes"],
        {
            "github.event_name": event,
            "steps.python_probe_ccache.outcome": outcome,
            "steps.python_probe_ccache.outputs.cache-hit": hit,
            "steps.python_probe_ccache.outputs.cache-primary-key": primary,
            "steps.python_tests.outcome": test_outcome,
        },
    ) == (consumes and event != "merge_group" and can_save)


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

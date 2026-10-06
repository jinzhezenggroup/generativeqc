"""Probe caches stay content-addressed, shard-scoped, and read-only in queues."""

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
    steps = dict(zip(parts[1::2], parts[2::2], strict=True))
    return steps


def test_probe_cache_is_separate_per_shard_and_saved_after_success() -> None:
    steps = _steps()
    names = list(steps)
    cache = steps["Restore native Python probes"]
    save = steps["Save native Python probes"]
    reset = steps["Reset native probe ccache statistics"]
    tests = steps["Run Python tests with coverage"]
    statistics = steps["Show post-test ccache statistics"]
    common = steps["Save Python-build ccache immediately"]

    assert (
        names.index("Save Python-build ccache immediately")
        < names.index("Restore native Python probes")
        < names.index("Reset native probe ccache statistics")
        < names.index("Run Python tests with coverage")
        < names.index("Save native Python probes")
        < names.index("Show post-test ccache statistics")
    )
    # Explicit saves follow successful tests; default success() gating must stay.
    assert "uses: actions/cache/restore@" in cache
    assert "id: python_probe_ccache" in cache
    assert "uses: actions/cache/save@" in save
    assert "always()" not in save and "failure()" not in save
    assert "key: ${{ steps.python_probe_ccache.outputs.cache-primary-key }}" in save
    key = next(line.strip() for line in cache.splitlines() if "key:" in line)
    assert key.startswith(
        "key: ccache-python-probes-v2-${{ matrix.shard }}-${{ hashFiles("
    )
    assert "github.sha" not in key
    assert "github.run_id" not in key
    probe_inputs = set(re.findall(r"'([^']+)'", key))
    build_key = next(
        line for line in steps["Restore ccache"].splitlines() if "key:" in line
    )
    assert set(re.findall(r"'([^']+)'", build_key)) <= probe_inputs
    assert {"tests/python/**", "tests/native/**", "benchmarks/**"} <= probe_inputs
    assert (
        "restore-keys: |\n            ccache-python-probes-v2-${{ matrix.shard }}-"
        in cache
    )
    location = "${{ runner.temp }}/ccache-python-probes"
    assert f"path: {location}" in cache
    assert f"path: {location}" in save
    assert f"CCACHE_DIR: {location}" in reset
    assert "run: ccache --zero-stats" in reset
    for step in (tests, statistics):
        assert f"CCACHE_DIR: {location}" in step
        assert "CCACHE_MAXSIZE: 256M" in step
    assert "path: ~/.cache/ccache" in common
    assert "key: ${{ steps.python_ccache.outputs.cache-primary-key }}" in common
    assert "uses: actions/cache/save@" in common
    assert "if: always()" in statistics


@pytest.mark.parametrize(
    "shard",
    ["core-a", "core-b", "compiler-heavy", "runtime-heavy", "posthf", "ecp-forces"],
)
@pytest.mark.parametrize("event", ["pull_request", "merge_group", "schedule"])
@pytest.mark.parametrize("cache_hit", ["true", "false", ""])
@pytest.mark.parametrize("restore_succeeds", [True, False])
def test_probe_cache_guards_preserve_queue_read_only_consumption(
    shard: str, event: str, cache_hit: str, restore_succeeds: bool
) -> None:
    steps = _steps()
    consumes = shard in {"core-a", "core-b", "compiler-heavy", "runtime-heavy"}
    outcome = ("success" if restore_succeeds else "failure") if consumes else "skipped"
    for name, expected in (
        ("Restore native Python probes", consumes),
        (
            "Save native Python probes",
            consumes
            and restore_succeeds
            and event != "merge_group"
            and cache_hit != "true",
        ),
    ):
        expression = steps[name].split("        if: ", 1)[1].splitlines()[0]
        for source, variable in (
            ("matrix.shard", "SHARD"),
            ("github.event_name", "EVENT"),
            ("steps.python_probe_ccache.outcome", "OUTCOME"),
            ("steps.python_probe_ccache.outputs.cache-hit", "CACHE_HIT"),
        ):
            expression = expression.replace(source, f'"${variable}"')
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"if [[ {expression} ]]; then printf run; else printf skip; fi",
            ],
            env={
                **os.environ,
                "SHARD": shard,
                "EVENT": event,
                "OUTCOME": outcome,
                "CACHE_HIT": cache_hit,
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout == ("run" if expected else "skip")

"""Probe cache persistence must include every shard without replacing build cache."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_probe_cache_is_separate_per_shard_and_saved_after_success() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    python = workflow.split("\n  python:\n", 1)[1].split("\n  cpu-benchmark:\n", 1)[0]
    parts = re.split(r"^      - name: (.+)\n", python, flags=re.MULTILINE)
    steps = dict(zip(parts[1::2], parts[2::2], strict=True))
    names = list(steps)
    cache = steps["Cache native Python probes"]
    reset = steps["Reset native probe ccache statistics"]
    tests = steps["Run Python tests with coverage"]
    statistics = steps["Show post-test ccache statistics"]
    common = steps["Save Python-build ccache immediately"]

    assert (
        names.index("Save Python-build ccache immediately")
        < names.index("Cache native Python probes")
        < names.index("Reset native probe ccache statistics")
        < names.index("Run Python tests with coverage")
        < names.index("Show post-test ccache statistics")
    )
    # The combined cache action registers its save as a success-only post step.
    assert "uses: actions/cache@" in cache
    assert "        if:" not in cache
    assert "key: ccache-python-probes-v1-${{ matrix.shard }}-${{ github.sha }}" in cache
    assert (
        "restore-keys: |\n            ccache-python-probes-v1-${{ matrix.shard }}-"
        in cache
    )
    location = "${{ runner.temp }}/ccache-python-probes"
    assert f"path: {location}" in cache
    assert f"CCACHE_DIR: {location}" in reset
    assert "run: ccache --zero-stats" in reset
    for step in (tests, statistics):
        assert f"CCACHE_DIR: {location}" in step
        assert "CCACHE_MAXSIZE: 256M" in step
    assert "path: ~/.cache/ccache" in common
    assert "key: ${{ steps.python_ccache.outputs.cache-primary-key }}" in common
    assert "uses: actions/cache/save@" in common
    assert "if: always()" in statistics

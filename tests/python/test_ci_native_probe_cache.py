"""Native Python probes consume the same local CPU cache without publishing."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _steps() -> dict[str, str]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    python = workflow.split("\n  python:\n", 1)[1].split("\n  cpu-benchmark:\n", 1)[0]
    parts = re.split(r"^      - name: (.+)\n", python, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_probes_reuse_restored_cpu_cache_without_a_second_snapshot() -> None:
    steps = _steps()
    restores = [
        (name, step)
        for name, step in steps.items()
        if "uses: actions/cache/restore@" in step and "ccache" in step
    ]
    assert len(restores) == 1
    restore_name, restore = restores[0]
    assert "path: ~/.cache/ccache" in restore
    assert "ccache-linux-cpu-v4-ubuntu24-x86_64-" in restore
    assert not any(
        "uses: actions/cache/save@" in step and "ccache" in step
        for step in steps.values()
    )
    names = list(steps)
    assert (
        names.index(restore_name)
        < names.index("Reset native probe ccache statistics")
        < names.index("Run Python tests with coverage")
        < names.index("Show post-test ccache statistics")
    )
    for name in (
        "Reset native probe ccache statistics",
        "Prepare bounded native diagnostics",
        "Run Python tests with coverage",
        "Show post-test ccache statistics",
    ):
        assert "CCACHE_DIR:" not in steps[name]
        assert "CCACHE_MAXSIZE:" not in steps[name]
    statistics = steps["Show post-test ccache statistics"]
    assert "if: always()" in statistics
    assert "ccache --show-stats" in statistics
    assert "continue-on-error:" not in steps["Run Python tests with coverage"]

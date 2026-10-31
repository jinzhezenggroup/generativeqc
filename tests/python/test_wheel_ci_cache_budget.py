"""Keep cold wheel qualification bounded without losing valid compiler work."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/wheels.yml").read_text()


def _step(name: str) -> str:
    marker = f"      - name: {name}\n"
    start = WORKFLOW.index(marker)
    following = re.search(r"\n      - (?:name|uses):", WORKFLOW[start + len(marker) :])
    end = (
        len(WORKFLOW) if following is None else start + len(marker) + following.start()
    )
    return WORKFLOW[start:end]


def _timeout(section: str) -> int:
    match = re.search(r"timeout-minutes: (\d+)", section)
    assert match is not None
    return int(match.group(1))


def test_cold_wheel_build_reserves_bounded_cache_save_time() -> None:
    job_header = WORKFLOW.split("  build_wheels:\n", 1)[1].split("    steps:\n", 1)[0]
    build = _step("Build and test wheel")
    save = _step("Save completed compiler cache entries")
    assert _timeout(build) == 330
    assert _timeout(job_header) == 360
    assert _timeout(save) == 10
    assert _timeout(build) + _timeout(save) < _timeout(job_header)
    assert _timeout(job_header) - _timeout(build) >= 30
    assert "continue-on-error" not in build
    assert "continue-on-error" not in job_header


def test_compiler_snapshots_refresh_without_changing_cache_scope() -> None:
    restore = _step("Restore ccache")
    assert "id: wheel_ccache" in restore
    assert (
        "uses: actions/cache/restore@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"
        in restore
    )
    key = re.search(r"^          key: (.+)$", restore, re.MULTILINE)
    assert key is not None
    suffix = "-${{ github.run_id }}-${{ github.run_attempt }}"
    assert key.group(1).endswith(suffix)
    content_key = key.group(1).removesuffix(suffix)
    prefixes = restore.split("          restore-keys: |\n", 1)[1].splitlines()
    assert [line.strip() for line in prefixes if line.strip()] == [
        content_key,
        "ccache-wheel-v3-manylinux_x86_64-",
        "ccache-wheel-v2-manylinux_x86_64-",
    ]
    assert "path: ${{ runner.temp }}/ccache" in restore
    assert "CCACHE_MAXSIZE: 3G" in WORKFLOW


def test_compiler_cache_saves_partial_progress_without_masking_failure() -> None:
    save = _step("Save completed compiler cache entries")
    assert (
        "if: ${{ always() && steps.wheel_ccache.outputs.cache-primary-key != '' }}"
        in save
    )
    assert "uses: actions/cache/save@55cc8345863c7cc4c66a329aec7e433d2d1c52a9" in save
    assert "key: ${{ steps.wheel_ccache.outputs.cache-primary-key }}" in save
    assert "path: ${{ runner.temp }}/ccache" in save
    assert (
        WORKFLOW.index("name: Build and test wheel")
        < WORKFLOW.index("name: Save completed compiler cache entries")
        < WORKFLOW.index("name: Verify wheel payload and CUDA linkage")
    )
    # Keep the required wheel artifact separate from the single optional,
    # master-only snapshot. Additional uploads must be explicitly reviewed.
    assert WORKFLOW.count("actions/upload-artifact@") == 2
    wheel_artifact = WORKFLOW.split("      - uses: actions/upload-artifact@", 1)[
        1
    ].split("\n      - name:", 1)[0]
    assert "name: generativeqc-wheel-manylinux_x86_64" in wheel_artifact
    assert "path: wheelhouse/*.whl" in wheel_artifact
    assert "if-no-files-found: error" in wheel_artifact
    assert "continue-on-error" not in wheel_artifact
    snapshot = _step("Publish bounded wheel compiler snapshot")
    assert "if: steps.wheel_snapshot.outcome == 'success'" in snapshot
    assert "continue-on-error: true" in snapshot
    assert "retention-days: 1" in snapshot
    assert _timeout(snapshot) == 2
    package = _step("Package bounded wheel compiler snapshot")
    assert "github.repository == 'jinzhezenggroup/generativeqc'" in package
    assert "github.event_name == 'push'" in package
    assert "github.ref == 'refs/heads/master'" in package
    assert "steps.wheel_build.outcome == 'success'" in package
    assert "steps.wheel_payload.outcome == 'success'" in package
    assert "permissions:\n  contents: read\n" in WORKFLOW
    assert "pull_request_target:" not in WORKFLOW
    assert "cancel-in-progress: false" in WORKFLOW


def test_retries_and_new_runs_can_save_a_more_complete_snapshot() -> None:
    restore = _step("Restore ccache")
    key = re.search(r"^          key: (.+)$", restore, re.MULTILINE)
    assert key is not None
    template = re.sub(r"\$\{\{ hashFiles\(.*?\) \}\}", "source-digest", key.group(1))

    def snapshot(run_id: int, attempt: int) -> str:
        return template.replace("${{ github.run_id }}", str(run_id)).replace(
            "${{ github.run_attempt }}", str(attempt)
        )

    keys = {snapshot(71, 1), snapshot(71, 2), snapshot(72, 1)}
    assert len(keys) == 3
    assert all(
        key.startswith("ccache-wheel-v3-manylinux_x86_64-source-digest-")
        for key in keys
    )


def test_tool_environment_cache_remains_success_only() -> None:
    tooling = _step("Restore cibuildwheel tool cache")
    assert "uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9" in tooling
    assert "save-always" not in tooling
    assert "always()" not in tooling
    assert WORKFLOW.count("actions/cache/save@") == 1
    assert "if:" not in _step("Verify wheel payload and CUDA linkage")
    aggregate = WORKFLOW.split("\n  pass:\n", 1)[1]
    assert "needs: [build_wheels]" in aggregate
    assert "if: always()" in aggregate

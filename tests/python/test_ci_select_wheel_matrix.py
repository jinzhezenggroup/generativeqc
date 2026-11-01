"""CI contracts for fast wheel PRs and protected full-matrix qualification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import ci_select_wheel_matrix as matrix

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("event", "paths", "expected"),
    [
        ("pull_request", ["src/cli/wheel_launcher.cpp"], matrix.FAST_ARCHITECTURES),
        (
            "pull_request",
            ["python/generativeqc/tune_cli.py"],
            matrix.FAST_ARCHITECTURES,
        ),
        ("pull_request", ["docs/developer/build.md"], matrix.FAST_ARCHITECTURES),
        ("pull_request", [".github/workflows/wheels.yml"], matrix.FULL_ARCHITECTURES),
        ("pull_request", ["CMakeLists.txt"], matrix.FULL_ARCHITECTURES),
        ("pull_request", ["pyproject.toml"], matrix.FULL_ARCHITECTURES),
        ("pull_request", ["cmake/cuda_wheel_payload.ld"], matrix.FULL_ARCHITECTURES),
        (
            "pull_request",
            ["cmake/3rdparty/implib/source.cpp"],
            matrix.FULL_ARCHITECTURES,
        ),
        (
            "pull_request",
            ["src/runtime/nvidia_host_api/provider.cu"],
            matrix.FULL_ARCHITECTURES,
        ),
        (
            "pull_request",
            ["tools/ci_select_wheel_matrix.py"],
            matrix.FULL_ARCHITECTURES,
        ),
        (
            "pull_request",
            ["src/cli/wheel_launcher.cpp", "python/ci/repair-wheel.sh"],
            matrix.FULL_ARCHITECTURES,
        ),
        ("push", [], matrix.FULL_ARCHITECTURES),
        ("schedule", [], matrix.FULL_ARCHITECTURES),
        ("workflow_dispatch", [], matrix.FULL_ARCHITECTURES),
    ],
)
def test_matrix_selection(event: str, paths: list[str], expected: str) -> None:
    assert matrix.select_architectures(event, paths) == expected


def test_other_event_is_not_silently_treated_as_fast() -> None:
    with pytest.raises(ValueError, match="unsupported wheel event"):
        matrix.select_architectures("merge_group")


def test_missing_pr_shas_fail_closed() -> None:
    with pytest.raises(ValueError, match="PR base/head SHA"):
        matrix.pull_request_paths("", "a" * 40)
    with pytest.raises(ValueError, match="PR base/head SHA"):
        matrix.pull_request_paths("b" * 40, "unknown")


def test_actual_diff_controls_the_selected_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "matrix-output"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GQC_PR_BASE_SHA", "a" * 40)
    monkeypatch.setenv("GQC_PR_HEAD_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", os.fspath(output_path))

    def fake_diff(argv: list[str]) -> bytes:
        assert argv == [
            "git",
            "diff",
            "--no-ext-diff",
            "--name-only",
            "-z",
            "a" * 40,
            "b" * 40,
        ]
        return b"src/cli/wheel_launcher.cpp\0.github/workflows/wheels.yml\0"

    monkeypatch.setattr(matrix.subprocess, "check_output", fake_diff)
    matrix.main()
    assert output_path.read_text() == ("architectures=80;90-real;120-real\nmode=full\n")


def test_wheel_workflow_keeps_release_and_pr_receipts_separate() -> None:
    assert "id: wheel_matrix" in WORKFLOW
    assert "run: python3 tools/ci_select_wheel_matrix.py" in WORKFLOW
    assert r"GQC_PR_BASE_SHA: ${{ github.event.pull_request.base.sha }}" in WORKFLOW
    assert r"GQC_PR_HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in WORKFLOW
    assert (
        'GENERATIVEQC_CUDA_ARCHITECTURES="'
        r'${{ steps.wheel_matrix.outputs.architectures }}"'
    ) in WORKFLOW
    assert (
        r"GQC_WHEEL_SNAPSHOT_ID=${{ github.event_name == 'push' && "
        "github.ref == 'refs/heads/master' && "
        "format('{0}-{1}', github.run_id, github.run_attempt) || '' }}"
    ) in WORKFLOW
    assert "pull_request_target:" not in WORKFLOW
    assert "cancel-in-progress: false" in WORKFLOW

"""Exercise hosted disk reclamation without deleting any real runner files."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from test_ci_prose_workflow import assert_full_coverage_step, job, selected, steps

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/ci_cuda_disk_space.sh"
STEP = "Reclaim hosted CUDA build disk space"


@pytest.mark.parametrize(
    ("actions", "runner", "reclaim"),
    [
        ("true", "github-hosted", True),
        ("true", "self-hosted", False),
        ("false", "github-hosted", False),
        ("", "github-hosted", False),
        ("true", "", False),
        ("", "", False),
    ],
)
def test_cleanup_is_hosted_only_and_never_targets_compiler_caches(
    tmp_path: Path, actions: str, runner: str, reclaim: bool
) -> None:
    """Fake privileged commands expose the deletion allowlist without running it."""
    commands = tmp_path / "commands"
    commands.mkdir()
    record = tmp_path / "sudo-arguments"
    for name, body in (
        ("sudo", 'printf "%s\\n" "$@" > "$DISK_COMMAND_RECORD"'),
        ("df", "exit 0"),
    ):
        executable = commands / name
        executable.write_text("#!/bin/sh\n" + body + "\n")
        executable.chmod(0o755)
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "GITHUB_ACTIONS": actions,
            "RUNNER_ENVIRONMENT": runner,
            "GITHUB_WORKSPACE": str(workspace),
            "DISK_COMMAND_RECORD": str(record),
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    if reclaim:
        assert record.read_text().splitlines() == [
            "rm",
            "-rf",
            "--",
            "/usr/local/lib/android",
            "/usr/share/dotnet",
            "/opt/ghc",
        ]
    else:
        assert not record.exists()
        assert "limited to GitHub-hosted runners" in result.stdout


@pytest.mark.parametrize("owner", ["cuda-compile", "cuda-resources", "cuda-direct-aot"])
def test_cuda_cleanup_keeps_fail_open_coverage_and_precedes_large_downloads(
    owner: str,
) -> None:
    """A failed prose classifier must retain cleanup and the full build."""
    body = steps(owner)[STEP]
    assert_full_coverage_step(body)
    section = job(owner)
    assert section.index(STEP) < section.index("Add NVIDIA CUDA repository")
    assert "bash tools/ci_cuda_disk_space.sh" in body


def test_cli_cleanup_is_shared_by_package_qualification_and_release() -> None:
    """The existing shared package-step owner covers every monolithic build."""
    section = job("native-cli-artifact")
    body = section.split(f"      - name: {STEP}\n", 1)[1].split("\n      - ", 1)[0]
    for tier in ("skip", "fast", "release"):
        assert selected(body, {"steps.cli_scope.outputs.tier": tier}) == (
            tier != "skip"
        )
    assert "bash tools/ci_cuda_disk_space.sh" in body
    assert section.index(STEP) < section.index("Add NVIDIA CUDA repository")

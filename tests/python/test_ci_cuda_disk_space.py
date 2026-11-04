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
            "timeout",
            "--kill-after=5s",
            "120s",
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


@pytest.mark.parametrize(
    ("available", "probe_status", "cleanup_status", "expected_status", "skip"),
    [
        (90 << 30, 0, 0, 0, True),
        (64 << 30, 0, 0, 0, True),
        ((64 << 30) - 1, 0, 0, 0, False),
        (32 << 30, 0, 124, 0, False),
        (32 << 30, 0, 137, 0, False),
        (32 << 30, 0, 1, 1, False),
        ("unavailable", 0, 0, 0, False),
        ("unavailable", 1, 0, 0, False),
        (90 << 30, 1, 0, 0, False),
    ],
)
def test_cleanup_skips_spacious_hosts_and_bounds_optional_deletion(
    tmp_path: Path,
    available: int | str,
    probe_status: int,
    cleanup_status: int,
    expected_status: int,
    skip: bool,
) -> None:
    """Exercise real shell decisions while privileged deletion remains a fake command."""
    commands = tmp_path / "commands"
    commands.mkdir()
    record = tmp_path / "timeout-arguments"
    for name, body in (
        (
            "df",
            (
                'printf "Available\\n%s\\n" "$DISK_AVAILABLE_BYTES"; '
                'if [ "$1" = "-B1" ]; then exit "$DISK_PROBE_STATUS"; fi'
            ),
        ),
        (
            "sudo",
            'printf "%s\\n" "$@" > "$DISK_TIMEOUT_RECORD"; exit "$DISK_TIMEOUT_STATUS"',
        ),
    ):
        executable = commands / name
        executable.write_text("#!/bin/sh\n" + body + "\n")
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "GITHUB_WORKSPACE": str(tmp_path),
            "DISK_AVAILABLE_BYTES": str(available),
            "DISK_PROBE_STATUS": str(probe_status),
            "DISK_TIMEOUT_RECORD": str(record),
            "DISK_TIMEOUT_STATUS": str(cleanup_status),
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == expected_status, result.stderr
    if skip:
        assert not record.exists()
        assert "Skipping optional toolchain cleanup" in result.stdout
    else:
        assert record.read_text().splitlines()[:3] == [
            "timeout",
            "--kill-after=5s",
            "120s",
        ]
        if cleanup_status in (124, 137):
            assert "::warning::" in result.stdout


# Keep these APT workflow contracts in an existing test module: creating a new
# test_*.py file would shift the selector's modulo-three core-file ownership
# and perturb the very Python CI timings this patch is meant to improve.
NATIVE_APT_ACTION = ROOT / ".github/actions/install-native-cli-deps/action.yml"


def test_native_cli_dependency_action_preserves_default_apt_refresh() -> None:
    action = NATIVE_APT_ACTION.read_text(encoding="utf-8")
    assert "apt-index-refreshed:" in action
    assert "default: 'false'" in action
    assert "if: inputs.apt-index-refreshed != 'true'" in action
    assert "sudo apt-get update -o Acquire::Retries=3" in action
    assert (
        "sudo apt-get install --yes nlohmann-json3-dev libssl-dev libarchive-dev"
        in action
    )
    assert action.index("- name: Refresh APT indices when required") < action.index(
        "- name: Install native CLI libraries"
    )


@pytest.mark.parametrize("owner", ("master-cpu-cache", "cpu", "python"))
def test_cpu_and_python_installers_still_refresh_apt(owner: str) -> None:
    native = steps(owner)["Install native CLI dependencies"]
    assert "uses: ./.github/actions/install-native-cli-deps" in native
    assert "apt-index-refreshed:" not in native


@pytest.mark.parametrize("owner", ("cuda-compile", "cuda-resources", "cuda-direct-aot"))
def test_cuda_installers_reuse_successfully_refreshed_apt_index(owner: str) -> None:
    section = job(owner)
    native = steps(owner)["Install native CLI dependencies"]
    assert "uses: ./.github/actions/install-native-cli-deps" in native
    assert "apt-index-refreshed: 'true'" in native
    repo_step = section.index("- name: Add NVIDIA CUDA repository")
    cuda_step = section.index("- name: Install CUDA 12.9 build dependencies")
    native_step = section.index("- name: Install native CLI dependencies")
    assert repo_step < cuda_step < native_step
    assert "sudo apt-get update" in section[repo_step:cuda_step]
    assert "sudo apt-get install --yes" in section[cuda_step:native_step]

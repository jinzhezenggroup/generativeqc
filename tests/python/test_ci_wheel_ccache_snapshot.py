"""Execute the wheel-only profile through the shared trusted snapshot boundary."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

# These fixture helpers use only standard-library parsing/archives; neither the
# workflow nor its tests add a PyYAML dependency to hosted Python shards.
from test_ci_cpu_ccache_snapshot import (
    DATA_NAME,
    REPO,
    fake_api,
    fields,
    outputs,
    replace_archive,
    steps,
)
from test_ci_cpu_ccache_snapshot import (
    environment as base_environment,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/ci_cpu_ccache_snapshot.py"


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    return base_environment.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def wheel(environment: dict[str, str]) -> Any:
    spec = importlib.util.spec_from_file_location("wheel_snapshot", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.select_profile("wheel")
    return module


def command(name: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), name, "--profile", "wheel"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def run_and_artifact(wheel: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    run = {
        "id": 456,
        "run_attempt": 3,
        "head_sha": "b" * 40,
        "path": wheel.WORKFLOW,
        "event": "push",
        "head_branch": "master",
        "status": "completed",
        "conclusion": "success",
        "repository": {"id": 9, "full_name": REPO},
        "head_repository": {"id": 9, "full_name": REPO},
    }
    artifact = {
        "id": 789,
        "name": f"{wheel.ARTIFACT}-3",
        "expired": False,
        "size_in_bytes": 1000,
        "digest": "sha256:" + "c" * 64,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow_run": {
            "id": 456,
            "head_sha": "b" * 40,
            "head_branch": "master",
            "repository_id": 9,
            "head_repository_id": 9,
        },
    }
    return run, artifact


def stage(
    wheel: Any, environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    path = Path(environment["RUNNER_TEMP"]) / "gqc-wheel-123-2-fixture"
    (path / "download").mkdir(parents=True)
    expected = wheel.identity() | {
        "run_id": "456",
        "run_attempt": "3",
        "source_sha": "b" * 40,
    }
    (path / "expected.json").write_text(json.dumps(expected))
    environment["WHEEL_CACHE_STAGE"] = str(path)
    monkeypatch.setenv("WHEEL_CACHE_STAGE", str(path))
    replace_archive(path, [(DATA_NAME, tarfile.REGTYPE, b"cached wheel object")])
    return path


def seed(
    wheel: Any, environment: dict[str, str], data: bytes = b"cache object"
) -> Path:
    target = wheel.cache_root()
    entry = target / DATA_NAME
    entry.parent.mkdir(parents=True)
    entry.write_bytes(data)
    target.with_name("ccache-wheel-toolchain.json").write_text(
        json.dumps(wheel.wheel_toolchain() | {"attempt": "123-2"})
    )
    return target


def test_wheel_profile_caps_are_separate_from_cpu(wheel: Any) -> None:
    assert wheel.MAX_ARTIFACT == 2 * 1024**3
    assert wheel.MAX_ARCHIVE == 2 * 1024**3 - 1024**2
    assert wheel.MAX_DECODED == 3 * 1024**3 and wheel.MAX_FILES == 100_000
    assert wheel.identity()["ccache_version"] == "4.14"
    assert wheel.identity()["cuda_arch"] == "120"
    assert wheel.cache_root() == Path(os.environ["RUNNER_TEMP"]) / "ccache"
    assert wheel.cache_root() != Path.home() / ".cache/ccache"


def test_discovery_uses_wheel_workflow_and_temp_not_home(
    wheel: Any,
    environment: dict[str, str],
    tmp_path: Path,
) -> None:
    home_cache = Path(environment["HOME"]) / ".cache/ccache"
    home_cache.mkdir(parents=True)
    (home_cache / "ccache.conf").write_text("remote_storage = poison\n")
    run, artifact = run_and_artifact(wheel)
    fake_api(tmp_path, environment, run, artifact)
    result = command("discover", environment)
    assert result.returncode == 0, result.stderr
    selected = outputs(environment)
    assert selected["artifact_id"] == "789"
    assert Path(selected["stage"]).parent == Path(environment["RUNNER_TEMP"])
    assert "/actions/workflows/wheels.yml/runs" in (tmp_path / "calls").read_text()
    assert not wheel.cache_root().exists()
    assert (home_cache / "ccache.conf").read_text() == "remote_storage = poison\n"


@pytest.mark.parametrize(
    "key,bad",
    [
        ("path", ".github/workflows/ci.yml"),
        ("event", "pull_request"),
        ("event", "merge_group"),
        ("event", "schedule"),
        ("head_branch", "topic"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("head_repository", {"id": 10, "full_name": REPO}),
        ("head_sha", "wrong"),
        ("run_attempt", 0),
    ],
)
def test_wheel_rejects_untrusted_producer(
    wheel: Any,
    environment: dict[str, str],
    tmp_path: Path,
    key: str,
    bad: Any,
) -> None:
    run, artifact = run_and_artifact(wheel)
    run[key] = bad
    fake_api(tmp_path, environment, run, artifact)
    assert command("discover", environment).returncode == 0
    assert not outputs(environment)
    assert "/artifacts" not in (tmp_path / "calls").read_text()


@pytest.mark.parametrize(
    "key,bad",
    [
        ("name", "cpu-ccache-master-ubuntu24-x86_64-v1-3"),
        ("name", "wheel-ccache-master-manylinux-cuda12.9-sm120-v1-2"),
        ("expired", True),
        ("size_in_bytes", 2 * 1024**3 + 1),
        ("digest", ""),
        ("digest", "sha256:bad"),
    ],
)
def test_wheel_rejects_untrusted_or_oversize_artifact(
    wheel: Any,
    environment: dict[str, str],
    tmp_path: Path,
    key: str,
    bad: Any,
) -> None:
    run, artifact = run_and_artifact(wheel)
    artifact[key] = bad
    fake_api(tmp_path, environment, run, artifact)
    assert command("discover", environment).returncode == 0
    assert not outputs(environment)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("GITHUB_REPOSITORY", "fork/generativeqc"),
        ("RUNNER_ARCH", "ARM64"),
        ("ImageOS", "ubuntu22"),
        ("RUNNER_TEMP", "relative"),
    ],
)
def test_unsupported_origin_does_not_query_or_publish(
    wheel: Any,
    environment: dict[str, str],
    tmp_path: Path,
    key: str,
    bad: str,
) -> None:
    environment[key] = bad
    result = command("discover", environment)
    assert result.returncode != 0
    assert not outputs(environment)
    assert not (tmp_path / "calls").exists()


def test_bounded_wheel_roundtrip_preserves_owned_data_without_config(
    wheel: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Incompressible bounded payload exercises real gzip/hash/extraction work
    # without assuming Actions/cache's transport bytes are this archive's size.
    data = os.urandom(2 * 1024**2)
    source = seed(wheel, environment, data)
    (source / "ccache.conf").write_text("compiler = poisoned\n")
    (source / "stats").write_text("old statistics")
    (source / "c").symlink_to(Path(environment["HOME"]), target_is_directory=True)
    wheel.package()
    packaged = Path(outputs(environment)["path"])
    manifest = json.loads((packaged / "manifest.json").read_text())
    assert manifest["files"] == 1 and manifest["decoded_bytes"] == len(data)
    assert 0 < manifest["archive_bytes"] < len(data) + 16384
    with tarfile.open(packaged / "cache.tar.gz") as archive:
        assert archive.getnames() == [DATA_NAME]
        assert all(m.uid == 0 and m.gid == 0 for m in archive.getmembers())
    # Retain source independently, proving only a missing normal target activates.
    source.rename(source.with_name("original-cache"))
    restored = stage(wheel, environment, monkeypatch)
    shutil.copyfile(packaged / "cache.tar.gz", restored / "download/cache.tar.gz")
    shutil.copyfile(packaged / "manifest.json", restored / "download/manifest.json")
    expected = {k: manifest[k] for k in wheel.identity()} | {
        "run_id": "123",
        "run_attempt": "2",
        "source_sha": "a" * 40,
    }
    (restored / "expected.json").write_text(json.dumps(expected))
    wheel.verify()
    assert not source.exists()
    inode = (restored / "cache").stat().st_ino
    wheel.activate()
    assert source.stat().st_ino == inode
    assert (source / DATA_NAME).read_bytes() == data
    assert (source / DATA_NAME).stat().st_uid == os.getuid()
    assert not (source / "ccache.conf").exists()
    wheel.cleanup()
    assert source.is_dir() and not restored.exists()
    assert (source.with_name("original-cache") / "ccache.conf").exists()


@pytest.mark.parametrize(
    "key",
    [
        "repository",
        "workflow",
        "format",
        "manylinux_image",
        "cuda_version",
        "cuda_arch",
        "ccache_version",
        "run_id",
        "run_attempt",
        "source_sha",
        "sha256",
    ],
)
def test_wheel_identity_and_digests_fail_closed(
    wheel: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    restored = stage(wheel, environment, monkeypatch)
    path = restored / "download/manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        wheel.verify()
    assert not (restored / "verified").exists() and not wheel.cache_root().exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside", tarfile.REGTYPE),
        ("ccache.conf", tarfile.REGTYPE),
        (DATA_NAME, tarfile.SYMTYPE),
        (DATA_NAME, tarfile.LNKTYPE),
        (DATA_NAME, tarfile.FIFOTYPE),
        (DATA_NAME, tarfile.XHDTYPE),
    ],
)
def test_wheel_archive_members_fail_closed(
    wheel: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kind: bytes,
) -> None:
    restored = stage(wheel, environment, monkeypatch)
    replace_archive(restored, [(name, kind, b"")])
    with pytest.raises((ValueError, tarfile.TarError)):
        wheel.verify()
    assert not wheel.cache_root().exists()


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE", "MAX_DECODED", "MAX_FILES"])
def test_wheel_caps_fail_before_activation_or_publication(
    wheel: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    stage(wheel, environment, monkeypatch)
    monkeypatch.setattr(wheel, limit, 0)
    with pytest.raises(ValueError):
        wheel.verify()
    assert not wheel.cache_root().exists()
    source = seed(wheel, environment)
    with pytest.raises(ValueError):
        wheel.package()
    assert not outputs(environment)
    assert (source / DATA_NAME).read_bytes() == b"cache object"


@pytest.mark.parametrize("kind", ["ordinary", "symlink", "interrupted"])
def test_wheel_activation_never_replaces_primary(
    wheel: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    restored = stage(wheel, environment, monkeypatch)
    wheel.verify()
    primary = wheel.cache_root()
    if kind == "ordinary":
        primary.mkdir()
        (primary / "sentinel").write_text("primary")
    elif kind == "symlink":
        primary.symlink_to(Path(environment["HOME"]), target_is_directory=True)
    else:
        (restored / "verified").unlink()
    with pytest.raises(ValueError):
        wheel.activate()
    wheel.cleanup()
    assert not restored.exists()
    if kind == "ordinary":
        assert (primary / "sentinel").read_text() == "primary"
    elif kind == "symlink":
        assert primary.is_symlink()
    else:
        assert not primary.exists()


@pytest.mark.parametrize("wrong", ["cpu", "attempt", "outside", "symlink"])
def test_wheel_cleanup_rejects_other_ownership(
    wheel: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    wrong: str,
) -> None:
    restored = stage(wheel, environment, monkeypatch)
    target = {
        "cpu": restored.with_name("gqc-cpu-123-2-fixture"),
        "attempt": restored.with_name("gqc-wheel-123-1-fixture"),
        "outside": Path(environment["HOME"]) / restored.name,
        "symlink": restored.with_name("gqc-wheel-123-2-link"),
    }[wrong]
    if wrong == "symlink":
        target.symlink_to(restored, target_is_directory=True)
    else:
        target.mkdir()
    monkeypatch.setenv("WHEEL_CACHE_STAGE", str(target))
    with pytest.raises(ValueError):
        wheel.cleanup()
    assert restored.is_dir() and target.is_dir()


@pytest.mark.parametrize("missing", ["missing", "version", "attempt", "symlink"])
def test_publication_requires_fresh_observed_toolchain(
    wheel: Any,
    environment: dict[str, str],
    missing: str,
) -> None:
    source = seed(wheel, environment)
    receipt = source.with_name("ccache-wheel-toolchain.json")
    value = json.loads(receipt.read_text())
    if missing == "missing":
        receipt.unlink()
    elif missing == "symlink":
        saved = receipt.with_suffix(".saved")
        receipt.rename(saved)
        receipt.symlink_to(saved)
    else:
        value["ccache_version" if missing == "version" else "attempt"] = "wrong"
        receipt.write_text(json.dumps(value))
    with pytest.raises((ValueError, OSError)):
        wheel.package()
    assert not outputs(environment)
    assert (source / DATA_NAME).is_file()


@pytest.mark.parametrize(
    "ccache,cuda,arch,valid",
    [
        ("4.14", "12.9", "120", True),
        ("4.15", "12.9", "120", False),
        ("4.14", "13.0", "120", False),
        ("4.14", "12.9", "90", False),
    ],
)
def test_recording_observes_tools_not_expected_label(
    wheel: Any,
    environment: dict[str, str],
    tmp_path: Path,
    ccache: str,
    cuda: str,
    arch: str,
    valid: bool,
) -> None:
    for name, line in (
        ("ccache", f"ccache version {ccache}"),
        ("nvcc", f"Cuda compilation tools, release {cuda}, V{cuda}.86"),
    ):
        binary = tmp_path / name
        binary.write_text(f"#!/bin/sh\nprintf '%s\\n' '{line}'\n")
        binary.chmod(0o755)
    environment.update(
        PATH=str(tmp_path) + os.pathsep + environment["PATH"],
        CCACHE_DIR=str(wheel.cache_root()),
        GQC_WHEEL_SNAPSHOT_ID="123-2",
        GENERATIVEQC_ENABLE_CUDA="ON",
        GENERATIVEQC_CUDA_ARCHITECTURES=arch,
    )
    result = command("record_wheel_toolchain", environment)
    assert (result.returncode == 0) == valid, result.stderr
    receipt = wheel.cache_root().with_name("ccache-wheel-toolchain.json")
    assert receipt.exists() == valid
    if valid:
        assert json.loads(receipt.read_text())["ccache_version"] == ccache


@pytest.mark.parametrize(
    "age,changed,expected",
    [
        (60, 0, "false"),
        (10801, 0, "true"),
        (-30, 0, "true"),
        (60, 1, "true"),
        (60, 128, "true"),
    ],
)
def test_refresh_age_and_wheel_input_diff(
    wheel: Any,
    environment: dict[str, str],
    tmp_path: Path,
    age: int,
    changed: int,
    expected: str,
) -> None:
    run, artifact = run_and_artifact(wheel)
    artifact["created_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=age)
    ).isoformat()
    fake_api(tmp_path, environment, run, artifact)
    binary = tmp_path / "git"
    binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" > '{tmp_path}/diff-call'\nexit {changed}\n"
    )
    binary.chmod(0o755)
    result = command("refresh", environment)
    assert result.returncode == 0, result.stderr
    assert outputs(environment)["publish"] == expected
    if age == 60:
        args = (tmp_path / "diff-call").read_text()
        assert "--no-ext-diff --no-textconv" in args
        assert all(
            path in args.split()
            for path in (
                "python/ci",
                ".github",
                "tools",
                "src",
                "data",
                "pyproject.toml",
            )
        )


def test_wheel_workflow_preserves_optional_budgets_and_required_gates() -> None:
    text = (ROOT / ".github/workflows/wheels.yml").read_text()
    build_text = text.split("  build_wheels:\n", 1)[1].split("\n  pass:\n", 1)[0]
    job = fields(build_text, 4)
    entries = steps(build_text, 6)
    named = {entry.get("name"): entry for entry in entries}
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    assert job["timeout-minutes"] == 180
    restore = named["Restore trusted wheel snapshot after primary miss"]
    assert (
        restore["if"]
        == "steps.wheel_ccache.outcome == 'success' && steps.wheel_ccache.outputs.cache-matched-key == ''"
    )
    assert restore["timeout-minutes"] == 2 and restore["continue-on-error"] is True
    package = named["Package bounded wheel compiler snapshot"]
    assert all(
        condition in package["if"]
        for condition in (
            "github.repository == 'jinzhezenggroup/generativeqc'",
            "github.event_name == 'push'",
            "github.ref == 'refs/heads/master'",
            "steps.wheel_build.outcome == 'success'",
            "steps.wheel_payload.outcome == 'success'",
        )
    )
    assert entries.index(package) > entries.index(
        named["Verify wheel payload and CUDA linkage"]
    )
    publish = named["Publish bounded wheel compiler snapshot"]
    assert publish["with"]["retention-days"] == 1
    assert publish["with"]["compression-level"] == 0
    for entry in (
        package,
        publish,
        named["Decide whether the wheel snapshot needs refresh"],
    ):
        assert entry["continue-on-error"] is True
        assert entry["timeout-minutes"] <= 2
    assert named["Build and test wheel"]["timeout-minutes"] == 150
    assert "continue-on-error" not in named["Build and test wheel"]
    assert "continue-on-error" not in named["Verify wheel payload and CUDA linkage"]
    action = steps(
        (ROOT / ".github/actions/restore-wheel-ccache/action.yml").read_text(), 4
    )
    download = next(entry for entry in action if entry.get("id") == "download")
    assert download["with"]["digest-mismatch"] == "error"
    assert "artifact-ids" in download["with"] and "run-id" in download["with"]
    verify = next(entry for entry in action if entry.get("id") == "verify")
    assert verify["if"] == "steps.download.outcome == 'success'"
    assert "45s" in verify["run"] and "--profile wheel" in verify["run"]
    activate = next(
        entry for entry in action if entry.get("name") == "Activate wheel snapshot"
    )
    assert activate["if"] == "steps.verify.outcome == 'success'"
    assert all(entry["continue-on-error"] is True for entry in action)

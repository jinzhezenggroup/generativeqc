"""Exercise trusted master probe snapshots without changing Python shard inputs."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/ci_cpu_ccache_snapshot.py"
REPO = "jinzhezenggroup/generativeqc"
SHARDS = (
    "core-a",
    "core-b",
    "core-c",
    "uccsd-forces",
    "runtime-heavy",
    "posthf",
    "compiler-heavy",
    "ecp-forces",
)
DATA_NAME = "a/b/" + "0123456789abcdef" * 2 + "R"


def load(profile: str = "probe", shard: str = "core-a") -> Any:
    spec = importlib.util.spec_from_file_location("probe_snapshot", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if profile != "cpu":
        module.select_profile(profile, shard if profile == "probe" else None)
    return module


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    home, temp = tmp_path / "home", tmp_path / "temp"
    home.mkdir()
    temp.mkdir()
    values = {
        "HOME": str(home),
        "RUNNER_TEMP": str(temp),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_EVENT_NAME": "schedule",
        "GITHUB_REF": "refs/heads/master",
        "ImageOS": "ubuntu24",
        "RUNNER_ARCH": "X64",
        "CCACHE_VERSION": "4.14",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return os.environ.copy()


@pytest.fixture
def probe(environment: dict[str, str]) -> Any:
    return load()


def command(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def outputs(env: dict[str, str]) -> dict[str, str]:
    path = Path(env["GITHUB_OUTPUT"])
    return (
        dict(line.split("=", 1) for line in path.read_text().splitlines())
        if path.exists()
        else {}
    )


def run_and_artifact(
    module: Any, event: str = "schedule", run_id: int = 456
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = {
        "id": run_id,
        "run_attempt": 3,
        "head_sha": "b" * 40,
        "path": module.WORKFLOW,
        "event": event,
        "head_branch": "master",
        "status": "completed",
        "conclusion": "success",
        "repository": {"id": 9, "full_name": REPO},
        "head_repository": {"id": 9, "full_name": REPO},
    }
    artifact = {
        "id": 789,
        "name": f"{module.ARTIFACT}-3",
        "expired": False,
        "size_in_bytes": 1000,
        "digest": "sha256:" + "c" * 64,
        "workflow_run": {
            "id": run_id,
            "head_sha": "b" * 40,
            "head_branch": "master",
            "repository_id": 9,
            "head_repository_id": 9,
        },
    }
    return run, artifact


def seed(probe: Any, data: bytes = b"compiled probe object") -> Path:
    root = probe.cache_root()
    path = root / DATA_NAME
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return root


def replace_archive(
    probe: Any, stage: Path, members: list[tuple[str, bytes, bytes]]
) -> None:
    archive = stage / "download/cache.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as tar:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            info.type, info.size = kind, len(data)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "/tmp/outside"
            tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    manifest = json.loads((stage / "expected.json").read_text()) | {
        "sha256": probe.digest(archive),
        "archive_bytes": archive.stat().st_size,
        "files": len(members),
        "decoded_bytes": sum(len(data) for _, _, data in members),
    }
    (stage / "download/manifest.json").write_text(json.dumps(manifest))


def stage(probe: Any, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = probe.cache_root().parent / (probe.stage_prefix() + "fixture")
    (path / "download").mkdir(parents=True)
    expected = probe.identity("schedule") | {
        "run_id": "456",
        "run_attempt": "3",
        "source_sha": "b" * 40,
    }
    (path / "expected.json").write_text(json.dumps(expected))
    monkeypatch.setenv("PROBE_CACHE_STAGE", str(path))
    replace_archive(probe, path, [(DATA_NAME, tarfile.REGTYPE, b"cached object")])
    return path


@pytest.mark.parametrize("shard", SHARDS)
def test_closed_shard_profiles_and_budgets(
    environment: dict[str, str], shard: str
) -> None:
    module = load(shard=shard)
    assert module.PROBE_SHARDS == frozenset(SHARDS)
    assert module.ARTIFACT == f"python-probe-ccache-master-ubuntu24-x86_64-v1-{shard}"
    assert module.MAX_ARTIFACT == module.MAX_DECODED == 256 * 1024**2
    assert module.MAX_ARCHIVE == module.MAX_ARTIFACT - 1024**2
    assert module.MAX_FILES == 100_000
    assert (
        module.cache_root() == Path(environment["RUNNER_TEMP"]) / "ccache-python-probes"
    )
    assert module.STAGE_ENV == "PROBE_CACHE_STAGE"
    assert module.identity()["shard"] == shard
    assert module.identity()["python_version"] == "3.11"
    assert module.identity()["ccache_version"] == environment["CCACHE_VERSION"]


@pytest.mark.parametrize("profile", ["cpu", "wheel"])
def test_other_profiles_keep_push_contract_and_original_limits(
    environment: dict[str, str], profile: str
) -> None:
    module = load(profile)
    assert module.identity()["event"] == "push"
    assert (
        "shard" not in module.identity() and "python_version" not in module.identity()
    )
    assert module.STAGE_ENV == f"{profile.upper()}_CACHE_STAGE"
    if profile == "cpu":
        assert module.ARTIFACT == "cpu-ccache-master-ubuntu24-x86_64-v1"
        assert module.MAX_ARTIFACT == 512 * 1024**2
        assert module.MAX_DECODED == 2 * 1024**3
        assert module.cache_root() == Path(environment["HOME"]) / ".cache/ccache"
    else:
        assert (
            module.ARTIFACT
            == "wheel-ccache-master-manylinux-cuda12.9-sm80-sm90-sm120-ptx80-v3"
        )
        assert module.MAX_ARTIFACT == 2 * 1024**3
        assert module.MAX_DECODED == 3 * 1024**3
        assert module.cache_root() == Path(environment["RUNNER_TEMP"]) / "ccache"
    assert module.MAX_ARCHIVE == module.MAX_ARTIFACT - 1024**2
    for event in (
        "push",
        "schedule",
        "workflow_dispatch",
        "pull_request",
        "merge_group",
    ):
        run, _ = run_and_artifact(module, event)
        assert module.trusted_run(run, REPO) == (event == "push")


@pytest.mark.parametrize(
    "args",
    [
        ("--profile", "probe"),
        ("--profile", "probe", "--shard", ""),
        ("--profile", "probe", "--shard", "all"),
        ("--profile", "probe", "--shard", "../core-a"),
        ("--profile", "cpu", "--shard", "core-a"),
        ("--profile", "wheel", "--shard", "core-a"),
        ("--shard", "core-a"),
    ],
)
@pytest.mark.parametrize(
    "operation", ["discover", "package", "verify", "activate", "cleanup"]
)
def test_invalid_profile_shard_arguments_fail_before_work(
    environment: dict[str, str], args: tuple[str, ...], operation: str
) -> None:
    result = command(environment, operation, *args)
    assert result.returncode == 2 and "error:" in result.stderr
    assert not outputs(environment)
    assert not list(Path(environment["RUNNER_TEMP"]).iterdir())


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
@pytest.mark.parametrize("consumer", ["pull_request", "merge_group", "push"])
def test_discovery_binds_producer_event_not_consumer(
    probe: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    consumer: str,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", consumer)
    monkeypatch.setenv("GITHUB_REF", "refs/pull/3000/merge")
    run, artifact = run_and_artifact(probe, event)
    calls = []

    def api(endpoint: str, **fields: str) -> dict[str, Any]:
        calls.append((endpoint, fields))
        if endpoint.endswith("/artifacts"):
            return {"artifacts": [artifact]}
        return {"workflow_runs": [run] if fields["event"] == event else []}

    monkeypatch.setattr(probe, "api", api)
    probe.discover()
    selected = outputs(environment)
    assert selected["artifact_id"] == "789" and selected["run_id"] == "456"
    path = Path(selected["stage"])
    assert path.parent == Path(environment["RUNNER_TEMP"])
    expected = json.loads((path / "expected.json").read_text())
    assert expected["event"] == event and expected["ref"] == "refs/heads/master"
    assert expected["source_sha"] == "b" * 40 and expected["run_attempt"] == "3"
    assert expected["shard"] == "core-a"
    assert [fields["event"] for _, fields in calls[:2]] == [
        "schedule",
        "workflow_dispatch",
    ]
    assert all(
        fields["branch"] == "master" and fields["status"] == "success"
        for _, fields in calls[:2]
    )
    assert calls[-1][0].endswith("/actions/runs/456/artifacts")
    assert calls[-1][1]["name"] == f"{probe.ARTIFACT}-3"
    assert len(calls) == 3 and not probe.cache_root().exists()


def test_discovery_merges_newest_producers_with_five_request_bound(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def api(endpoint: str, **fields: str) -> dict[str, Any]:
        calls.append((endpoint, fields))
        if endpoint.endswith("/artifacts"):
            return {"artifacts": []}
        ids = [1, 9, 8] if fields["event"] == "schedule" else [2, 10, 7]
        return {
            "workflow_runs": [
                run_and_artifact(probe, fields["event"], rid)[0] for rid in ids
            ]
        }

    monkeypatch.setattr(probe, "api", api)
    assert probe.find_snapshot() is None
    assert len(calls) == 5
    assert [endpoint.rsplit("/", 2)[1] for endpoint, _ in calls[2:]] == ["10", "9", "8"]


@pytest.mark.parametrize(
    "key,bad",
    [
        ("path", ".github/workflows/wheels.yml"),
        ("event", "push"),
        ("event", "pull_request"),
        ("event", "merge_group"),
        ("event", "pull_request_target"),
        ("event", ""),
        ("head_branch", "topic"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("conclusion", None),
        ("head_sha", "b" * 39),
        ("run_attempt", 0),
        ("run_attempt", True),
        ("id", "456"),
        ("id", -1),
        ("repository", {"id": 9, "full_name": "fork/generativeqc"}),
        ("head_repository", {"id": 10, "full_name": REPO}),
        ("head_repository", {"id": 9, "full_name": "fork/generativeqc"}),
    ],
)
def test_untrusted_producer_is_rejected(probe: Any, key: str, bad: Any) -> None:
    run, _ = run_and_artifact(probe)
    run[key] = bad
    assert not probe.trusted_run(run, REPO)


def test_fork_producer_is_rejected_even_if_repository_argument_matches(
    probe: Any,
) -> None:
    run, _ = run_and_artifact(probe)
    for key in ("repository", "head_repository"):
        run[key]["full_name"] = "fork/generativeqc"
    assert not probe.trusted_run(run, "fork/generativeqc")


@pytest.mark.parametrize(
    "key,bad",
    [
        ("name", "python-probe-ccache-master-ubuntu24-x86_64-v1-core-b-3"),
        ("name", "python-probe-ccache-master-ubuntu24-x86_64-v1-core-a-2"),
        ("name", "cpu-ccache-master-ubuntu24-x86_64-v1-3"),
        ("expired", True),
        ("id", 0),
        ("id", True),
        ("size_in_bytes", 256 * 1024**2 + 1),
        ("size_in_bytes", 0),
        ("digest", ""),
        ("digest", "sha256:" + "a" * 63),
        ("digest", "sha256:" + "g" * 64),
    ],
)
def test_wrong_artifact_shard_attempt_or_digest_is_rejected(
    probe: Any, key: str, bad: Any
) -> None:
    run, artifact = run_and_artifact(probe)
    artifact[key] = bad
    assert not probe.trusted_artifact(artifact, run)


@pytest.mark.parametrize(
    "key", ["id", "head_sha", "head_branch", "repository_id", "head_repository_id"]
)
def test_artifact_must_match_immutable_run_origin(probe: Any, key: str) -> None:
    run, artifact = run_and_artifact(probe)
    artifact["workflow_run"][key] = "wrong"
    assert not probe.trusted_artifact(artifact, run)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("GITHUB_EVENT_NAME", "push"),
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("GITHUB_EVENT_NAME", "merge_group"),
        ("GITHUB_REF", "refs/heads/topic"),
        ("GITHUB_REPOSITORY", "fork/generativeqc"),
        ("GITHUB_SHA", "a" * 39),
        ("GITHUB_RUN_ID", "0"),
        ("GITHUB_RUN_ATTEMPT", "../2"),
        ("ImageOS", "ubuntu22"),
        ("RUNNER_ARCH", "ARM64"),
        ("RUNNER_TEMP", "relative"),
    ],
)
def test_package_rejects_wrong_origin_without_modifying_source(
    probe: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad: str,
) -> None:
    source = seed(probe)
    monkeypatch.setenv(key, bad)
    with pytest.raises(ValueError):
        probe.package()
    assert not outputs(environment)
    assert (source / DATA_NAME).read_bytes() == b"compiled probe object"


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_safe_roundtrip_excludes_configuration_and_preserves_primary(
    probe: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    event: str,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    data = os.urandom(512 * 1024)
    source = seed(probe, data)
    (source / "ccache.conf").write_text("sloppiness = time_macros\n")
    (source / "stats").write_text("untrusted statistics")
    (source / "a/tmp.log").write_text("untrusted log")
    (source / "c").symlink_to(Path(environment["HOME"]), target_is_directory=True)
    probe.package()
    packaged = Path(outputs(environment)["path"])
    manifest = json.loads((packaged / "manifest.json").read_text())
    assert manifest["event"] == event and manifest["shard"] == "core-a"
    assert manifest["files"] == 1 and manifest["decoded_bytes"] == len(data)
    assert manifest["run_id"] == "123" and manifest["run_attempt"] == "2"
    assert manifest["source_sha"] == "a" * 40
    with tarfile.open(packaged / "cache.tar.gz") as tar:
        assert tar.getnames() == [DATA_NAME]
    original = source.with_name("original-probes")
    source.rename(original)
    restored = stage(probe, monkeypatch)
    expected = {key: manifest[key] for key in probe.identity(event)} | {
        "run_id": "123",
        "run_attempt": "2",
        "source_sha": "a" * 40,
    }
    (restored / "expected.json").write_text(json.dumps(expected))
    for name in ("cache.tar.gz", "manifest.json"):
        shutil.copyfile(packaged / name, restored / "download" / name)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "merge_group")
    probe.verify()
    assert not source.exists()
    inode = (restored / "cache").stat().st_ino
    probe.activate()
    assert source.stat().st_ino == inode and (source / DATA_NAME).read_bytes() == data
    assert not (source / "ccache.conf").exists() and not (source / "stats").exists()
    probe.cleanup()
    assert not restored.exists() and source.is_dir()
    assert (original / "ccache.conf").read_text() == "sloppiness = time_macros\n"


@pytest.mark.parametrize(
    "key",
    [
        "format",
        "repository",
        "workflow",
        "event",
        "ref",
        "shard",
        "image_os",
        "arch",
        "ccache_version",
        "python_version",
        "run_id",
        "run_attempt",
        "source_sha",
        "sha256",
    ],
)
def test_manifest_identity_and_full_digest_fail_closed(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    restored = stage(probe, monkeypatch)
    path = restored / "download/manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        probe.verify()
    assert not (restored / "verified").exists() and not probe.cache_root().exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        (name, tarfile.REGTYPE)
        for name in (
            "../escape",
            "/tmp/escape",
            "a/../ccache.conf",
            "ccache.conf",
            "a/stats",
            "a/log",
        )
    ]
    + [
        (DATA_NAME, kind)
        for kind in (
            tarfile.SYMTYPE,
            tarfile.LNKTYPE,
            tarfile.FIFOTYPE,
            tarfile.CHRTYPE,
            tarfile.DIRTYPE,
            tarfile.XHDTYPE,
            tarfile.GNUTYPE_LONGNAME,
        )
    ],
)
def test_malicious_archive_members_never_activate(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kind: bytes,
) -> None:
    restored = stage(probe, monkeypatch)
    replace_archive(probe, restored, [(name, kind, b"")])
    with pytest.raises((ValueError, tarfile.TarError)):
        probe.verify()
    assert not (restored / "verified").exists() and not probe.cache_root().exists()


@pytest.mark.parametrize(
    "damage", ["checksum", "truncated", "footer", "appended", "duplicate", "padding"]
)
def test_malformed_archive_framing_is_bounded_and_rejected(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    restored = stage(probe, monkeypatch)
    archive = restored / "download/cache.tar.gz"
    if damage == "duplicate":
        replace_archive(probe, restored, [(DATA_NAME, tarfile.REGTYPE, b"a")] * 2)
    else:
        compressed = archive.read_bytes()
        if damage == "appended":
            changed = compressed + gzip.compress(b"hidden payload")
        elif damage == "padding":
            decoded = bytearray(gzip.decompress(compressed))
            decoded[512 + len(b"cached object")] = 1
            changed = gzip.compress(decoded)
        else:
            changed = (
                compressed[: len(compressed) // 2]
                if damage == "truncated"
                else compressed[:-4]
            )
        archive.write_bytes(changed)
        if damage != "checksum":
            path = restored / "download/manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(
                sha256=probe.digest(archive), archive_bytes=archive.stat().st_size
            )
            path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, EOFError, OSError, tarfile.TarError)):
        probe.verify()
    assert not (restored / "verified").exists()


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE", "MAX_DECODED", "MAX_FILES"])
def test_size_and_count_caps_leave_source_unchanged(
    probe: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    restored = stage(probe, monkeypatch)
    monkeypatch.setattr(probe, limit, 0)
    with pytest.raises(ValueError):
        probe.verify()
    assert not probe.cache_root().exists() and not (restored / "verified").exists()
    source = seed(probe)
    with pytest.raises(ValueError):
        probe.package()
    assert not outputs(environment)
    assert (source / DATA_NAME).read_bytes() == b"compiled probe object"


@pytest.mark.parametrize("kind", ["ordinary", "symlink", "interrupted"])
def test_activation_never_replaces_primary_or_exposes_unverified_data(
    probe: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    restored = stage(probe, monkeypatch)
    probe.verify()
    target = probe.cache_root()
    if kind == "ordinary":
        target.mkdir()
        (target / "sentinel").write_text("original")
    elif kind == "symlink":
        target.symlink_to(Path(environment["HOME"]), target_is_directory=True)
    else:
        (restored / "verified").unlink()
    with pytest.raises(ValueError):
        probe.activate()
    probe.cleanup()
    assert not restored.exists()
    if kind == "ordinary":
        assert (target / "sentinel").read_text() == "original"
    elif kind == "symlink":
        assert target.is_symlink()
    else:
        assert not target.exists()


@pytest.mark.parametrize("wrong", ["shard", "attempt", "profile", "outside", "symlink"])
@pytest.mark.parametrize("operation", ["verify", "activate", "cleanup"])
def test_stage_operations_require_this_shard_run_attempt_and_directory(
    probe: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    wrong: str,
    operation: str,
) -> None:
    restored = stage(probe, monkeypatch)
    target = {
        "shard": restored.with_name("gqc-probe-core-b-123-2-fixture"),
        "attempt": restored.with_name("gqc-probe-core-a-123-1-fixture"),
        "profile": restored.with_name("gqc-cpu-123-2-fixture"),
        "outside": Path(environment["HOME"]) / restored.name,
        "symlink": restored.with_name(probe.stage_prefix() + "symlink"),
    }[wrong]
    if wrong == "symlink":
        target.symlink_to(restored, target_is_directory=True)
    else:
        target.mkdir()
    monkeypatch.setenv("PROBE_CACHE_STAGE", str(target))
    with pytest.raises(ValueError, match="owned"):
        getattr(probe, operation)()
    assert restored.is_dir() and target.is_dir()


@pytest.mark.parametrize("failure", ["schedule", "workflow_dispatch", "both"])
def test_failed_probe_event_query_can_use_other_producer_without_retries(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    calls = []
    available = "workflow_dispatch" if failure == "schedule" else "schedule"
    run, artifact = run_and_artifact(probe, available)

    def api(endpoint: str, **fields: str) -> dict[str, Any]:
        calls.append((endpoint, fields))
        if endpoint.endswith("/artifacts"):
            return {"artifacts": [artifact]}
        if fields["event"] == failure or failure == "both":
            raise subprocess.TimeoutExpired("gh", 8)
        return {"workflow_runs": [run]}

    monkeypatch.setattr(probe, "api", api)
    found = probe.find_snapshot()
    assert (found is None) == (failure == "both")
    assert len(calls) == (2 if failure == "both" else 3)
    if found is not None:
        assert found[0]["event"] == available


@pytest.mark.parametrize("profile", ["cpu", "wheel"])
def test_other_profiles_still_propagate_run_query_failure(
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    module = load(profile)

    def api(endpoint: str, **fields: str) -> dict[str, Any]:
        raise subprocess.TimeoutExpired("gh", 8)

    monkeypatch.setattr(module, "api", api)
    with pytest.raises(subprocess.TimeoutExpired):
        module.find_snapshot()


@pytest.mark.parametrize("kind", ["root-link", "data-link"])
def test_publication_rejects_symlinked_probe_data(
    probe: Any,
    environment: dict[str, str],
    kind: str,
) -> None:
    source = seed(probe)
    if kind == "root-link":
        saved = source.with_name("original-probes")
        source.rename(saved)
        source.symlink_to(saved, target_is_directory=True)
    else:
        data = source / DATA_NAME
        saved = source / "saved-object"
        data.rename(saved)
        data.symlink_to(saved)
    with pytest.raises(ValueError):
        probe.package()
    assert not outputs(environment)
    assert (source / DATA_NAME).read_bytes() == b"compiled probe object"


def test_existing_normal_probe_cache_skips_network_discovery(
    probe: Any,
    environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = seed(probe)

    def api(endpoint: str, **fields: str) -> dict[str, Any]:
        pytest.fail("existing cache must not trigger snapshot discovery")

    monkeypatch.setattr(probe, "api", api)
    probe.discover()
    assert not outputs(environment)
    assert (source / DATA_NAME).read_bytes() == b"compiled probe object"


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_cli_package_uses_probe_profile_and_selected_shard(
    environment: dict[str, str],
    event: str,
) -> None:
    module = load(shard="ecp-forces")
    seed(module)
    environment["GITHUB_EVENT_NAME"] = event
    result = command(
        environment, "package", "--profile", "probe", "--shard", "ecp-forces"
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (Path(outputs(environment)["path"]) / "manifest.json").read_text()
    )
    assert manifest["shard"] == "ecp-forces" and manifest["event"] == event

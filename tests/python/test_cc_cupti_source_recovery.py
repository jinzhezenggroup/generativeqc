"""Synthetic Git/blob/patch failure controls; never native or scientific execution."""

from __future__ import annotations

import difflib
import gzip
import hashlib
import json
import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from benchmarks.cc_cupti_source_recovery import (
    COMPANIONS,
    PUBLICATION,
    RECOVERY_SCHEMA,
    SourceUnavailable,
    recover_tooling_sources,
)
from benchmarks.cc_trace_reader import InvalidTrace

if TYPE_CHECKING:
    from pathlib import Path


def git(root: Path, *arguments: str, data: bytes | None = None) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
    )
    return subprocess.run(
        ["git", "-C", str(root), "-c", "core.autocrlf=false", *arguments],
        input=data,
        capture_output=True,
        check=True,
        env=env,
        timeout=15,
    ).stdout


def identity(raw: bytes) -> dict:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def fixture(root: Path, fault: str | None = None) -> tuple[str, dict, dict]:
    root.mkdir()
    git(root, "init", "--quiet", "--template=", ".")
    canonical = {
        name: ("canonical fixture " + name + "\n").encode() for name in COMPANIONS
    }
    measured = dict(canonical)
    for name in ("benchmarks/cc_cupti_campaign.py", "benchmarks/cc_source_census.py"):
        measured[name] = ("historical fixture " + name + "\n").encode()
    measured["worker.py"] = (
        b"raise RuntimeError('historical source must never be executed')\n"
    )
    old_ids = {
        name: identity(raw)
        | {
            "reconstructed_path": "frozen-campaign/worker.py"
            if name == "worker.py"
            else name
        }
        for name, raw in measured.items()
    }
    source_ids = {
        name: identity(raw)
        | {
            "git_blob_oid": git(root, "hash-object", "--stdin", data=raw)
            .decode()
            .strip()
        }
        for name, raw in canonical.items()
    }
    patch = []
    for name in (
        "benchmarks/cc_cupti_campaign.py",
        "benchmarks/cc_source_census.py",
        "worker.py",
    ):
        target = old_ids[name]["reconstructed_path"]
        before = canonical.get(name, b"")
        patch.append(
            "diff --git a/"
            + target
            + " b/"
            + target
            + "\n"
            + ("new file mode 100644\n" if not before else "")
            + "".join(
                difflib.unified_diff(
                    before.decode().splitlines(keepends=True),
                    measured[name].decode().splitlines(keepends=True),
                    fromfile="a/" + target if before else "/dev/null",
                    tofile="b/" + target,
                )
            )
        )
    delta = "".join(patch).encode()
    if fault == "reverse-content":
        delta = delta.replace(b"historical fixture", b"corrupted fixture")
    elif fault == "unsafe-path":
        delta = delta.replace(
            b"diff --git a/frozen-campaign/worker.py b/frozen-campaign/worker.py",
            b"diff --git a/../../escaped.py b/../../escaped.py",
        )
    elif fault == "symlink-mode":
        delta = delta.replace(b"new file mode 100644", b"new file mode 120000")
    elif fault == "broken-patch":
        delta = delta[:-10]
    capsule = {"original_documents": {"worker.py": identity(measured["worker.py"])}}
    record = {
        "historical_tooling_input_identities": old_ids,
        "historical_tooling_reconstruction": {
            "schema": RECOVERY_SCHEMA,
            "canonical_input_identities": source_ids,
        },
    }
    members = {
        "README.md": b"synthetic storage fixture, not measured evidence\n",
        "campaign.json.gz": gzip.compress(json.dumps(capsule).encode(), mtime=0),
        "oracle.json.gz": gzip.compress(b"{}", mtime=0),
        "validation.json.gz": gzip.compress(json.dumps(record).encode(), mtime=0),
        "measured-tooling.patch.gz": gzip.compress(delta, mtime=0),
    }
    entries = [{"path": name, **identity(raw)} for name, raw in members.items()]
    entry = next(e for e in entries if e["path"] == "measured-tooling.patch.gz")
    entry.update(
        decoded_bytes=len(delta), decoded_sha256=hashlib.sha256(delta).hexdigest()
    )
    members["publication.json"] = json.dumps({"files": entries}).encode()
    for name, raw in canonical.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            raw
            + (b"changed" if fault == "wrong-blob" and name == COMPANIONS[0] else b"")
        )
    for name, raw in members.items():
        path = root / PUBLICATION / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    paths = [*canonical, *(PUBLICATION + "/" + name for name in members)]
    git(root, "add", "--", *paths)
    if fault == "symlink-blob":
        name = COMPANIONS[0]
        git(
            root,
            "update-index",
            "--add",
            "--cacheinfo",
            "120000," + source_ids[name]["git_blob_oid"] + "," + name,
        )
    return git(root, "write-tree").decode().strip(), capsule, measured


def test_source_recovery_closes_all_inputs_without_history_or_source_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    tree, capsule, measured = fixture(root)
    output = tmp_path / "recovered"
    result = recover_tooling_sources(root, tree, output, capsule)
    assert result["status"].startswith("PASS") and result[
        "recovered_input_count"
    ] == len(measured)
    assert result["historical_source_executed"] is False
    for name, raw in measured.items():
        target = "frozen-campaign/worker.py" if name == "worker.py" else name
        assert (output / target).read_bytes() == raw


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-blob",
        "reverse-content",
        "unsafe-path",
        "symlink-mode",
        "broken-patch",
        "symlink-blob",
    ],
)
def test_source_recovery_refuses_wrong_or_unsafe_inputs(
    tmp_path: Path, fault: str
) -> None:
    root = tmp_path / "repository"
    tree, capsule, _ = fixture(root, fault)
    with pytest.raises((InvalidTrace, ValueError)):
        recover_tooling_sources(root, tree, tmp_path / "recovered", capsule)
    assert not (tmp_path.parent / "escaped.py").exists()


def test_missing_git_object_is_unavailable_not_pass(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _, capsule, _ = fixture(root)
    with pytest.raises(SourceUnavailable):
        recover_tooling_sources(root, "0" * 40, tmp_path / "recovered", capsule)
    assert not (tmp_path / "recovered").exists()


def test_recovery_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    tree, capsule, _ = fixture(root)
    output = tmp_path / "recovered"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_bytes(b"existing user output")
    with pytest.raises(FileExistsError):
        recover_tooling_sources(root, tree, output, capsule)
    assert sentinel.read_bytes() == b"existing user output"


def test_wrong_publication_cannot_bind_an_unrelated_capsule(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    tree, capsule, _ = fixture(root)
    capsule["original_documents"]["worker.py"]["sha256"] = "0" * 64
    with pytest.raises(InvalidTrace, match="this capsule"):
        recover_tooling_sources(root, tree, tmp_path / "recovered", capsule)


def test_caller_git_overrides_cannot_redirect_or_enable_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    tree, capsule, _ = fixture(root)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "foreign.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "foreign-tree"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "https:file:ext")
    result = recover_tooling_sources(root, tree, tmp_path / "recovered", capsule)
    assert result["status"].startswith("PASS")
    assert not (tmp_path / "foreign-tree").exists()


def test_legacy_ignored_lazy_switch_still_cannot_fetch_missing_promisor_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    tree, capsule, _ = fixture(root)
    oid = git(root, "rev-parse", tree + ":" + COMPANIONS[0]).decode().strip()
    missing = root / ".git/objects" / oid[:2] / oid[2:]
    assert missing.resolve().is_relative_to((root / ".git/objects").resolve())
    missing.chmod(0o600)  # Git's private fixture object is read-only on Windows.
    missing.unlink()
    git(
        root,
        "config",
        "remote.origin.url",
        "file://" + (tmp_path / "must-not-be-read").as_posix(),
    )
    git(root, "config", "remote.origin.promisor", "true")
    git(root, "config", "protocol.file.allow", "always")
    real_run = subprocess.run
    failures = []

    def legacy(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        env = dict(kwargs["env"])
        assert env["GIT_ALLOW_PROTOCOL"] == "" and env["GIT_TERMINAL_PROMPT"] == "0"
        env.pop("GIT_NO_LAZY_FETCH", None)  # Model an older Git ignoring this switch.
        kwargs["env"] = env
        result = real_run(*args, **kwargs)
        if result.returncode:
            failures.append(result.stderr)
        return result

    monkeypatch.setattr(subprocess, "run", legacy)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file:https:ext")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    with pytest.raises(SourceUnavailable):
        recover_tooling_sources(root, tree, tmp_path / "recovered", capsule)
    assert any(b"transport 'file' not allowed" in row for row in failures)
    assert not missing.exists() and not (tmp_path / "recovered").exists()

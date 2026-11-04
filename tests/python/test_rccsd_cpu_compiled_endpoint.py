"""Qualification provenance and bounded subprocesses for the RCCSD CPU campaign."""

from __future__ import annotations

import hashlib
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from benchmarks import rccsd_cpu_compiled_endpoint as benchmark

if TYPE_CHECKING:
    from pathlib import Path


def test_native_artifact_receipts_survive_cache_removal(tmp_path: Path) -> None:
    artifact = tmp_path / "program" / "program.so"
    artifact.parent.mkdir()
    artifact.write_bytes(b"compiled-example")
    (artifact.parent / "source.cpp").write_text("not a shared library")
    receipts = benchmark._native_artifacts(tmp_path)
    artifact.unlink()
    assert receipts == [
        {
            "path": "program/program.so",
            "bytes": len(b"compiled-example"),
            "sha256": hashlib.sha256(b"compiled-example").hexdigest(),
        }
    ]


def test_archive_source_identity_never_invents_a_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128),
    )
    assert benchmark._source_identity(tmp_path)["kind"] == "unavailable"
    assert benchmark._source_identity(tmp_path)["commit"] is None
    commit = "1234567890" * 4
    (tmp_path / "SOURCE_COMMIT").write_text(commit + "\n")
    identity = benchmark._source_identity(tmp_path)
    assert identity["kind"] == "archive-marker"
    assert identity["commit"] == commit
    assert identity["tracked_dirty"] is None


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_worker_timeout_refuses_before_starting_a_campaign(
    timeout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark", "--worker-timeout", timeout])
    with pytest.raises(SystemExit) as failure:
        benchmark.main()
    assert failure.value.code == 2


def test_worker_timeout_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark", "--worker-timeout", "0.25"])

    def timed_out(command: list[str], **kwargs: object) -> None:
        assert kwargs["timeout"] == 0.25
        raise subprocess.TimeoutExpired(command, 0.25)

    monkeypatch.setattr(benchmark.subprocess, "run", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        benchmark.main()

"""The retained Ragged XC reader must enforce its offline recovery contract."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/results/ragged-xc-default-20261012"
REVISION = "47c7da365ff8b8b26579ddd18ca8240fb19348c1"


@pytest.fixture
def reader() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ragged_xc_evidence_summary", EVIDENCE / "summarize.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("returncode", (0, 1))
def test_git_read_overrides_inherited_network_and_prompt_settings(
    reader: ModuleType, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    inherited = {
        "GIT_NO_LAZY_FETCH": "0",
        "GIT_ALLOW_PROTOCOL": "https:ssh:file",
        "GIT_TERMINAL_PROMPT": "1",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    calls = []
    payload = b"\x00\xffhistorical raw bytes\n"

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, payload, b"")

    monkeypatch.setattr(reader.subprocess, "run", run)
    path = "benchmarks/results/example.json.gz"
    if returncode:
        with pytest.raises(ValueError, match=f"git fetch origin {REVISION}"):
            reader.read_git_blob(ROOT, REVISION, path)
    else:
        assert reader.read_git_blob(ROOT, REVISION, path) == payload
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == ["git", "cat-file", "blob", f"{REVISION}:{path}"]
    assert kwargs["cwd"] == ROOT
    assert kwargs["capture_output"] is True and kwargs["check"] is False
    assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
    assert kwargs["env"]["GIT_ALLOW_PROTOCOL"] == ""
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert all(os.environ[name] == value for name, value in inherited.items())


def test_local_object_bytes_remain_exact(reader: ModuleType, tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    payload = b"\x1f\x8b\x00\xffcomplete retained bytes\n"
    (tmp_path / "payload.bin").write_bytes(payload)
    subprocess.run(["git", "add", "payload.bin"], cwd=tmp_path, check=True)
    tree = subprocess.check_output(
        ["git", "write-tree"], cwd=tmp_path, text=True
    ).strip()
    assert reader.read_git_blob(tmp_path, tree, "payload.bin") == payload
    with pytest.raises(ValueError, match="historical object unavailable"):
        reader.read_git_blob(tmp_path, tree, "missing.bin")


def test_retained_consumer_manifest_matches_current_files() -> None:
    for line in (EVIDENCE / "files.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        assert hashlib.sha256((EVIDENCE / path).read_bytes()).hexdigest() == digest

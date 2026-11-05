"""Exercise committed-input cache identity using real isolated Git inventories."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "probe_key", ROOT / "tools/ci_python_probe_key.py"
)
assert spec and spec.loader
key = importlib.util.module_from_spec(spec)
spec.loader.exec_module(key)


def git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Cache regression",
            "-c",
            "user.email=cache-regression@example.invalid",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def commit(root: Path) -> None:
    git(root, "add", ".")
    git(root, "commit", "--allow-empty", "-qm", "fixture")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    for name in ("src/probe.cpp", "tools/probe.py", "pyproject.toml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    commit(tmp_path)
    return tmp_path


def test_generated_files_and_commit_metadata_do_not_change_key(
    repository: Path,
) -> None:
    before = key.fingerprint(repository)
    assert len(before) == 64
    (repository / "tools/__pycache__").mkdir()
    (repository / "tools/__pycache__/probe.pyc").write_bytes(b"generated-bytecode")
    (repository / "src/generated.hpp").write_text("runtime generated\n")
    (repository / "src/probe.cpp").write_text("uncommitted build output\n")
    assert key.fingerprint(repository) == before
    git(repository, "commit", "--allow-empty", "-qm", "different commit metadata")
    assert key.fingerprint(repository) == before


@pytest.mark.parametrize("change", ("edit", "rename", "delete", "mode", "add"))
def test_committed_source_inventory_changes_key(repository: Path, change: str) -> None:
    before = key.fingerprint(repository)
    source = repository / "src/probe.cpp"
    if change == "edit":
        source.write_text("changed\n")
    elif change == "rename":
        source.rename(source.with_name("renamed.cpp"))
    elif change == "delete":
        source.unlink()
    elif change == "mode":
        source.chmod(0o755)
    else:
        (repository / "src/new.cpp").write_text("new\n")
    commit(repository)
    assert key.fingerprint(repository) != before


def test_unrelated_document_commit_does_not_churn_probe_key(repository: Path) -> None:
    before = key.fingerprint(repository)
    (repository / "README.md").write_text("new prose\n")
    commit(repository)
    assert key.fingerprint(repository) == before


def test_missing_git_inventory_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        key.fingerprint(tmp_path)
    git(tmp_path, "init", "-q")
    commit(tmp_path)
    with pytest.raises(ValueError, match="nonempty committed inventory"):
        key.fingerprint(tmp_path)


@pytest.mark.parametrize(
    "relative",
    (
        ".github/scripts/ci_select_python_shard.sh",
        ".github/scripts/ci_collect_python_shard.sh",
        ".github/workflows/ci.yml",
        ".github/actions/restore-python-probe-ccache/action.yml",
    ),
)
def test_committed_ci_routing_and_transport_changes_key(
    repository: Path, relative: str
) -> None:
    before = key.fingerprint(repository)
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("tracked transport or routing contract\n")
    commit(repository)
    assert key.fingerprint(repository) != before

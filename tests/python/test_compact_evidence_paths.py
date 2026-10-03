"""Compaction must not move files outside its declared publication."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import compact_evidence_publications as compact


def snapshot(root: Path) -> dict[str, bytes | str]:
    return {
        path.relative_to(root).as_posix(): str(path.readlink())
        if path.is_symlink()
        else path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "checkout"
    directory = root / "campaign"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"[17, 23]\n")
    files = []
    for name, role, raw in (
        ("samples.json", "samples", b"[1, 2]\n"),
        ("evidence.json", "evidence", b'{"energy": -1.25}\n'),
    ):
        (directory / name).write_bytes(raw)
        files.append(
            {
                "path": name,
                "role": role,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    (directory / "publication.json").write_text(json.dumps({"files": files}))
    monkeypatch.setattr(compact, "ROOT", root)
    monkeypatch.setattr(compact, "THRESHOLD", 1)
    return root, outside


def change_member(root: Path, name: str, raw: bytes) -> None:
    path = root / "campaign/publication.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0].update(
        path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize(
    "mode",
    ["traversal", "absolute", "alias", "symlink", "directory-symlink", "duplicate"],
)
def test_bad_inventory_cannot_move_or_delete_outside_files(
    publication: tuple[Path, Path], check: bool, mode: str
) -> None:
    root, outside = publication
    directory = root / "campaign"
    name = "samples.json"
    raw = outside.read_bytes()
    if mode == "traversal":
        name = "../../outside.json"
    elif mode == "absolute":
        name = str(outside)
    elif mode == "alias":
        name = "./samples.json"
        raw = (directory / "samples.json").read_bytes()
    elif mode == "symlink":
        (directory / name).unlink()
        (directory / name).symlink_to(outside)
    elif mode == "directory-symlink":
        (directory / "linked").symlink_to(outside.parent, target_is_directory=True)
        name = "linked/outside.json"
    else:
        raw = (directory / name).read_bytes()
    change_member(root, name, raw)
    if mode == "duplicate":
        path = directory / "publication.json"
        manifest = json.loads(path.read_text())
        manifest["files"].append(dict(manifest["files"][0]))
        path.write_text(json.dumps(manifest))
    before = snapshot(root.parent)
    with pytest.raises(ValueError, match="path|publication|duplicate"):
        compact.compact_publication("campaign/publication.json", check=check)
    assert snapshot(root.parent) == before


@pytest.mark.parametrize("check", [False, True])
def test_legal_compaction_preserves_decoded_bytes(
    publication: tuple[Path, Path], check: bool
) -> None:
    root, outside = publication
    before = snapshot(root.parent)
    changes = compact.compact_publication("campaign/publication.json", check=check)
    assert len(changes) == 2
    if check:
        assert snapshot(root.parent) == before
    else:
        for old, new, packed in changes:
            assert gzip.decompress(packed) == before["checkout/" + old]
            assert (root / new).read_bytes() == packed
            assert not (root / old).exists()
        assert outside.read_bytes() == before["outside.json"]


def test_script_entrypoint_imports_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(compact.__file__).resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "--check" in completed.stdout


@pytest.mark.parametrize("packed_parent", [False, True])
def test_compaction_rebinds_parts_inside_sample_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packed_parent: bool
) -> None:
    from tools.generativeqc_validation.record import load_publication_record

    root = tmp_path / "checkout"
    directory = root / "campaign"
    (directory / "parts").mkdir(parents=True)
    child = b'[{"seconds":0.12345678901234567},{"seconds":1e-14}]\n'
    samples = {
        "record_parts": {
            "timings": [
                {
                    "path": "parts/water.json",
                    "bytes": len(child),
                    "sha256": hashlib.sha256(child).hexdigest(),
                }
            ]
        }
    }
    samples_data = json.dumps(samples).encode()
    samples_name = "samples.json"
    if packed_parent:
        samples_name += ".gz"
        samples_data = gzip.compress(samples_data, mtime=0)
    evidence = json.dumps(
        {
            "attachments": [
                {
                    "path": samples_name,
                    "bytes": len(samples_data),
                    "sha256": hashlib.sha256(samples_data).hexdigest(),
                }
            ]
        }
    ).encode()
    members = [
        ("evidence.json", "evidence", evidence),
        (samples_name, "samples", samples_data),
        ("parts/water.json", "samples", child),
    ]
    files = []
    for name, role, data in members:
        (directory / name).write_bytes(data)
        files.append(
            {
                "path": name,
                "role": role,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (directory / "publication.json").write_text(json.dumps({"files": files}))
    monkeypatch.setattr(compact, "ROOT", root)
    monkeypatch.setattr(compact, "THRESHOLD", 1)
    expected = load_publication_record(directory, role="samples", name="samples.json")
    compact.compact_publication("campaign/publication.json")
    assert not (directory / "parts/water.json").exists()
    assert gzip.decompress((directory / "parts/water.json.gz").read_bytes()) == child
    assert (
        load_publication_record(directory, role="samples", name="samples.json")
        == expected
    )
    active = load_publication_record(directory)
    assert active["attachments"][0]["path"] == "samples.json.gz"
    assert (
        active["attachments"][0]["sha256"]
        == hashlib.sha256((directory / "samples.json.gz").read_bytes()).hexdigest()
    )
    assert compact.compact_publication("campaign/publication.json", check=True) == []


def test_cyclic_record_dependencies_fail_without_writes(
    publication: tuple[Path, Path],
) -> None:
    root, _ = publication
    directory = root / "campaign"
    files = []
    for name, role, target in [
        ("evidence.json", "evidence", "samples.json"),
        ("samples.json", "samples", "evidence.json"),
    ]:
        data = json.dumps(
            {"attachments": [{"path": target, "sha256": "0" * 64}]}
        ).encode()
        (directory / name).write_bytes(data)
        files.append(
            {
                "path": name,
                "role": role,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (directory / "publication.json").write_text(json.dumps({"files": files}))
    before = snapshot(root)
    with pytest.raises(ValueError, match="cyclic"):
        compact.compact_publication("campaign/publication.json")
    assert snapshot(root) == before


def test_default_inventory_discovers_only_tracked_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "benchmarks/results/current/publication.json"
    untracked = tmp_path / "benchmarks/results/transient/publication.json"
    ignored = tmp_path / ".artifacts/private/publication.json"
    for path in [tracked, untracked, ignored]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    subprocess.run(["git", "add", str(tracked)], cwd=tmp_path, check=True)
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    assert compact.publication_paths() == (
        "benchmarks/results/current/publication.json",
    )

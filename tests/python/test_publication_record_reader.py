"""Resolve stored records through their inventory without weakening identities."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import load_publication_record


def bundle(directory: Path, *, packed: bool = False) -> dict:
    values = [{"seconds": 0.12345678901234567}, {"seconds": 1e-14}]
    child = json.dumps(values).encode()
    child_name = "parts/water.json"
    if packed:
        child = gzip.compress(child, mtime=0)
        child_name += ".gz"
    record = {
        "record_parts": {
            "timings": [
                {
                    "path": child_name,
                    "bytes": len(child),
                    "sha256": hashlib.sha256(child).hexdigest(),
                }
            ]
        }
    }
    primary = json.dumps(record).encode()
    name = "samples.json"
    if packed:
        primary = gzip.compress(primary, mtime=0)
        name += ".gz"
    files = []
    for path, data in [(name, primary), (child_name, child)]:
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append(
            {
                "path": path,
                "role": "samples",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {"files": files}
    (directory / "publication.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("packed", [False, True])
def test_manifest_selection_preserves_record_values(
    tmp_path: Path, packed: bool
) -> None:
    bundle(tmp_path, packed=packed)
    assert load_publication_record(tmp_path, role="samples", name="samples.json") == {
        "timings": [{"seconds": 0.12345678901234567}, {"seconds": 1e-14}]
    }
    with pytest.raises(ValueError, match="unique"):
        load_publication_record(tmp_path, role="samples")


@pytest.mark.parametrize("fault", ["duplicate", "bytes", "sha256", "missing", "path"])
def test_invalid_inventory_fails_before_reading_record(
    tmp_path: Path, fault: str
) -> None:
    manifest = bundle(tmp_path)
    if fault == "duplicate":
        manifest["files"].append(dict(manifest["files"][0]))
    elif fault in {"bytes", "sha256"}:
        manifest["files"][1][fault] = 0 if fault == "bytes" else "0" * 64
    elif fault == "missing":
        (tmp_path / "parts/water.json").unlink()
    else:
        manifest["files"][1]["path"] = "../outside.json"
    (tmp_path / "publication.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_publication_record(tmp_path, role="samples", name="samples.json")


def test_missing_inventory_member_never_uses_an_undeclared_companion(
    tmp_path: Path,
) -> None:
    bundle(tmp_path)
    data = (tmp_path / "samples.json").read_bytes()
    (tmp_path / "samples.json.gz").write_bytes(gzip.compress(data, mtime=0))
    (tmp_path / "samples.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_publication_record(tmp_path, role="samples", name="samples.json")


def test_declared_member_symlink_cannot_escape(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    bundle(directory)
    member = directory / "parts/water.json"
    outside = tmp_path / "outside.json"
    member.rename(outside)
    member.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_publication_record(directory, role="samples", name="samples.json")


def test_inventory_symlink_cannot_escape(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    bundle(directory)
    inventory = directory / "publication.json"
    outside = tmp_path / "outside.json"
    inventory.rename(outside)
    inventory.symlink_to(outside)
    with pytest.raises(ValueError, match="inventory escapes"):
        load_publication_record(directory, role="samples", name="samples.json")


def test_logical_part_keys_stay_posix_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import PureWindowsPath

    from tools.generativeqc_validation import record

    bundle(tmp_path, packed=True)
    native_path = Path

    def windows_relative_path(value: str | Path) -> Path | PureWindowsPath:
        path = native_path(value)
        return path if path.is_absolute() else PureWindowsPath(value)

    # Keep the actual test filesystem native while exercising Windows semantics
    # for any accidental platform-specific calculation of logical member names.
    monkeypatch.setattr(record, "Path", windows_relative_path)
    assert record.load_publication_record(
        tmp_path, role="samples", name="samples.json"
    ) == {"timings": [{"seconds": 0.12345678901234567}, {"seconds": 1e-14}]}

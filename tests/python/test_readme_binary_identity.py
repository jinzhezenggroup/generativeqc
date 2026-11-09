"""CPU-only checks for the actual native owners behind endpoint metadata."""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from benchmarks import readme_method_endpoints as endpoints


def mapping(path: Path, *, inode: int | None = None, deleted: bool = False) -> str:
    """Model Linux maps using real file identity, without loading a native DSO."""
    status = path.stat()
    device = f"{os.major(status.st_dev):02x}:{os.minor(status.st_dev):02x}"
    selected_inode = status.st_ino if inode is None else inode
    suffix = " (deleted)" if deleted else ""
    return f"1000-2000 r-xp 00000000 {device} {selected_inode} {path}{suffix}\n"


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(endpoints.platform, "system", lambda: "Linux")
    selected = tmp_path / "libgenerativeqc.so.0.1.0"
    selected.write_bytes(b"selected native image")
    return selected


def test_segments_and_aliases_are_one_native_owner(
    library: Path, tmp_path: Path
) -> None:
    alias = tmp_path / "libgenerativeqc.so.0"
    alias.hardlink_to(library)
    maps = tmp_path / "maps"
    maps.write_text(mapping(library) * 3 + mapping(alias))
    owners = endpoints._native_library_mappings(library, maps=maps)
    assert owners is not None
    assert len(owners) == 1
    assert owners[0]["inode"] == library.stat().st_ino
    assert owners[0]["paths"] == sorted((str(library), str(alias)))


@pytest.mark.parametrize("deleted", (False, True))
def test_distinct_native_images_fail_even_when_old_path_is_deleted(
    library: Path, tmp_path: Path, deleted: bool
) -> None:
    old = tmp_path / "older" / "libgenerativeqc.so.0"
    old.parent.mkdir()
    old.write_bytes(b"older native image")
    maps = tmp_path / "maps"
    maps.write_text(mapping(library) + mapping(old, deleted=deleted))
    if deleted:
        old.unlink()
    with pytest.raises(RuntimeError, match="Multiple distinct native libraries"):
        endpoints._native_library_mappings(library, maps=maps)


def test_unrelated_overlay_and_cuda_libraries_are_not_native_owners(
    library: Path, tmp_path: Path
) -> None:
    maps = tmp_path / "maps"
    lines = [mapping(library)]
    for name in ("control.so", "libcudart.so.12"):
        overlay = tmp_path / name
        overlay.write_bytes(b"unrelated image")
        lines.append(mapping(overlay))
    maps.write_text("".join(lines) + "3000-4000 rw-p 00000000 00:00 0 [heap]\n")
    assert len(endpoints._native_library_mappings(library, maps=maps)) == 1


def test_replaced_selected_file_cannot_attest_an_old_mapping(
    library: Path, tmp_path: Path
) -> None:
    maps = tmp_path / "maps"
    maps.write_text(mapping(library, inode=library.stat().st_ino + 1))
    with pytest.raises(RuntimeError, match="not mapped at its current identity"):
        endpoints._native_library_mappings(library, maps=maps)


def test_unavailable_linux_maps_fail_closed(library: Path, tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError, match="Cannot verify native library process mappings"
    ):
        endpoints._native_library_mappings(library, maps=tmp_path / "missing")


def test_non_linux_keeps_selected_library_contract(
    library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(endpoints.platform, "system", lambda: "Darwin")
    assert (
        endpoints._native_library_mappings(library, maps=tmp_path / "missing") is None
    )


def test_mixed_owners_prevent_a_selected_library_receipt(
    library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(path: Path) -> list[dict]:
        assert path == library
        raise RuntimeError("Multiple distinct native libraries are mapped")

    monkeypatch.setattr(endpoints, "_native_library_mappings", reject)
    calculator = SimpleNamespace(_library=SimpleNamespace(_name=str(library)))
    with pytest.raises(RuntimeError, match="Multiple distinct native libraries"):
        endpoints.binary_identity(calculator)


def test_selected_receipt_retains_hash_source_and_verified_mappings(
    library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings = [
        {"device": "1:2", "inode": library.stat().st_ino, "paths": [str(library)]}
    ]
    monkeypatch.setattr(endpoints, "_native_library_mappings", lambda path: mappings)

    def source_identity() -> bytes:
        return b"authenticated-source"

    calculator = SimpleNamespace(
        _library=SimpleNamespace(
            _name=str(library), generativeqc_get_source_identity=source_identity
        )
    )
    identity = endpoints.binary_identity(calculator)
    assert identity == {
        "library_path": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_identity": "authenticated-source",
        "native_library_mappings": mappings,
    }

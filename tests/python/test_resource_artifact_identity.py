"""Provenance follows a retained native loader image, not a mutable pathname."""

from __future__ import annotations

import ctypes
import hashlib
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from generativeqc import profiles
from generativeqc.resources_hf import _cuda_library_identity

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def digest(library: object) -> str:
    """Read the same image identity included in CUDA resource plans."""
    return _cuda_library_identity(library)["native_binary_sha256"]


def build_image(folder: Path, compiler: NativeCxx, value: int) -> Path:
    """Build a real host-only loader fixture through the verified compiler cache."""
    folder.mkdir()
    source = folder / "image.cpp"
    source.write_text(f'extern "C" int image_value() {{ return {value}; }}\n')
    return compiler.build_shared([source], folder / "image.so")


def test_wrappers_reuse_retained_loaded_image(
    tmp_path: Path, required_native_cxx: NativeCxx, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = build_image(tmp_path / "first", required_native_cxx, 17)
    original = ctypes.CDLL(str(artifact))
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert original.image_value() == 17
    assert digest(original) == expected

    def forbidden(path: Path) -> str:
        raise AssertionError("retained loaded image was rehashed")

    monkeypatch.setattr(profiles, "file_hash", forbidden)
    another = ctypes.CDLL(str(artifact))
    another._generativeqc_profile_diagnostics = {"identity": "another profile"}
    assert another.image_value() == 17
    assert digest(another) == expected
    assert _cuda_library_identity(another)["profile_identity"] == "another profile"
    assert _cuda_library_identity(original)["profile_identity"] is None


def test_replaced_path_does_not_relabel_an_existing_loaded_image(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    artifact = build_image(tmp_path / "first", required_native_cxx, 17)
    replacement = build_image(tmp_path / "second", required_native_cxx, 23)
    original, newer = ctypes.CDLL(str(artifact)), ctypes.CDLL(str(replacement))
    original_digest, newer_digest = digest(original), digest(newer)
    assert original_digest != newer_digest
    replacement.replace(artifact)
    # The loader may keep the already-loaded image for the old pathname. Its
    # actual returned symbol, rather than the current file, is our oracle.
    another = ctypes.CDLL(str(artifact))
    assert (
        digest(another)
        == {17: original_digest, 23: newer_digest}[another.image_value()]
    )
    assert digest(original) == original_digest


@pytest.mark.parametrize("replace_file", (False, True))
def test_adapter_without_loader_lifetime_rehashes_changed_file(
    tmp_path: Path, replace_file: bool
) -> None:
    artifact = tmp_path / "library.so"
    artifact.write_bytes(b"old binary")
    previous = digest(SimpleNamespace(_name=str(artifact)))
    before = artifact.stat()
    if replace_file:
        replacement = tmp_path / "replacement.so"
        replacement.write_bytes(b"new binary")
        replacement.replace(artifact)
    else:
        artifact.write_bytes(b"new binary")
    os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
    current = digest(SimpleNamespace(_name=str(artifact)))
    assert current == hashlib.sha256(b"new binary").hexdigest()
    assert current != previous

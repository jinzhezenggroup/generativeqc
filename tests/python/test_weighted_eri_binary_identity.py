"""Small-file integrity regressions; no compiler or native library is required."""

from __future__ import annotations

import os
import typing
from types import SimpleNamespace

import pytest
from generativeqc_compiler.integral import weighted_eri_execute as runtime

if typing.TYPE_CHECKING:
    from pathlib import Path


def _artifact(path: Path, *, packaged: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        backend="cpu",
        native=SimpleNamespace(
            library=path,
            metadata={
                "identity": {
                    "schema": "generativeqc.weighted-packaged.v1"
                    if packaged
                    else "jit",
                },
            },
        ),
    )


def test_unchanged_packaged_binary_is_hashed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "library.so"
    path.write_bytes(b"unchanged library")
    original = runtime.file_hash
    calls = []

    def counted(candidate: Path) -> str:
        calls.append(candidate)
        return original(candidate)

    monkeypatch.setattr(runtime, "file_hash", counted)
    expected = original(path)
    for _ in range(5):
        assert runtime._artifact_binary_hash(_artifact(path)) == expected
    assert len(calls) == 1


@pytest.mark.parametrize("replacement", (False, True))
def test_file_generation_change_invalidates_packaged_hash(
    tmp_path: Path, replacement: bool
) -> None:
    path = tmp_path / "library.so"
    path.write_bytes(b"old built library")
    previous = path.stat()
    digest = runtime._artifact_binary_hash(_artifact(path))
    if replacement:
        alternate = tmp_path / "replacement.so"
        alternate.write_bytes(b"new built library")
        # Same size and mtime are insufficient cache identity after replacement.
        os.utime(alternate, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        os.replace(alternate, path)
    else:
        path.write_bytes(b"new built library")
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    current = runtime._artifact_binary_hash(_artifact(path))
    assert current == runtime.file_hash(path)
    assert current != digest


def test_cached_path_must_still_exist(tmp_path: Path) -> None:
    path = tmp_path / "library.so"
    path.write_bytes(b"library")
    runtime._packaged_library_hash(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        runtime._packaged_library_hash(path)


def test_change_during_hashing_is_rejected_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "library.so"
    path.write_bytes(b"old library")
    original = runtime.file_hash

    def changed(candidate: Path) -> str:
        digest = original(candidate)
        candidate.write_bytes(b"new and differently sized library")
        return digest

    monkeypatch.setattr(runtime, "file_hash", changed)
    with pytest.raises(ValueError, match="changed during verification"):
        runtime._packaged_library_hash(path)
    monkeypatch.setattr(runtime, "file_hash", original)
    assert runtime._packaged_library_hash(path) == original(path)


def test_standalone_jit_keeps_uncached_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "jit.so"
    path.write_bytes(b"standalone")
    original = runtime.file_hash
    calls = []

    def counted(candidate: Path) -> str:
        calls.append(candidate)
        return original(candidate)

    monkeypatch.setattr(runtime, "file_hash", counted)
    monkeypatch.setattr(
        runtime,
        "_packaged_library_hash",
        lambda _: pytest.fail("standalone JIT used packaged hash caching"),
    )
    artifact = _artifact(path, packaged=False)
    for _ in range(2):
        assert runtime._artifact_binary_hash(artifact) == original(path)
    assert len(calls) == 2


def test_stale_packaged_artifact_is_rejected_before_cdll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "library.so"
    path.write_bytes(b"old library")
    expected = runtime._packaged_library_hash(path)
    path.write_bytes(b"rebuilt library with changed bytes")
    native = SimpleNamespace(
        library=path,
        metadata={
            "identity": {"schema": "generativeqc.weighted-packaged.v1"},
            "key": "a" * 64,
            "binary_sha256": expected,
        },
    )
    # Isolate binary admission from the separately tested mathematical metadata.
    monkeypatch.setattr(runtime.CompiledWeightedEri, "validate", lambda _: None)
    artifact = runtime.CompiledWeightedEri(
        native, None, None, (0,), ((0,) * 12,), "cpu", "fixture-program"
    )
    monkeypatch.setattr(
        runtime.ct, "CDLL", lambda _: pytest.fail("stale binary reached CDLL")
    )
    with pytest.raises(ValueError, match="binary hash mismatch"):
        runtime.PreparedWeightedEri(artifact)

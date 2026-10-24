"""Native input binding retains complete-byte identity and external integrity gates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_runtime import CudaArtifact
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.resources import ResourceBudget
from generativeqc_compiler.dft import cuda


def basis() -> SimpleNamespace:
    """Supply only the host resource and ownership contract consumed before load."""
    return SimpleNamespace(
        nao=32, natom=3, numeric_bytes=8192, packed=np.zeros(256), identity="test-basis"
    )


class StopBeforeDevice(RuntimeError):
    """A loader sentinel prevents device execution in the host gate."""


def admitted_owner(monkeypatch: pytest.MonkeyPatch, artifact: Any) -> cuda.CudaGrid:
    """Execute real resource/identity admission and stop at the native loader."""

    def loader(path: str) -> None:
        assert Path(path) == artifact.library.resolve()
        raise StopBeforeDevice

    monkeypatch.setattr(cuda.ct, "CDLL", loader)
    owner = object.__new__(cuda.CudaGrid)
    with pytest.raises(StopBeforeDevice):
        owner.__init__(basis(), artifact, tile_points=7)
    return owner


def test_native_request_binds_current_complete_binary_per_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"initial binary")
    request = cuda.NativeGridLibrary(library, "sm_120")
    # Construction does not establish an artifact identity. Replacement before
    # owner admission must be visible even when the request object is reused.
    for contents in (b"first admitted binary", b"changed admitted binary"):
        library.write_bytes(contents)
        owner = admitted_owner(monkeypatch, request)
        digest = hashlib.sha256(contents).hexdigest()
        identity = {
            "schema": "generativeqc.native-grid-aot.v1",
            "target": {"architecture": "sm_120"},
        }
        assert isinstance(owner.artifact, CudaArtifact)
        assert owner.artifact.library == library.resolve()
        assert owner.artifact.metadata == {
            "identity": identity,
            "binary_sha256": digest,
            "key": canonical_hash({"identity": identity, "binary_sha256": digest}),
            "artifact_kind": "native-build-aot",
        }


def test_external_artifact_still_rejects_changed_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"authenticated binary")
    artifact = cuda.NativeGridLibrary(library, "sm_120").bind()
    library.write_bytes(b"different current binary")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("changed artifact reached CUDA loader")

    monkeypatch.setattr(cuda.ct, "CDLL", forbidden)
    with pytest.raises(ValueError, match="CUDA grid binary hash mismatch"):
        cuda.CudaGrid(basis(), artifact, tile_points=7)


@pytest.mark.parametrize("native", [False, True])
def test_missing_binary_is_rejected_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
) -> None:
    library = tmp_path / "missing.so"
    artifact = (
        cuda.NativeGridLibrary(library, "sm_120")
        if native
        else CudaArtifact(library, {"binary_sha256": "0" * 64})
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("missing artifact reached CUDA loader")

    monkeypatch.setattr(cuda.ct, "CDLL", forbidden)
    with pytest.raises(FileNotFoundError):
        cuda.CudaGrid(basis(), artifact, tile_points=7)


def test_native_binding_preserves_preflight_budget_rejection(tmp_path: Path) -> None:
    # A missing binary would fail if binding or loading ran before admission.
    request = cuda.NativeGridLibrary(tmp_path / "missing.so", "sm_120")
    with pytest.raises(MemoryError):
        cuda.CudaGrid(basis(), request, resource_budget=ResourceBudget(host_bytes=1))

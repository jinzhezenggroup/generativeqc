"""Host-execute the production snapshot constructor with a native ABI stand-in."""

from __future__ import annotations

import ast
import builtins
import ctypes as ct
import typing
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _constructor() -> CodeType:
    path = ROOT / "python/vibeqc/_ks_snapshot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeKsSnapshot"
    )
    init = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    module = compile(ast.Module(body=[init], type_ignores=[]), str(path), "exec")
    return next(
        value
        for value in module.co_consts
        if isinstance(value, CodeType) and value.co_name == "__init__"
    )


@pytest.mark.parametrize(
    ("code", "wire_version", "work_version", "accepted"),
    [
        (0x30001, 4, 4, True),
        (0x30001, 3, 4, False),
        (0x30001, 7, 7, True),
        (0x30001, 4, 7, False),
        (0, 1, 7, True),
        (1, 1, 7, True),
        (2, 1, 7, True),
        (3, 2, 7, True),
        (4, 3, 7, True),
        (0x10001, 4, 7, True),
        (0x10001, 7, 7, False),
    ],
)
def test_snapshot_domain_is_shared_and_rejects_mismatch(
    code: int, wire_version: int, work_version: int, accepted: bool
) -> None:
    copied: list[int] = []
    destroyed: list[int] = []

    def create(
        batch: typing.Any,
        index: int,
        handle: typing.Any,
        metadata: typing.Any,
        count: int,
    ) -> int:
        assert index == 0 and count == 16
        ct.cast(handle, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(42)
        metadata[0] = 2  # CPU wire representation, independent of XC domain.
        metadata[6] = code
        metadata[7] = wire_version
        metadata[12] = 2**64 - 1
        metadata[15] = 0
        return 0

    def copy(batch: typing.Any, handle: int, values: typing.Any, count: int) -> int:
        copied.append(handle)
        return 0

    def destroy(handle: int) -> None:
        destroyed.append(handle)

    library = SimpleNamespace(
        vibeqc_ks_snapshot_create_v1=create,
        vibeqc_ks_snapshot_check_v1=lambda *args: 0,
        vibeqc_ks_snapshot_copy_v1=copy,
        vibeqc_ks_snapshot_destroy_v1=destroy,
    )

    class Batch:
        system_count = 1
        _batch = None
        _context = None
        _calculator = SimpleNamespace(_method_name="test-selector")
        _library = library

        def _ensure_open(self) -> None:
            pass

    class Snapshot:
        _handle: int
        _library: typing.Any
        backend: str

        def close(self) -> None:
            if self._handle:
                self._library.vibeqc_ks_snapshot_destroy_v1(self._handle)
                self._handle = 0

    def check(library: typing.Any, status: int, **kwargs: typing.Any) -> None:
        assert status == 0

    namespace = {
        "__builtins__": builtins.__dict__,
        "ct": ct,
        "typing": typing,
        "np": np,
        "PreparedBatch": Batch,
        "_native": SimpleNamespace(check=check),
        "native_xc_functional_code": lambda method: code,
        "AUTOMATIC_FUNCTIONAL_CODE_BASE": 0x30000,
        "LIBXC_WORK_DOMAIN_VERSION": work_version,
        "immutable": lambda value: value,
    }
    initialize = FunctionType(_constructor(), namespace)
    snapshot = Snapshot()
    if accepted:
        initialize(snapshot, Batch(), 0)
        assert snapshot.backend == "cpu"
        assert copied == [42] and destroyed == []
        snapshot.close()
        assert destroyed == [42]
    else:
        with pytest.raises(NotImplementedError, match="snapshot/domain version"):
            initialize(snapshot, Batch(), 0)
        assert copied == [] and destroyed == [42]
        assert snapshot._handle == 0


def test_snapshot_imports_the_producer_work_domain_version() -> None:
    path = ROOT / "python/vibeqc/_ks_snapshot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "vibeqc_compiler.xc.libxc_work"
        and any(alias.name == "LIBXC_WORK_DOMAIN_VERSION" for alias in node.names)
        for node in tree.body
    )

"""Exercise the geometry-only reset protocol without loading a CUDA library."""

from __future__ import annotations

import ast
import ctypes as ct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
COUNTERS = (
    "used",
    "pending_primitive_records",
    "primitive_pages",
    "primitive_page_peak_records",
    "bulk_pack_chunks",
    "bulk_packed_descriptors",
    "scalar_packed_descriptors",
)


def _owner() -> ast.ClassDef:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "_CudaSources"
    )


def _reset() -> Callable[..., None]:
    method = next(
        node
        for node in _owner().body
        if isinstance(node, ast.FunctionDef) and node.name == "reset_geometry"
    )
    namespace = {"_ptr": lambda value: value}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, "<actual reset_geometry>", "exec"), namespace)  # noqa: S102
    return namespace["reset_geometry"]


@pytest.mark.parametrize("fail", [False, True])
def test_geometry_reset_clears_pages_and_passes_only_centers(fail: bool) -> None:
    calls = []
    failure = RuntimeError("native reset failed")

    def call(*args: object) -> None:
        calls.append(args)
        if fail:
            raise failure

    centers, handle = object(), object()
    streams = {11, 22}
    owner = SimpleNamespace(
        centers=centers,
        handle=handle,
        borrowed_streams=streams,
        _call=call,
        **dict.fromkeys(COUNTERS, 17),
    )
    reset = _reset()
    if fail:
        with pytest.raises(RuntimeError) as caught:
            reset(owner, 1e-9)
        assert caught.value is failure
    else:
        reset(owner, 1e-9)
    assert calls == [("stationary_geometry_reset", handle, centers, 1e-9)]
    assert all(getattr(owner, name) == 0 for name in COUNTERS)
    assert owner.borrowed_streams is streams and not streams
    assert owner.centers is centers and owner.handle is handle


def test_geometry_reset_repeated_use_does_not_retain_page_state() -> None:
    calls = []
    owner = SimpleNamespace(
        centers=object(),
        handle=object(),
        borrowed_streams=set(),
        _call=lambda *args: calls.append(args),
        **dict.fromkeys(COUNTERS, 0),
    )
    reset = _reset()
    for attempt in range(2):
        for name in COUNTERS:
            setattr(owner, name, attempt + 1)
        owner.borrowed_streams.add(attempt + 1)
        reset(owner, 0.0)
        assert all(getattr(owner, name) == 0 for name in COUNTERS)
        assert not owner.borrowed_streams
    assert len(calls) == 2 and calls[0] == calls[1]


def test_geometry_reset_ffi_keeps_pointer_and_tolerance_types() -> None:
    init = next(
        node for node in _owner().body if getattr(node, "name", None) == "__init__"
    )
    binding = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "lib.stationary_geometry_reset.argtypes"
    )
    function = SimpleNamespace()
    double_pointer = ct.POINTER(ct.c_double)
    namespace = {
        "lib": SimpleNamespace(stationary_geometry_reset=function),
        "ct": ct,
        "_DOUBLE": double_pointer,
        "tail": [ct.c_char_p, ct.c_size_t],
    }
    module = ast.Module(body=[binding], type_ignores=[])
    exec(compile(module, "<actual ABI>", "exec"), namespace)  # noqa: S102
    assert function.argtypes == [
        ct.c_void_p,
        double_pointer,
        ct.c_double,
        ct.c_char_p,
        ct.c_size_t,
    ]


def _body(source: str, name: str) -> str:
    begin = source.index("{", source.index("int " + name + "("))
    depth = 0
    for end in range(begin, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[begin : end + 1]
    raise AssertionError("unterminated native reset")


def test_native_geometry_reset_preserves_normal_reset_without_weight_uploads() -> None:
    source = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    normal = _body(source, "stationary_reset")
    geometry = _body(source, "stationary_geometry_reset")
    lines = normal.splitlines(keepends=True)
    removed = [
        line
        for line in lines
        if "upload(*p, p->density," in line or "upload(*p, p->weighted_density," in line
    ]
    assert len(removed) == 2
    expected = "".join(line for line in lines if line not in removed)
    assert geometry == expected.replace(
        '"invalid reset"', '"invalid geometry-only reset"'
    )

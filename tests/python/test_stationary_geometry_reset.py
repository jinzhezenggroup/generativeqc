"""Exercise the geometry-only reset protocol without loading a CUDA library."""

from __future__ import annotations

import ast
import ctypes as ct
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

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


@pytest.mark.parametrize("integral_derivatives", [False, True])
def test_geometry_only_owner_does_not_encode_cartesian_derivative_kinds(
    integral_derivatives: bool,
) -> None:
    """Nuclear-only binaries dispatch plain kinds even when the AO basis has f."""
    from generativeqc._stationary_cuda import _component_mode

    init = next(
        node for node in _owner().body if getattr(node, "name", None) == "__init__"
    )
    binding = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "self.component_mode"
    )
    owner = SimpleNamespace(expansions=((("xxx", 1.0),),))
    namespace = {
        "self": owner,
        "integral_derivatives": integral_derivatives,
        "_component_mode": _component_mode,
    }
    module = ast.Module(body=[binding], type_ignores=[])
    exec(compile(module, "<actual component mode>", "exec"), namespace)  # noqa: S102
    assert owner.component_mode is integral_derivatives


@pytest.mark.parametrize("component", ["xx", "xxx"])
def test_geometry_only_nuclear_dispatch_matches_actual_primitive(
    component: str, tmp_path: Path
) -> None:
    """Host-execute the emitted nuclear dispatcher with the real owner's kind."""
    from generativeqc._stationary_cuda import _component_mode
    from generativeqc_compiler.integral.first_derivative_native import (
        emit_first_derivative_cuda,
    )
    from generativeqc_compiler.integral.first_derivative_schedule import (
        derivative_binding,
    )
    from generativeqc_compiler.method.stationary_cuda import (
        encode_stationary_derivative_kind,
    )

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    init = next(
        node for node in _owner().body if getattr(node, "name", None) == "__init__"
    )
    binding = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "self.component_mode"
    )
    nuclear = next(
        node for node in _owner().body if getattr(node, "name", None) == "nuclear"
    )
    calls = []
    owner = SimpleNamespace(
        expansions=(((component, 1.0),),),
        kinds={("nuclear", ()): 0},
        handle=object(),
        flush=lambda: None,
        _call=lambda *args: calls.append(args),
    )
    namespace = {
        "self": owner,
        "integral_derivatives": False,
        "_component_mode": _component_mode,
        "typing": SimpleNamespace(Any=object),
        "encode_stationary_derivative_kind": encode_stationary_derivative_kind,
        "derivative_binding": derivative_binding,
    }
    exec(  # noqa: S102
        compile(
            ast.Module(body=[binding, nuclear], type_ignores=[]),
            "<actual nuclear>",
            "exec",
        ),
        namespace,
    )
    namespace["nuclear"](owner, 0, 1, [8.0, 1.0])
    assert len(calls) == 1 and calls[0][0] == "stationary_nuclear"
    kind = calls[0][2]
    # This composite compiles only the plain nuclear primitive for every AO basis.
    composite = (ROOT / "python/generativeqc/_stationary_composite_cuda.py").read_text()
    calls_ast = [
        node
        for node in ast.walk(ast.parse(composite))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_CudaSources"
    ]
    assert len(calls_ast) == 2
    for call in calls_ast:
        assert any(
            k.arg == "integral_derivatives"
            and isinstance(k.value, ast.Constant)
            and k.value.value is False
            for k in call.keywords
        )
    (tmp_path / "cuda_runtime.h").write_text("#include <cmath>\n")
    source = tmp_path / "nuclear_dispatch.cpp"
    source.write_text(
        "#define __device__\n#define __noinline__\n#include <cassert>\n"
        + emit_first_derivative_cuda((("nuclear", ()),))
        + f"""
int main() {{
  double e[] = {{8., 1.}}, c[] = {{0., 0., 0., 0., 0., 2.}}, out[12]{{}};
  assert(first_derivative({kind}U, e, c, out));
  assert(std::abs(out[2] - 2.) < 1e-12 && std::abs(out[5] + 2.) < 1e-12);
}}
"""
    )
    executable = tmp_path / "nuclear_dispatch"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


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

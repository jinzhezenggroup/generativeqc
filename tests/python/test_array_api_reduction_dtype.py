"""Explicit finite reduction precision, independently checked before reduction."""

from dataclasses import replace
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import VibeArray, input_array
from generativeqc_compiler.array_api import namespace as ns
from generativeqc_compiler.tensor import autodiff, ir
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.optimize import optimize, prepare_for_backend
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, Symmetry, TensorSpec


def symbolic(shape: tuple[int, ...], dtype: str = "float32") -> VibeArray:
    return input_array(
        "x", TensorSpec(ns._generic_indices(shape), dtype=dtype, role="input")
    )


def with_dtype(name: str, dtype: Any) -> Any:
    """Bind a static dtype in a closure using the existing one-input binder."""
    return lambda x: getattr(xp, name)(x, dtype=dtype)


@pytest.mark.parametrize("name", ("sum", "prod"))
@pytest.mark.parametrize("source_dtype", ("float32", "float64"))
@pytest.mark.parametrize("target", (None, "float32", "float64"))
@pytest.mark.parametrize("keepdims", (False, True))
@pytest.mark.parametrize(
    "shape,axis",
    (
        ((), None),
        ((), ()),
        ((1,), 0),
        ((2, 3, 5), None),
        ((2, 3, 5), 1),
        ((2, 3, 5), -1),
        ((2, 3, 5), (-1, 0)),
        ((2, 3, 5), ()),
        ((0,), None),
        ((2, 0, 5), 1),
        ((0, 3, 5), 1),
        ((2, 0, 5), (0, 2)),
        ((0, 0), 0),
    ),
)
def test_oracles_identity_serialization_replay(
    name: str,
    source_dtype: str,
    target: str | None,
    keepdims: bool,
    shape: tuple,
    axis: Any,
) -> None:
    strict.set_array_api_strict_flags(api_version="2025.12")
    x = np.random.default_rng(711).uniform(-1.25, 1.25, shape).astype(source_dtype)
    if x.ndim:
        x = x[..., ::-1]
    before = x.copy()
    dtype = None if target is None else getattr(xp, target)
    expected = getattr(np, name)(
        x.astype(target or source_dtype), axis=axis, keepdims=keepdims
    )
    independent = np.asarray(
        getattr(strict, name)(
            strict.asarray(x.copy()),
            axis=axis,
            keepdims=keepdims,
            dtype=None if target is None else getattr(strict, target),
        )
    )
    function = lambda x: getattr(xp, name)(x, axis=axis, dtype=dtype, keepdims=keepdims)
    source = xp.compile(function, backend="reference")
    try:
        program = source.lower(x)
        restored = Program.loads(program.dumps())
        assert restored.dumps() == program.dumps() == source.lower(x).dumps()
        if target is None or target == source_dtype:
            plain = xp.compile(
                lambda x: getattr(xp, name)(x, axis=axis, keepdims=keepdims),
                backend="reference",
            )
            try:
                assert plain.lower(x).dumps() == program.dumps()
            finally:
                plain.close()
        for result in (
            source(x),
            source(x),
            function(xp.asarray(x)),
            execute(restored, {"x": x}).outputs["output"],
            execute(optimize(program), {"x": x}).outputs["output"],
        ):
            assert result.shape == expected.shape and result.dtype == expected.dtype
            tol = 4e-6 if expected.dtype == np.dtype("float32") else 4e-13
            np.testing.assert_allclose(result, expected, rtol=tol, atol=1e-15)
            np.testing.assert_allclose(result, independent, rtol=tol, atol=1e-15)
            assert not np.shares_memory(result, x)
        np.testing.assert_array_equal(x, before)
    finally:
        source.close()


@pytest.mark.parametrize(
    "name,source_dtype,target,values",
    (
        ("sum", "float32", "float64", [2**24, 1, -(2**24)]),
        ("sum", "float64", "float32", [2**24 + 1, -(2**24), 0]),
        ("prod", "float32", "float64", [1.1, -1.3, 0.7, 1.2, -0.9]),
        ("prod", "float64", "float32", [1 + 2**-25, 1 + 2**-25, 1 + 2**-25]),
    ),
)
def test_cast_before_discriminating_values(
    name: str, source_dtype: str, target: str, values: list
) -> None:
    x = np.asarray(values, dtype=source_dtype)
    converted = np.asarray([np.dtype(target).type(v) for v in x], dtype=target)
    expected = getattr(np, name)(converted)
    late_cast = np.asarray(getattr(np, name)(x), dtype=target)
    assert expected != late_cast
    source = xp.compile(
        lambda x: getattr(xp, name)(x, dtype=getattr(xp, target)), backend="reference"
    )
    try:
        actual = source(x)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(
            actual, expected, rtol=2e-7 if target == "float32" else 1e-15, atol=0
        )
        assert actual != late_cast
    finally:
        source.close()


@pytest.mark.parametrize("name", ("sum", "prod"))
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.dtype("float32"), "float64")
)
def test_public_dtype_normalization(name: str, dtype: Any) -> None:
    source = xp.compile(
        lambda x: getattr(xp, name)(x, dtype=dtype), backend="reference"
    )
    try:
        assert source(np.array([1.25, 2.5])).dtype == np.dtype(dtype)
    finally:
        source.close()


@pytest.mark.parametrize("name", ("sum", "prod"))
def test_metadata_dtype_and_validation_fail_closed(name: str) -> None:
    base = symbolic((2, 2)).node.spec
    specs = [
        replace(base, representation="spin_orbital"),
        replace(base, symmetries=(Symmetry((1, 0)),)),
    ]
    specs += [
        TensorSpec((Index("i", IndexSpace("annotated", kind, 2)),), role="input")
        for kind in ("ao", "occupied", "virtual", "auxiliary", "spin")
    ]
    specs += [
        TensorSpec(
            (Index("i", IndexSpace("_array_extent_2", "matrix", 2, "alpha")),),
            role="input",
        )
    ]
    for spec in specs:
        for target in ("float32", "float64"):
            for axis in (None, ()):
                with pytest.raises(
                    (TypeError, ValueError), match="scientific annotations"
                ):
                    getattr(ns, name)(
                        VibeArray(ir.input_tensor("x", spec)), axis=axis, dtype=target
                    )
    for bad in (True, int, complex, "int64", "bool", "complex128", object(), [], {}):
        source = xp.compile(with_dtype(name, bad), backend="reference")
        try:
            with pytest.raises((TypeError, ValueError)):
                source(np.ones(3))
        finally:
            source.close()
    for source_dtype in ("float32", "float64"):
        for axis in (True, [0], (True,), (0, 0), (0, -1), 2):
            with pytest.raises((TypeError, ValueError)):
                getattr(ns, name)(
                    symbolic((3,), source_dtype), dtype="float64", axis=axis
                )
        with pytest.raises(TypeError, match="keepdims"):
            getattr(ns, name)(symbolic((3,), source_dtype), dtype="float64", keepdims=1)
    with pytest.raises(TypeError, match="real float"):
        getattr(ns, name)(symbolic((3,), "bool"), dtype="float64")


@pytest.mark.parametrize("name", ("sum", "prod"))
@pytest.mark.parametrize(
    "source_dtype,target", (("float32", "float64"), ("float64", "float32"))
)
@pytest.mark.parametrize("zeros", (0, 1, 2))
def test_cast_first_order_convention(
    name: str, source_dtype: str, target: str, zeros: int
) -> None:
    # Canonical cast AD differentiates the real map and casts tangents/adjoints;
    # this is not a claim that IEEE rounding is everywhere differentiable.
    x = np.array([[1.25, -0.5, 2.0], [-1.5, 0.75, 0.5]], dtype=source_dtype)
    if zeros:
        x[0, :zeros] = 0
    direction = np.full_like(x, 0.25)
    weight = np.array([0.5, -0.75], dtype=target)
    derivative = np.ones(x.shape, dtype=target)
    if name == "prod":
        for row, col in np.ndindex(x.shape):
            derivative[row, col] = np.prod(
                [np.dtype(target).type(x[row, j]) for j in range(3) if j != col]
            )
    tangent = np.sum(derivative * direction.astype(target), axis=1)
    adjoint = (derivative * weight[:, None]).astype(source_dtype)
    source = xp.compile(
        lambda x: getattr(xp, name)(x, axis=1, dtype=getattr(xp, target)),
        backend="reference",
        differentiable=("x",),
    )
    try:
        program = source.lower(x)
        jvp = autodiff.jvp(program, {"x": x}, {"x": direction}).output_tangents[
            "output"
        ]
        vjp = autodiff.vjp(program, {"x": x}, {"output": weight}).input_cotangents["x"]
        assert jvp.dtype == np.dtype(target) and vjp.dtype == x.dtype
        np.testing.assert_array_equal(jvp, tangent)
        np.testing.assert_array_equal(vjp, adjoint)
        assert autodiff.dot_test(
            program,
            {"x": x},
            {"x": direction},
            {"output": weight},
            rtol=1e-6,
            atol=1e-6,
        ).passed
    finally:
        source.close()


@pytest.mark.parametrize("name", ("sum", "prod"))
def test_cast_storage_preflight_and_backend_limits(name: str) -> None:
    # Symbolic-only dimensions: full target cast buffer must be checked even
    # when the reduced output is tiny. No large arrays/index tuples are built.
    with pytest.raises(ValueError, match="graph budget"):
        getattr(ns, name)(symbolic((34_000_000,)), dtype="float64")
    with pytest.raises(ValueError, match="graph budget"):
        getattr(ns, name)(symbolic((18_000_000,)), axis=(), dtype="float64")
    if name == "prod":
        assert ns.prod(symbolic((3, 3_500_000)), axis=0).shape == (3_500_000,)
        with pytest.raises(ValueError, match="graph budget"):
            ns.prod(symbolic((3, 3_500_000)), axis=0, dtype="float64")
    for shape, axes in (
        ((3, 5, 7), (0, 2)),
        ((2, 9), (1,)),
        ((1,), ()),
        ((0, 3), (0,)),
        ((0, 3), (1,)),
    ):
        value = symbolic(shape)
        result = getattr(ns, name)(value, axis=axes, dtype="float64", keepdims=True)
        program = Program({"output": result.node})
        new = [n for n in program.live_nodes if n is not value.node]
        actual_bytes = sum(n.spec.size * n.spec.itemsize for n in new)
        assert actual_bytes <= 256 * 1024 * 1024 and len(new) <= 512
        execution = execute(program, {"x": np.ones(shape, dtype="float32")})
        # Public interpreter storage admission includes source and detached
        # output; compare its independently produced plan with frontend bounds.
        external_bytes = value.size * 4 + result.size * 8
        if name == "prod":
            planned, nodes, positions = ns._prod_plan(
                shape, axes, 8, result.shape, cast_bytes=value.size * 8
            )
            assert actual_bytes <= planned and len(new) <= nodes
            assert execution.logical_retained_bytes <= planned + external_bytes
            assert (
                sum(len(n.attrs["positions"]) for n in new if n.op == "gather")
                <= positions
            )
        else:
            planned = (value.size + 2 * result.size) * 8
            assert execution.logical_retained_bytes <= planned + external_bytes
    for limits in ({"max_bytes": 1},):
        source = xp.compile(
            lambda x: getattr(xp, name)(x, dtype=xp.float64),
            backend="reference",
            **limits,
        )
        try:
            with pytest.raises(ValueError, match="budget"):
                source(np.array([1, 2, 3], dtype="float32"))
        finally:
            source.close()
    source = xp.compile(
        lambda x: getattr(xp, name)(x, dtype=xp.float64), backend="cpu", max_nodes=1
    )
    try:
        with pytest.raises(ValueError, match="budget"):
            source(np.array([1.0, 2.0, 3.0]))
    finally:
        source.close()


@pytest.mark.parametrize("name", ("sum", "prod"))
def test_finite_results_and_signed_zero(name: str) -> None:
    for x in (
        np.array([1e300, 1e300]),
        np.array([np.inf, 1.0]),
        np.array([np.nan, 1.0]),
    ):
        source = xp.compile(
            lambda x: getattr(xp, name)(x, dtype=xp.float32), backend="reference"
        )
        try:
            with pytest.raises(ValueError, match="finite"):
                source(x)
        finally:
            source.close()
    for dtype in ("float32", "float64"):
        source = xp.compile(with_dtype(name, getattr(xp, dtype)), backend="reference")
        try:
            for x in (
                np.array([-0.0]),
                np.array([-0.0, -2, 3]),
                np.array([-0.0, 0.0, -2, 3, 1]),
            ):
                expected = getattr(np, name)(x.astype(dtype))
                assert np.signbit(source(x)) == np.signbit(expected)
            value = np.finfo(dtype).max
            with pytest.raises(ValueError, match="finite"):
                source(np.array([value, value], dtype=dtype))
        finally:
            source.close()


@pytest.mark.parametrize("name", ("sum", "prod"))
def test_unqualified_native_and_cuda_before_executor(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc.experimental._array_api_execution import _Executor
    from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
    from generativeqc_compiler.tensor.cpu import describe_cpu_compilation

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unqualified capture reached compiler/device work")

    monkeypatch.setattr(CppCompilerAdapter, "__init__", forbidden)
    for source_dtype, target in (
        ("float32", "float64"),
        ("float64", "float32"),
        ("float32", "float32"),
    ):
        source = xp.compile(with_dtype(name, getattr(xp, target)), backend="reference")
        try:
            with pytest.raises(ValueError, match="float64|cast"):
                describe_cpu_compilation(source.lower(np.ones(3, dtype=source_dtype)))
        finally:
            source.close()
        native = xp.compile(with_dtype(name, getattr(xp, target)), backend="cpu")
        try:
            with pytest.raises(ValueError, match="float64|cast"):
                native(np.ones(3, dtype=source_dtype))
        finally:
            native.close()

    monkeypatch.setattr(_Executor, "_build", forbidden)
    for target in (xp.float32, xp.float64):
        source = xp.compile(with_dtype(name, target), backend="cuda")
        try:
            x = np.ones(3)
            for action in (
                lambda source=source, x=x: source.lower(x),
                lambda source=source, x=x: source(x),
                lambda source=source, x=x: xp.jvp(source, (x,), (x,)),
                lambda source=source, x=x: xp.vjp(source, x),
            ):
                with pytest.raises(ValueError, match="not qualified.*cuda"):
                    action()
            gradient = xp.grad(source)
            try:
                with pytest.raises(ValueError, match="not qualified.*cuda"):
                    gradient(x)
            finally:
                gradient.close()
        finally:
            source.close()
    # Raw compiler primitives and reference-lowered programs keep their original
    # backend admission. Public dtype requests do not persist a backend marker.
    raw = Program(
        {"output": getattr(ns, name)(symbolic((3,), "float64"), dtype="float64").node}
    )
    prepare_for_backend(raw, backend="cuda")


@pytest.mark.parametrize("name", ("sum", "prod"))
def test_native_fp64_same_dtype_forward_ad(
    name: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native explicit dtype used reference execution/AD")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)
    for zeros in (0, 1, 2):
        x = np.array([[1.25, -0.5, 2], [-1.5, 0.75, 0.5]])
        if zeros:
            x[0, :zeros] = 0
        d = np.full_like(x, 0.25)
        derivative = np.ones_like(x)
        if name == "prod":
            for i, j in np.ndindex(x.shape):
                derivative[i, j] = np.prod([x[i, k] for k in range(3) if j != k])
        expected = getattr(np, name)(x, axis=1, keepdims=True)
        tangent = np.sum(derivative * d, axis=1, keepdims=True)
        source = xp.compile(
            lambda x: getattr(xp, name)(x, axis=1, keepdims=True, dtype=xp.float64),
            backend="cpu",
            cache=tmp_path,
        )
        try:
            for _ in range(2):
                np.testing.assert_array_equal(source(x), expected)
            value, actual = xp.jvp(source, (x,), (d,))
            np.testing.assert_array_equal(value, expected)
            np.testing.assert_array_equal(actual, tangent)
            _, pullback = xp.vjp(source, x, argnums=0)
            try:
                (actual_adjoint,) = pullback(np.full(expected.shape, 0.5))
                np.testing.assert_array_equal(actual_adjoint, derivative * 0.5)
                np.testing.assert_allclose(
                    np.sum(actual * 0.5), np.sum(d * actual_adjoint), rtol=1e-14
                )
            finally:
                pullback.close()
            step = 1e-5
            difference = (
                getattr(np, name)(x + step * d, axis=1, keepdims=True)
                - getattr(np, name)(x - step * d, axis=1, keepdims=True)
            ) / (2 * step)
            np.testing.assert_allclose(difference, tangent, rtol=1e-8, atol=1e-10)
        finally:
            source.close()

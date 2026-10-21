"""Finite variance capture against independent value and analytic AD oracles."""

from dataclasses import replace
from fractions import Fraction
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


def symbolic(shape: tuple[int, ...], dtype: str = "float64") -> VibeArray:
    return input_array(
        "x", TensorSpec(ns._generic_indices(shape), dtype=dtype, role="input")
    )


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize(
    "shape,axis",
    (
        ((), None),
        ((), ()),
        ((1,), 0),
        ((2, 3, 5), None),
        ((2, 3, 5), 1),
        ((2, 3, 5), -1),
        ((2, 3, 5), (0, -1)),
        ((2, 3, 5), (-1, 0)),
        ((2, 3, 5), ()),
        ((0, 3, 5), 1),
        ((2, 0, 5), (0, 2)),
        ((0, 3), ()),
    ),
)
@pytest.mark.parametrize("keepdims", (False, True))
@pytest.mark.parametrize("correction", (0, 0.0, 0.5, -2.5))
def test_independent_oracles_replay_and_serialization(
    dtype: str,
    shape: tuple,
    axis: Any,
    keepdims: bool,
    correction: float,
) -> None:
    strict.set_array_api_strict_flags(api_version="2025.12")
    x = np.random.default_rng(510).uniform(-4, 3, size=shape).astype(dtype)
    if x.ndim:
        x = x[..., ::-1]
    before = x.copy()
    expected = np.var(x, axis=axis, ddof=correction, keepdims=keepdims)
    independent = np.asarray(
        strict.var(
            strict.asarray(x.copy()),
            axis=axis,
            correction=correction,
            keepdims=keepdims,
        )
    )
    function = lambda x: xp.var(x, axis=axis, correction=correction, keepdims=keepdims)
    source = xp.compile(function, backend="reference")
    try:
        program = source.lower(x)
        restored = Program.loads(program.dumps())
        assert program.dumps() == restored.dumps() == source.lower(x).dumps()
        tolerance = 2e-6 if dtype == "float32" else 3e-14
        for actual in (
            source(x),
            source(x),
            function(xp.asarray(x)),
            execute(restored, {"x": x}).outputs["output"],
            execute(optimize(program), {"x": x}).outputs["output"],
        ):
            assert actual.shape == expected.shape and actual.dtype == x.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-14)
            np.testing.assert_allclose(actual, independent, rtol=tolerance, atol=1e-14)
            assert not np.shares_memory(actual, x)
        np.testing.assert_array_equal(x, before)
    finally:
        source.close()


@pytest.mark.parametrize("dtype,offset", (("float32", 2**16), ("float64", 2**30)))
def test_centered_not_moment_formula(dtype: str, offset: int) -> None:
    x = np.array([-2, -1, 0, 1, 2], dtype=dtype) + offset
    source = xp.compile(xp.var, backend="reference")
    try:
        assert source(x) == np.var(x) == 2
        assert np.mean(x * x) - np.mean(x) ** 2 != 2
        for value in (0.0, -0.0, 3.5):
            assert source(np.full(5, value, dtype=dtype)) == 0
    finally:
        source.close()


def analytic(
    x: np.ndarray,
    direction: np.ndarray,
    weights: np.ndarray,
    axes: tuple[int, ...],
    correction: float,
    keepdims: bool,
) -> tuple:
    count = np.prod([x.shape[i] for i in axes], dtype=int)
    derivative = 2 * (x - np.mean(x, axis=axes, keepdims=True)) / (count - correction)
    tangent = np.sum(derivative * direction, axis=axes, keepdims=keepdims)
    target = tuple(1 if i in axes else n for i, n in enumerate(x.shape))
    return tangent, derivative * weights.reshape(target)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize(
    "axes,keepdims,correction",
    (
        ((0, 2), True, 1),
        ((1,), False, 0.5),
        ((), False, -1),
        ((0, 1, 2), False, 0),
    ),
)
@pytest.mark.parametrize("constant", (False, True))
def test_reference_first_order(
    dtype: str, axes: tuple, keepdims: bool, correction: float, constant: bool
) -> None:
    x = np.linspace(-2, 3, 30, dtype=dtype).reshape(2, 3, 5)
    if constant:
        x.fill(2)
    direction = np.linspace(-0.3, 0.7, x.size, dtype=dtype).reshape(x.shape)
    weights = np.full(np.var(x, axis=axes, keepdims=keepdims).shape, 0.7, dtype=dtype)
    tangent, adjoint = analytic(x, direction, weights, axes, correction, keepdims)
    source = xp.compile(
        lambda x: xp.var(x, axis=axes, correction=correction, keepdims=keepdims),
        backend="reference",
        differentiable=("x",),
    )
    try:
        program = source.lower(x)
        tol = 2e-6 if dtype == "float32" else 3e-13
        actual = autodiff.jvp(program, {"x": x}, {"x": direction})
        np.testing.assert_allclose(
            actual.output_tangents["output"], tangent, rtol=tol, atol=tol
        )
        actual_vjp = autodiff.vjp(program, {"x": x}, {"output": weights})
        np.testing.assert_allclose(
            actual_vjp.input_cotangents["x"], adjoint, rtol=tol, atol=tol
        )
        assert autodiff.dot_test(
            program, {"x": x}, {"x": direction}, {"output": weights}, rtol=tol, atol=tol
        ).passed
        step = 0.005 if dtype == "float32" else 1e-5
        difference = (
            np.var(x + step * direction, axis=axes, ddof=correction, keepdims=keepdims)
            - np.var(
                x - step * direction, axis=axes, ddof=correction, keepdims=keepdims
            )
        ) / (2 * step)
        np.testing.assert_allclose(
            difference, tangent, rtol=2e-3, atol=3e-5 if dtype == "float32" else 1e-9
        )
    finally:
        source.close()


@pytest.mark.parametrize(
    "shape,axes,keepdims,correction",
    (
        ((2, 3, 5), (0, 2), True, 1),
        ((2, 3, 5), (1,), False, 0.5),
        ((3,), (0,), False, -2),
        ((3,), (), False, 0),
        ((), (), False, 0),
        ((0, 3), (1,), True, 0),
        ((3, 1), (1,), False, 0),
    ),
)
@pytest.mark.parametrize("constant", (False, True))
def test_native_fp64_no_reference(
    shape: tuple,
    axes: tuple,
    keepdims: bool,
    correction: float,
    constant: bool,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native variance used reference execution/AD")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)
    x = np.linspace(-2, 3, np.prod(shape, dtype=int)).reshape(shape)
    if constant:
        x.fill(2)
    direction = np.linspace(-0.3, 0.7, x.size).reshape(shape)
    expected = np.var(x, axis=axes, ddof=correction, keepdims=keepdims)
    weights = np.full(expected.shape, 0.7)
    tangent, adjoint = analytic(x, direction, weights, axes, correction, keepdims)
    source = xp.compile(
        lambda x: xp.var(x, axis=axes, correction=correction, keepdims=keepdims),
        backend="cpu",
        cache=tmp_path,
    )
    try:
        for _ in range(2):
            np.testing.assert_allclose(source(x), expected, rtol=3e-13, atol=3e-13)
        value, actual = xp.jvp(source, (x,), (direction,))
        np.testing.assert_allclose(value, expected, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(actual, tangent, rtol=3e-13, atol=3e-13)
        _, pullback = xp.vjp(source, x, argnums=0)
        try:
            (actual_adjoint,) = pullback(weights)
            np.testing.assert_allclose(actual_adjoint, adjoint, rtol=3e-13, atol=3e-13)
            np.testing.assert_allclose(
                np.sum(actual * weights),
                np.sum(direction * actual_adjoint),
                rtol=3e-13,
                atol=3e-13,
            )
        finally:
            pullback.close()
    finally:
        source.close()


def test_invalid_static_domains_and_metadata() -> None:
    x = symbolic((2, 2))
    for axis in (True, 1.0, [0], (True,)):
        with pytest.raises(TypeError):
            ns.var(x, axis=axis)
    for axis in (2, -3, (0, 0), (1, -1)):
        with pytest.raises(ValueError):
            ns.var(x, axis=axis)
    for correction in (True, np.float64(0), Fraction(1, 2), "1", x):
        with pytest.raises(TypeError, match="static"):
            ns.var(x, correction=correction)
    for correction in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            ns.var(x, correction=correction)
    for correction in (4, 5, 4.0):
        with pytest.raises(ValueError, match="positive divisor"):
            ns.var(x, correction=correction)
    for shape, axis in (((0,), None), ((0, 0), 0), ((2, 0), 1)):
        with pytest.raises(ValueError, match="nonempty"):
            ns.var(symbolic(shape), axis=axis)
    for keepdims in (1, None, np.bool_(True)):
        with pytest.raises(TypeError, match="keepdims"):
            ns.var(x, keepdims=keepdims)
    with pytest.raises(TypeError, match="real float"):
        ns.var(symbolic((2,), "bool"), axis=())
    base = x.node.spec
    specs = [
        replace(base, representation="spin_orbital"),
        replace(base, symmetries=(Symmetry((1, 0)),)),
    ]
    specs += [
        TensorSpec((Index("i", IndexSpace("tagged", kind, 2)),), role="input")
        for kind in ("ao", "occupied", "virtual", "auxiliary", "spin")
    ]
    specs += [
        TensorSpec(
            (Index("i", IndexSpace("_array_extent_2", "matrix", 2, "alpha")),),
            role="input",
        )
    ]
    for spec in specs:
        for axis in (None, ()):
            with pytest.raises(TypeError, match="scientific annotations"):
                ns.var(input_array("x", spec), axis=axis)
    with pytest.raises(ValueError, match="coefficient"):
        ns.var(x, correction=-(10**400))


def test_nonfinite_intermediates_preserve_eager_semantics() -> None:
    for dtype in ("float32", "float64"):
        maximum = np.finfo(dtype).max
        source = xp.compile(xp.var, backend="reference")
        try:
            for values in (
                [np.inf, 1],
                [np.nan, 1],
                [maximum, maximum],
                [maximum / 4, -maximum / 4],
            ):
                with pytest.raises(ValueError, match="finite"):
                    source(np.array(values, dtype=dtype))
        finally:
            source.close()
    assert np.isnan(xp.var(xp.asarray([1.0]), correction=1))
    assert np.isnan(xp.var(xp.asarray([np.nan, 1.0])))


def test_constant_graph_and_static_correction_identity() -> None:
    x = np.array([1.0, 2.0, 3.0])
    constant = xp.compile(lambda x: xp.var(xp.full((3,), 2.5)), backend="reference")
    try:
        assert constant(x) == 0
        _, tangent = xp.jvp(constant, (x,), (np.ones_like(x),))
        np.testing.assert_array_equal(tangent, 0)
    finally:
        constant.close()

    def factory(c: float) -> Any:
        return lambda x: xp.var(x, correction=c)

    sources = [
        xp.compile(factory(c), backend="reference") for c in (0, 0.0, -0.0, 1, 0.5)
    ]
    try:
        programs = [source.lower(x) for source in sources]
        assert (
            programs[0].logical_hash
            == programs[1].logical_hash
            == programs[2].logical_hash
        )
        assert len({p.logical_hash for p in programs}) == 3
        for c, source in zip((0, 0.0, -0.0, 1, 0.5), sources, strict=True):
            np.testing.assert_allclose(source(x), np.var(x, ddof=c), rtol=3e-14)
    finally:
        for source in sources:
            source.close()


def test_resources_include_zero_byte_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    for shape, axes in (
        ((2, 3, 5), (0, 2)),
        ((0, 3), (1,)),
        ((), ()),
        ((1,) * 63 + (2,), (63,)),
        ((0, 0, 3), (2,)),
    ):
        x = symbolic(shape)
        result = ns.var(x, axis=axes)
        new = [n for n in Program({"out": result.node}).live_nodes if n is not x.node]
        planned, nodes = ns._var_plan(shape, axes, 8)
        assert sum(n.spec.size * n.spec.itemsize for n in new) <= planned
        assert len(new) <= nodes
        if not x.size:
            assert len(new) > 0

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("oversized variance constructed a graph")

    x = symbolic((5_000_000,))
    monkeypatch.setattr(ir, "reduce_sum", forbidden)
    with pytest.raises(ValueError, match="graph budget"):
        ns.var(x)
    monkeypatch.undo()
    for limits in ({"max_bytes": 1}, {"max_nodes": 1}, {"max_work": 0}):
        source = xp.compile(xp.var, backend="cpu", **limits)
        try:
            with pytest.raises(ValueError, match="budget"):
                source(np.arange(15.0))
        finally:
            source.close()


def test_cuda_and_native_f32_rejected_before_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc.experimental._array_api_execution import _Executor

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unqualified variance reached native executor")

    monkeypatch.setattr(_Executor, "_build", forbidden)
    x = np.array([1.0, 2.0, 3.0])
    for function in (xp.var, lambda x: xp.var(x)):
        source = xp.compile(function, backend="cuda")
        gradient = xp.grad(source)
        try:
            for action in (
                lambda source=source: source.lower(x),
                lambda source=source: source(x),
                lambda source=source: xp.jvp(source, (x,), (np.ones_like(x),)),
                lambda source=source: xp.vjp(source, x),
                lambda gradient=gradient: gradient(x),
            ):
                with pytest.raises(ValueError, match="var.*not qualified"):
                    action()
        finally:
            source.close()
            gradient.close()
    monkeypatch.undo()
    source = xp.compile(xp.var, backend="cpu")
    try:
        with pytest.raises(ValueError, match="float64"):
            source(x.astype("float32"))
    finally:
        source.close()
    raw = Program({"out": ns.var(symbolic((3,))).node})
    assert prepare_for_backend(raw, backend="cuda").outputs["out"].spec.shape == ()
    from generativeqc_compiler.array_api.capabilities import capabilities

    matrix = capabilities()["operation_matrix"]["var"]
    assert "first-order" in matrix["ad"]

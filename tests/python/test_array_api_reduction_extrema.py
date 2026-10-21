"""Bounded max/min reductions: independent values, selectors and native AD."""

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
from generativeqc_compiler.tensor.optimize import optimize
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, Symmetry, TensorSpec


def symbolic(shape: tuple[int, ...], dtype: str = "float64") -> VibeArray:
    return input_array(
        "x", TensorSpec(ns._generic_indices(shape), dtype=dtype, role="input")
    )


@pytest.mark.parametrize("name", ("max", "min"))
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
def test_two_oracles_replay_serialization(
    name: str, dtype: str, shape: tuple, axis: Any, keepdims: bool
) -> None:
    x = (
        np.random.default_rng(193)
        .permutation(np.prod(shape, dtype=int))
        .astype(dtype)
        .reshape(shape)
    )
    if x.ndim:
        x = x[..., ::-1]  # genuinely noncontiguous for nontrivial last axes
    before = x.copy()
    function = lambda x: getattr(xp, name)(x, axis=axis, keepdims=keepdims)
    expected = getattr(np, name)(x, axis=axis, keepdims=keepdims)
    independent = np.asarray(
        getattr(strict, name)(strict.asarray(x.copy()), axis=axis, keepdims=keepdims)
    )
    source = xp.compile(function, backend="reference")
    try:
        program = source.lower(x)
        restored = Program.loads(program.dumps())
        assert restored.dumps() == program.dumps() == source.lower(x).dumps()
        for result in (
            source(x),
            source(x),
            function(xp.asarray(x)),
            execute(restored, {"x": x}).outputs["output"],
            execute(optimize(program), {"x": x}).outputs["output"],
        ):
            assert result.shape == expected.shape and result.dtype == x.dtype
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(result, independent)
            assert not np.shares_memory(result, x)
        np.testing.assert_array_equal(x, before)
    finally:
        source.close()


@pytest.mark.parametrize("name", ("max", "min"))
def test_axes_metadata_and_empty_errors(name: str) -> None:
    operation = getattr(ns, name)
    for axis in (True, 1.0, [0], (True,), (0.0,)):
        with pytest.raises(TypeError):
            operation(symbolic((2, 3)), axis=axis)
    for axis in (2, -3, (0, 0), (1, -1)):
        with pytest.raises(ValueError):
            operation(symbolic((2, 3)), axis=axis)
    for axis in (0, -1):
        with pytest.raises(ValueError):
            operation(symbolic(()), axis=axis)
    for keepdims in (0, 1, None, np.bool_(True)):
        with pytest.raises(TypeError, match="keepdims"):
            operation(symbolic((2, 3)), keepdims=keepdims)
    for shape, axis in (((0,), None), ((2, 0, 3), 1), ((0, 0), 0)):
        with pytest.raises(ValueError, match="empty reduction"):
            operation(symbolic(shape), axis=axis)
    with pytest.raises(TypeError, match="real float"):
        operation(symbolic((3,), "bool"), axis=())
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
        for axis in (None, ()):
            with pytest.raises(TypeError, match="scientific annotations"):
                operation(VibeArray(ir.input_tensor("x", spec)), axis=axis)


@pytest.mark.parametrize("name", ("max", "min"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_finite_capture_and_eager_ieee(name: str, dtype: str) -> None:
    source = xp.compile(lambda x: getattr(xp, name)(x), backend="reference")
    try:
        for bad in (np.nan, np.inf, -np.inf):
            x = np.array([1, bad, -2], dtype=dtype)
            with pytest.raises(ValueError, match="finite"):
                source(x)
            np.testing.assert_array_equal(
                getattr(xp, name)(xp.asarray(x)), getattr(np, name)(x)
            )
        limit = np.finfo(dtype)
        x = np.array([-limit.max, -limit.tiny, limit.tiny, limit.max], dtype=dtype)
        np.testing.assert_array_equal(source(x), getattr(np, name)(x))
        with pytest.raises((TypeError, ValueError)):
            source(np.ones(3, dtype=np.int64))
        with pytest.raises((TypeError, ValueError)):
            source(np.ones(3, dtype=np.complex128))
    finally:
        source.close()


@pytest.mark.parametrize("name", ("max", "min"))
def test_static_graph_and_runtime_budgets(name: str) -> None:
    operation = getattr(ns, name)
    for shape in ((65_536,), (2,) * 33, (4096, 4096)):
        with pytest.raises(ValueError, match="graph budget"):
            operation(symbolic(shape))
    # A zero output requires no comparisons or materialized gather maps.
    assert operation(symbolic((0, 100_000)), axis=1).shape == (0,)
    for limits in ({"max_bytes": 1}, {"max_nodes": 1}, {"max_work": 0}):
        source = xp.compile(lambda x: getattr(xp, name)(x), backend="cpu", **limits)
        try:
            with pytest.raises(ValueError, match="budget"):
                source(np.arange(15.0).reshape(3, 5))
        finally:
            source.close()
    # Logarithmic graph construction remains usable for a sizeable vector.
    source = xp.compile(lambda x: getattr(xp, name)(x), backend="reference")
    try:
        x = np.arange(4097.0)[::-1]
        np.testing.assert_array_equal(source(x), getattr(np, name)(x))
    finally:
        source.close()


@pytest.mark.parametrize("name", ("max", "min"))
def test_rank64_and_default_reduction(name: str) -> None:
    x = np.array([-2.0, 5.0]).reshape((1,) * 63 + (2,))
    source = xp.compile(
        lambda x: getattr(xp, name)(x, axis=(-1, 0), keepdims=True), backend="reference"
    )
    try:
        result = source(x)
        assert result.shape == (1,) * 64
        np.testing.assert_array_equal(
            result, getattr(np, name)(x, axis=(-1, 0), keepdims=True)
        )
    finally:
        source.close()
    source = xp.compile(lambda x: getattr(xp, name)(x), backend="reference")
    try:
        np.testing.assert_array_equal(
            source(np.arange(6.0).reshape(2, 3)), getattr(np, name)(np.arange(6.0))
        )
    finally:
        source.close()


def selected_oracle(
    x: np.ndarray,
    direction: np.ndarray,
    weights: np.ndarray,
    axes: tuple[int, ...],
    name: str,
    keepdims: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar index oracle, independent of comparison/where graph lowering."""
    retained = tuple(i for i in range(x.ndim) if i not in axes)
    output_shape = tuple(x.shape[i] for i in retained)
    tangent = np.empty(output_shape, dtype=x.dtype)
    adjoint = np.zeros_like(x)
    flat_weights = weights.reshape(output_shape)
    for out in np.ndindex(output_shape):
        candidates = []
        for reduced in np.ndindex(tuple(x.shape[i] for i in axes)):
            index = [0] * x.ndim
            for i, v in zip(retained, out, strict=True):
                index[i] = v
            for i, v in zip(axes, reduced, strict=True):
                index[i] = v
            candidates.append(tuple(index))
        chosen = (max if name == "max" else min)(candidates, key=lambda i: x[i])
        tangent[out] = direction[chosen]
        adjoint[chosen] = flat_weights[out]
    return tangent.reshape(weights.shape) if keepdims else tangent, adjoint


@pytest.mark.parametrize("name", ("max", "min"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize(
    "axes,keepdims", (((1,), False), ((0, 2), True), ((0, 1, 2), False), ((), True))
)
def test_independent_selected_index_ad(
    name: str, dtype: str, axes: tuple, keepdims: bool
) -> None:
    x = np.random.default_rng(19).permutation(30).astype(dtype).reshape(2, 3, 5) / 4
    direction = np.linspace(-0.7, 0.9, 30, dtype=dtype).reshape(x.shape)
    expected = getattr(np, name)(x, axis=axes, keepdims=keepdims)
    weights = (np.arange(expected.size, dtype=dtype).reshape(expected.shape) + 1) / 7
    tangent, adjoint = selected_oracle(x, direction, weights, axes, name, keepdims)
    function = lambda x: getattr(xp, name)(x, axis=axes, keepdims=keepdims)
    source = xp.compile(function, backend="reference", differentiable=("x",))
    try:
        program = source.lower(x)
        feeds, directions = {"x": x}, {"x": direction}
        forward = autodiff.jvp(program, feeds, directions, allow_comparison_where=True)
        reverse = autodiff.vjp(
            program, feeds, {"output": weights}, allow_comparison_where=True
        )
        tol = 2e-6 if dtype == "float32" else 2e-13
        np.testing.assert_allclose(
            forward.output_tangents["output"], tangent, atol=tol, rtol=tol
        )
        np.testing.assert_allclose(
            reverse.input_cotangents["x"], adjoint, atol=tol, rtol=tol
        )
        assert autodiff.dot_test(
            program,
            feeds,
            directions,
            {"output": weights},
            allow_comparison_where=True,
            atol=tol,
            rtol=tol,
        ).passed
        step = 0.002 if dtype == "float32" else 1e-5
        difference = (
            getattr(np, name)(x + step * direction, axis=axes, keepdims=keepdims)
            - getattr(np, name)(x - step * direction, axis=axes, keepdims=keepdims)
        ) / (2 * step)
        np.testing.assert_allclose(
            difference,
            tangent,
            atol=3e-4 if dtype == "float32" else 1e-10,
            rtol=2e-3 if dtype == "float32" else 2e-9,
        )
    finally:
        source.close()


@pytest.mark.parametrize("name", ("max", "min"))
@pytest.mark.parametrize("backend", ("reference", "cpu"))
def test_ties_signed_zero_and_selected_branch(
    name: str, backend: str, tmp_path: Any
) -> None:
    # Equal extrema at 1 and 2 catch non-adjacent pairings that select index 2.
    x = np.array([[-1.0, 0.0, -0.0, -2.0, -3.0], [-4.0, -5.0, -6.0, -7.0, -8.0]])
    if name == "min":
        x = -x
    direction = np.arange(10.0).reshape(x.shape) + 0.25
    function = lambda x: getattr(xp, name)(x)
    source = xp.compile(function, backend=backend, cache=tmp_path)
    try:
        with pytest.raises(ValueError, match="Boolean"):
            xp.jvp(source, (x,), (direction,))
        value, tangent = xp.jvp(source, (x,), (direction,), allow_comparison_where=True)
        assert not bool(np.signbit(value)) if name == "max" else bool(np.signbit(value))
        assert tangent == direction[0, 1]
        _, pullback = xp.vjp(source, x, argnums=0, allow_comparison_where=True)
        try:
            (result,) = pullback(np.array(3.0))
            expected = np.zeros_like(x)
            expected[0, 1] = 3
            np.testing.assert_array_equal(result, expected)
        finally:
            pullback.close()
    finally:
        source.close()


@pytest.mark.parametrize("name", ("max", "min"))
@pytest.mark.parametrize("axes,keepdims", (((1,), False), ((0, 2), True), ((), False)))
def test_native_fp64_without_reference(
    name: str,
    axes: tuple,
    keepdims: bool,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native reduction used reference execution/AD")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)
    x = np.random.default_rng(193).permutation(30).astype(float).reshape(2, 3, 5)
    direction = np.linspace(-0.7, 0.9, 30).reshape(x.shape)
    expected = getattr(np, name)(x, axis=axes, keepdims=keepdims)
    weights = (np.arange(expected.size).reshape(expected.shape) + 1.0) / 7
    tangent, adjoint = selected_oracle(x, direction, weights, axes, name, keepdims)
    source = xp.compile(
        lambda x: getattr(xp, name)(x, axis=axes, keepdims=keepdims),
        backend="cpu",
        cache=tmp_path,
    )
    try:
        for _ in range(2):
            np.testing.assert_array_equal(source(x), expected)
        _, actual = xp.jvp(source, (x,), (direction,), allow_comparison_where=True)
        np.testing.assert_allclose(actual, tangent, atol=2e-13, rtol=2e-13)
        _, pullback = xp.vjp(source, x, argnums=0, allow_comparison_where=True)
        try:
            (actual_adjoint,) = pullback(weights)
            np.testing.assert_allclose(actual_adjoint, adjoint, atol=2e-13, rtol=2e-13)
        finally:
            pullback.close()
    finally:
        source.close()


@pytest.mark.parametrize("name", ("max", "min"))
@pytest.mark.parametrize("backend", ("reference", "cpu"))
@pytest.mark.parametrize("shape,axis", (((2, 0, 5), (0, 2)), ((2, 1, 3), 1)))
def test_empty_output_and_singleton_ad(
    name: str, backend: str, shape: tuple, axis: Any, tmp_path: Any
) -> None:
    x = np.arange(np.prod(shape), dtype=float).reshape(shape)
    direction = np.full(shape, 2.5)
    source = xp.compile(
        lambda x: getattr(xp, name)(x, axis=axis, keepdims=True),
        backend=backend,
        cache=tmp_path,
    )
    try:
        expected = getattr(np, name)(x, axis=axis, keepdims=True)
        value, tangent = xp.jvp(source, (x,), (direction,))
        np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(tangent, np.full(expected.shape, 2.5))
        _, pullback = xp.vjp(source, x)
        try:
            (adjoint,) = pullback(np.full(expected.shape, 3.5))
            np.testing.assert_array_equal(adjoint, np.full(shape, 3.5))
        finally:
            pullback.close()
    finally:
        source.close()


@pytest.mark.parametrize("name", ("max", "min"))
def test_capabilities_and_unqualified_backends(name: str) -> None:
    from generativeqc_compiler.tensor.cpu import describe_cpu_compilation
    from generativeqc_compiler.tensor.optimize import prepare_for_backend

    report = xp.capabilities()
    matrix = report["operation_matrix"][name]
    from generativeqc_compiler.array_api.capabilities import capabilities

    assert capabilities()["array_api_version"] is None
    assert report["array_api_version"] == "2025.12"
    assert (
        matrix["capture"] == "tensorir" and "nondifferentiable-at-ties" in matrix["ad"]
    )
    assert matrix["native_cpu"] is False and matrix["native_cuda"] is False
    assert matrix["native_cpu_notes"] == "not-qualified"
    source = xp.compile(lambda x: getattr(xp, name)(x), backend="reference")
    try:
        with pytest.raises(ValueError, match="float64"):
            describe_cpu_compilation(source.lower(np.ones(3, dtype="float32")))
        with pytest.raises(ValueError):
            prepare_for_backend(source.lower(np.ones(3)), backend="cuda")
    finally:
        source.close()

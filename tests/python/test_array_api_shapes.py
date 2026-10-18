"""Eager vs captured Array API 2025.12 generic shape-operation regression tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc.extensions import tensor
from generativeqc_compiler.array_api import namespace

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("axes", (0, -1, (0, -1), (), (-3, -1)))
def test_expand_dims_matches_numpy_in_eager_and_captured_modes(
    dtype: object, axes: object
) -> None:
    x = np.arange(6.0, dtype=dtype).reshape(2, 3)
    # Tuple (-3,-1) adds two singleton axes to a rank-two input.
    expected = np.expand_dims(x, axis=axes)
    eager = xp.expand_dims(x, axes)

    @xp.compile(backend="reference")
    def expression(a: object) -> object:
        return xp.expand_dims(a, axes)

    compiled = expression(x)
    assert eager.dtype == compiled.dtype == x.dtype
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(compiled, expected)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("axes", (0, -1, (0, -1), ()))
def test_squeeze_matches_numpy_in_eager_and_captured_modes(
    dtype: object, axes: object
) -> None:
    x = np.arange(3.0, dtype=dtype).reshape(1, 3, 1)

    @xp.compile(backend="reference")
    def expression(a: object) -> object:
        return xp.squeeze(a, axes)

    expected = np.squeeze(x, axis=axes)
    np.testing.assert_array_equal(xp.squeeze(x, axes), expected)
    np.testing.assert_array_equal(expression(x), expected)


@pytest.mark.parametrize("axes", (None, 0, -1, (0, 2), ()))
def test_flip_eager_and_compiled_matches_numpy(axes: object) -> None:
    x = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)

    @xp.compile(backend="reference")
    def expression(a: object) -> object:
        return xp.flip(a, axis=axes)

    expected = np.flip(x, axis=axes)
    np.testing.assert_array_equal(xp.flip(x, axis=axes), expected)
    np.testing.assert_array_equal(expression(x), expected)


@pytest.mark.parametrize(
    ("source", "destination"),
    (
        (0, -1),
        (-1, 0),
        ((0, 2), (2, 0)),
        ((-1, 0), (-2, 2)),
        ((), ()),
    ),
)
def test_moveaxis_eager_and_compiled_matches_numpy(
    source: object, destination: object
) -> None:
    x = np.arange(24.0).astype(np.float64).reshape(2, 3, 4)

    @xp.compile(backend="reference")
    def expression(a: object) -> object:
        return xp.moveaxis(a, source, destination)

    expected = np.moveaxis(x, source, destination)
    np.testing.assert_array_equal(xp.moveaxis(x, source, destination), expected)
    np.testing.assert_array_equal(expression(x), expected)


@pytest.mark.parametrize("axes", ((-1, 0, -2), (2, 1, 0), (0, 1, 2)))
def test_permute_dims_2025_12_signed_axes(axes: tuple[int, ...]) -> None:
    values = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4)

    @xp.compile(backend="reference")
    def expression(x: object) -> object:
        return xp.permute_dims(x, axes)

    expected = np.transpose(values, axes)
    np.testing.assert_array_equal(xp.permute_dims(values, axes), expected)
    np.testing.assert_array_equal(expression(values), expected)


def test_broadcast_shapes_matches_static_numpy_shape_rules() -> None:
    assert xp.broadcast_shapes() == ()
    assert xp.broadcast_shapes((2, 1), (1, 3), ()) == (2, 3)
    assert xp.broadcast_shapes((0, 1), (1, 2)) == (0, 2)
    with pytest.raises(ValueError, match="cannot be broadcast"):
        xp.broadcast_shapes((2,), (3,))
    with pytest.raises(TypeError, match="nonnegative"):
        xp.broadcast_shapes((True,))


def test_broadcast_arrays_eager_compiled_and_mixed_dtype_preserve_types() -> None:
    left = np.arange(2.0, dtype=np.float32).reshape((2, 1))
    right = np.arange(3.0, dtype=np.float64).reshape((1, 3))
    eager = xp.broadcast_arrays(left, right)
    assert isinstance(eager, tuple)
    np.testing.assert_array_equal(eager[0], np.broadcast_to(left, (2, 3)))
    np.testing.assert_array_equal(eager[1], np.broadcast_to(right, (2, 3)))
    assert eager[0].dtype == left.dtype
    assert eager[1].dtype == right.dtype

    @xp.compile(backend="reference")
    def expression(x: object, y: object) -> object:
        a, b = xp.broadcast_arrays(x, y)
        return xp.sum(a) + xp.sum(b)

    promoted = expression(left, right)
    assert promoted.dtype == np.dtype("float64")
    np.testing.assert_allclose(promoted, np.sum(eager[0]) + np.sum(eager[1]))

    right32 = right.astype(np.float32)
    np.testing.assert_array_equal(
        expression(left, right32), np.sum(eager[0]) + np.sum(right32.repeat(2, axis=0))
    )


def test_shape_ops_use_existing_tensorir_jvp_and_scientific_guardrails() -> None:
    @xp.compile(differentiable=("x",), backend="reference")
    def expression(x: object) -> object:
        reshaped = xp.expand_dims(x, axis=(0, -1))
        return xp.sum(xp.flip(reshaped, axis=-2))

    values = np.asarray([1.0, -2.0, 4.0], dtype=np.float64)
    program = expression.lower(values)
    result = tensor.jvp(program, {"x": values}, {"x": np.ones_like(values)})
    np.testing.assert_array_equal(result.output_tangents["output"], 3.0)

    ao = xp.IndexSpace("ao", "ao", 2)
    spec = xp.TensorSpec((xp.Index("p", ao),), role="input")
    annotated = xp.input_array("x", spec)
    for operation in (
        lambda x: xp.expand_dims(x, 0),
        lambda x: xp.squeeze(x, ()),
        lambda x: xp.flip(x, axis=0),
    ):
        with pytest.raises(ValueError, match="scientific arrays require"):
            operation(annotated)
    with pytest.raises(TypeError, match="scientific domains"):
        xp.broadcast_arrays(annotated)

    # Axis-only permutations preserve the scientific indices without relabeling.
    matrix = xp.TensorSpec((xp.Index("p", ao), xp.Index("q", ao)), role="input")
    manual = xp.trace(lambda x: {"out": xp.permute_dims(x, (-1, -2))}, {"x": matrix})
    assert tuple(index.name for index in manual.outputs["out"].spec.indices) == (
        "q",
        "p",
    )


@pytest.mark.parametrize(
    "operation",
    (
        lambda x: xp.expand_dims(x, axis=(0, 0)),
        lambda x: xp.expand_dims(x, axis=True),
        lambda x: xp.squeeze(x, axis=1),
        lambda x: xp.moveaxis(x, (0, 1), (1, 1)),
        lambda x: xp.flip(x, axis=(0, 0)),
        lambda x: xp.permute_dims(x, (0, 0, 2)),
    ),
)
def test_invalid_axes_fail_closed_before_eager_or_captured_execution(
    operation: object,
) -> None:
    values = np.ones((1, 3, 1), dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        operation(values)
    with pytest.raises((TypeError, ValueError)):
        xp.compile(lambda x: operation(x), backend="reference").lower(values)


def test_capture_flip_rejects_excessive_static_gather_index_budget() -> None:
    @xp.compile(backend="reference")
    def expression(x: object) -> object:
        return xp.flip(x, axis=0)

    values = np.zeros((65537,), dtype=np.float64)
    with pytest.raises(ValueError, match="bounded static gather"):
        expression.lower(values)
    np.testing.assert_array_equal(xp.flip(values, axis=0), np.flip(values, axis=0))


def test_flip_does_not_enumerate_unselected_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Lowering must remain bounded even when an untouched axis is enormous.
    def bounded_range(*args: int) -> range:
        result = range(*args)
        assert len(result) <= 65536, "flip enumerated an unselected axis"
        return result

    monkeypatch.setattr(namespace, "range", bounded_range, raising=False)
    spec = xp.TensorSpec(namespace._generic_indices((10**12, 2)), role="input")
    program = xp.trace(lambda x: xp.flip(x, axis=-1), {"x": spec})
    assert program.outputs["output"].spec.shape == (10**12, 2)
    gathers = [node for node in program.nodes if node.op == "gather"]
    assert len(gathers) == 1
    assert dict(gathers[0].attrs)["positions"] == (1, 0)
    unchanged = xp.trace(lambda x: xp.flip(x, axis=()), {"x": spec})
    assert all(node.op != "gather" for node in unchanged.nodes)


@pytest.mark.parametrize(
    ("shape", "key", "expected"),
    (
        ((10**12, 2), (slice(None), slice(None)), (10**12, 2)),
        ((10**12, 0), (slice(None), slice(None)), (10**12, 0)),
        ((10**12, 2), (slice(11, 14), slice(None)), (3, 2)),
    ),
)
def test_contiguous_symbolic_subscript_does_not_expand_large_axes(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    key: tuple[slice, ...],
    expected: tuple[int, ...],
) -> None:
    # A symbolic slice must not allocate an index tuple proportional to its extent.
    def bounded_range(*args: int) -> range:
        result = range(*args)
        assert len(result) <= 65536, "contiguous slice enumerated symbolic indices"
        return result

    monkeypatch.setattr(namespace, "range", bounded_range, raising=False)
    spec = xp.TensorSpec(namespace._generic_indices(shape), role="input")
    captured = xp.trace(lambda x: x[key], {"x": spec})
    assert captured.outputs["output"].spec.shape == expected


@pytest.mark.parametrize(
    ("size", "key"),
    (
        (131074, slice(None, None, 2)),
        (65537, slice(None, None, -1)),
    ),
)
def test_symbolic_strided_subscript_rejects_oversized_gather(
    size: int, key: slice
) -> None:
    spec = xp.TensorSpec(namespace._generic_indices((size,)), role="input")
    with pytest.raises(ValueError, match="bounded static gather"):
        xp.trace(lambda x: x[key], {"x": spec})


@pytest.mark.parametrize(
    "key",
    (
        slice(None),
        slice(1, 5),
        slice(3, 3),
        slice(None, None, 2),
        slice(None, None, -1),
        slice(5, 0, -2),
    ),
)
def test_captured_symbolic_subscript_matches_numpy(key: slice) -> None:
    @xp.compile(backend="reference")
    def expression(x: object) -> object:
        return x[key]

    values = np.arange(6, dtype=np.float64)
    np.testing.assert_array_equal(expression(values), values[key])


@pytest.mark.parametrize("dtype", (np.bool_, np.float32, np.float64))
@pytest.mark.parametrize(
    "shape,repetitions",
    (
        ((2, 3), (2, 3)),
        ((2, 3), (3,)),
        ((2, 3), ()),
        ((2, 3), (2, 1, 3)),
        ((2, 3), (0, 2)),
        ((2, 3), (1, 1)),
        ((), ()),
        ((), (2, 3)),
        ((), (0,)),
        ((2, 0, 3), (2, 3, 1)),
        ((0, 3), (2, 0, 3)),
    ),
)
def test_tile_eager_and_captured_values(
    dtype: Any, shape: tuple[int, ...], repetitions: tuple[int, ...]
) -> None:
    values = np.resize(np.array([-0.0, 0.0, 1.25, -2.5]), shape).astype(dtype)
    if values.ndim:
        values = values[::-1]
    expected = np.tile(values, repetitions)
    compiled = xp.compile(lambda x: xp.tile(x, repetitions), backend="reference")
    try:
        for result in (xp.tile(values, repetitions), compiled(values)):
            assert result.shape == expected.shape
            assert result.dtype == expected.dtype
            np.testing.assert_array_equal(result, expected)
            if dtype is not np.bool_:
                np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
            assert not np.shares_memory(result, values)
    finally:
        compiled.close()


@pytest.mark.parametrize(
    "repetitions,error",
    (
        (2, TypeError),
        ([2, 3], TypeError),
        (np.array([2, 3]), TypeError),
        ((True,), TypeError),
        ((1.0,), TypeError),
        ((np.int64(2),), TypeError),
        ((-1, 2), ValueError),
    ),
)
def test_tile_rejects_nonstatic_or_invalid_repetitions(
    repetitions: object, error: type[Exception]
) -> None:
    compiled = xp.compile(lambda x: xp.tile(x, repetitions), backend="reference")
    try:
        with pytest.raises(error):
            compiled.lower(np.ones((2, 3)))
    finally:
        compiled.close()


@pytest.mark.parametrize("repetitions", ((), (1,), (0,), (2,)))
def test_tile_keeps_scientific_index_spaces_explicit(
    repetitions: tuple[int, ...],
) -> None:
    ao = xp.IndexSpace("ao", "ao", 2)
    value = xp.input_array("x", xp.TensorSpec((xp.Index("p", ao),), role="input"))
    with pytest.raises(ValueError, match="scientific"):
        xp.tile(value, repetitions)


def test_tile_checks_shape_overflow_without_enumerating_repetitions() -> None:
    value = namespace.ones((65537,))
    assert xp.tile(value, (2,)).shape == (131074,)
    with pytest.raises(ValueError, match="signed-64-bit"):
        xp.tile(value, (2**63,))
    with pytest.raises(ValueError, match="byte count"):
        xp.tile(namespace.ones((2,)), (2**59,))
    empty = namespace.ones((2**62, 0, 2))
    assert xp.tile(empty, (1, 3, 1)).shape == (2**62, 0, 2)
    # Validate the final empty shape without expanding an enormous earlier axis.
    assert xp.tile(namespace.ones((2, 3)), (2**60, 0)).shape == (2**61, 0)
    with pytest.raises(ValueError, match="signed-64-bit"):
        xp.tile(namespace.ones((2, 3)), (2**63, 0))


@pytest.mark.parametrize("backend", ("reference", "cpu"))
@pytest.mark.parametrize("repetitions", ((2, 3), (3,), (2, 1, 2), (0, 3)))
def test_tile_jvp_vjp_against_numpy_finite_differences(
    backend: str,
    repetitions: tuple[int, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if backend == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("native tile used reference execution or AD")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    rng = np.random.default_rng(2111)
    values, direction = (rng.normal(size=(2, 3)) for _ in range(2))
    expression = lambda x: xp.tile(x, repetitions)
    result, tangent = xp.jvp(
        expression, (values,), (direction,), backend=backend, cache=tmp_path
    )
    np.testing.assert_array_equal(result, np.tile(values, repetitions))
    step = 1e-5
    finite = (
        np.tile(values + step * direction, repetitions)
        - np.tile(values - step * direction, repetitions)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite, rtol=1e-9, atol=1e-9)
    weight = rng.normal(size=result.shape)
    _, pullback = xp.vjp(expression, values, backend=backend, cache=tmp_path)
    try:
        (gradient,) = pullback(weight)
        expected = np.empty_like(values)
        for index in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            plus[index] += step
            minus[index] -= step
            expected[index] = np.sum(
                weight * (np.tile(plus, repetitions) - np.tile(minus, repetitions))
            ) / (2 * step)
        np.testing.assert_allclose(gradient, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            np.sum(weight * tangent),
            np.sum(gradient * direction),
            rtol=1e-13,
            atol=1e-13,
        )
    finally:
        pullback.close()


@pytest.mark.parametrize("dtype", (np.bool_, np.float64))
@pytest.mark.parametrize(
    "shape,repetitions",
    (((2, 3), (2, 3)), ((0, 3), (2,)), ((), (2, 3)), ((2, 3), (0, 2))),
)
def test_tile_native_replay_and_output_ownership(
    dtype: Any, shape: tuple[int, ...], repetitions: tuple[int, ...], tmp_path: Path
) -> None:
    values = np.resize(np.array([-0.0, 0.0, 1.25, -2.5]), shape).astype(dtype)
    compiled = xp.compile(lambda x: xp.tile(x, repetitions), cache=tmp_path)
    try:
        for source in (
            values,
            np.logical_not(values) if dtype is np.bool_ else -values,
        ):
            result = compiled(source)
            expected = np.tile(source, repetitions)
            np.testing.assert_array_equal(result, expected)
            assert result.dtype == expected.dtype
            if dtype is not np.bool_:
                np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
            assert not np.shares_memory(result, source)
    finally:
        compiled.close()


def test_tile_keeps_native_resource_limits(tmp_path: Path) -> None:
    compiled = xp.compile(lambda x: xp.tile(x, (1000,)), cache=tmp_path, max_bytes=64)
    try:
        with pytest.raises(ValueError):
            compiled(np.ones(2))
    finally:
        compiled.close()


@pytest.mark.parametrize("axis", (0, 31, 63))
@pytest.mark.parametrize("empty", (False, True))
def test_tile_supports_maximum_numpy_rank(axis: int, empty: bool) -> None:
    shape = (2,) + (1,) * 30 + (3,) + (1,) * 31 + (0 if empty else 2,)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    repetitions = (1,) * axis + (2,) + (1,) * (63 - axis)
    compiled = xp.compile(lambda x: xp.tile(x, repetitions), backend="reference")
    try:
        np.testing.assert_array_equal(compiled(values), np.tile(values, repetitions))
    finally:
        compiled.close()

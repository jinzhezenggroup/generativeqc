"""Eager vs captured Array API 2025.12 generic shape-operation regression tests."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc.extensions import tensor
from generativeqc_compiler.array_api import namespace


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("axes", (0, -1, (0, -1), (), (-3, -1)))
def test_expand_dims_matches_numpy_in_eager_and_captured_modes(
    dtype: object, axes: object
) -> None:
    x = np.arange(6.0, dtype=dtype).reshape(2, 3)
    # Tuple (-3,-1) adds two singleton axes to a rank-two input.
    expected = np.expand_dims(x, axis=axes)
    eager = xp.expand_dims(x, axes)

    @xp.compile
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

    @xp.compile
    def expression(a: object) -> object:
        return xp.squeeze(a, axes)

    expected = np.squeeze(x, axis=axes)
    np.testing.assert_array_equal(xp.squeeze(x, axes), expected)
    np.testing.assert_array_equal(expression(x), expected)


@pytest.mark.parametrize("axes", (None, 0, -1, (0, 2), ()))
def test_flip_eager_and_compiled_matches_numpy(axes: object) -> None:
    x = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)

    @xp.compile
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

    @xp.compile
    def expression(a: object) -> object:
        return xp.moveaxis(a, source, destination)

    expected = np.moveaxis(x, source, destination)
    np.testing.assert_array_equal(xp.moveaxis(x, source, destination), expected)
    np.testing.assert_array_equal(expression(x), expected)


@pytest.mark.parametrize("axes", ((-1, 0, -2), (2, 1, 0), (0, 1, 2)))
def test_permute_dims_2025_12_signed_axes(axes: tuple[int, ...]) -> None:
    values = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4)

    @xp.compile
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

    @xp.compile
    def expression(x: object, y: object) -> object:
        a, b = xp.broadcast_arrays(x, y)
        return xp.sum(a) + xp.sum(b)

    with pytest.raises(ValueError, match="dtype"):
        expression(left, right)

    right32 = right.astype(np.float32)
    np.testing.assert_array_equal(
        expression(left, right32), np.sum(eager[0]) + np.sum(right32.repeat(2, axis=0))
    )


def test_shape_ops_use_existing_tensorir_jvp_and_scientific_guardrails() -> None:
    @xp.compile(differentiable=("x",))
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
        xp.compile(lambda x: operation(x)).lower(values)


def test_capture_flip_rejects_excessive_static_gather_index_budget() -> None:
    @xp.compile
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

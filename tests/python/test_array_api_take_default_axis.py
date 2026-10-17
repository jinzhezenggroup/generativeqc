"""Static take uses the sole input axis when its axis argument is omitted."""

from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import namespace


@pytest.mark.parametrize("dtype", [np.bool_, np.float32, np.float64])
@pytest.mark.parametrize("indices", [(2, 0, 2), (), (0,)])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_take_default_axis_numpy_parity(
    dtype: Any, indices: Any, explicit_none: bool
) -> None:
    data = np.array([-0.0, 0.0, 1.25, -2.5], dtype=dtype)[::-1]
    kwargs = {"axis": None} if explicit_none else {}
    expected = np.take(data, indices, axis=0)
    compiled = xp.compile(lambda x: xp.take(x, indices, **kwargs), backend="reference")
    try:
        for result in (xp.take(data, indices, **kwargs), compiled(data)):
            assert result.shape == expected.shape
            assert result.dtype == expected.dtype
            np.testing.assert_array_equal(result, expected)
            if dtype is not np.bool_:
                np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
            assert not np.shares_memory(result, data)
    finally:
        compiled.close()


@pytest.mark.parametrize("shape", [(), (2, 3), (0, 2)])
def test_take_default_axis_requires_vector(shape: Any) -> None:
    compiled = xp.compile(lambda x: xp.take(x, ()), backend="reference")
    try:
        with pytest.raises(ValueError, match="one-dimensional"):
            compiled.lower(np.ones(shape))
    finally:
        compiled.close()


@pytest.mark.parametrize(
    "indices,error",
    [
        (True, TypeError),
        ((True,), TypeError),
        ((1.0,), TypeError),
        ((-1,), ValueError),
        ((3,), ValueError),
    ],
)
def test_take_default_axis_preserves_static_index_admission(
    indices: Any, error: Any
) -> None:
    compiled = xp.compile(lambda x: xp.take(x, indices), backend="reference")
    try:
        with pytest.raises(error):
            compiled.lower(np.ones(3))
    finally:
        compiled.close()


def test_take_default_axis_preserves_scientific_index_metadata() -> None:
    ao = xp.IndexSpace("ao", "ao", 3)
    value = xp.input_array("x", xp.TensorSpec((xp.Index("p", ao),), role="input"))
    implicit = namespace.take(value, (2, 0, 2))
    explicit = namespace.take(value, (2, 0, 2), axis=0)
    assert implicit.node.spec == explicit.node.spec
    assert implicit.node.spec.indices == (xp.Index("p", ao, selection=(2, 0, 2)),)
    assert implicit.shape == (3,)


@pytest.mark.parametrize("backend", ["reference", "cpu"])
@pytest.mark.parametrize("indices", [(2, 0, 2), ()])
def test_take_default_axis_derivatives(
    backend: str, indices: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    if backend == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native take used reference execution or AD")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    data = np.array([1.25, -2.5, 4.0])
    direction = np.array([-0.5, 1.5, 2.0])
    expression = lambda x: xp.take(x, indices)
    value, tangent = xp.jvp(
        expression, (data,), (direction,), backend=backend, cache=tmp_path
    )
    np.testing.assert_array_equal(value, np.take(data, indices))
    step = 1e-5
    finite = (
        np.take(data + step * direction, indices)
        - np.take(data - step * direction, indices)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite, rtol=1e-9, atol=1e-9)
    weight = np.arange(len(indices), dtype=np.float64) + 0.75
    _, pullback = xp.vjp(expression, data, backend=backend, cache=tmp_path)
    try:
        (gradient,) = pullback(weight)
        expected = np.zeros_like(data)
        np.add.at(expected, list(indices), weight)
        np.testing.assert_array_equal(gradient, expected)
        np.testing.assert_allclose(np.dot(weight, tangent), np.dot(gradient, direction))
    finally:
        pullback.close()


@pytest.mark.parametrize("dtype", [np.bool_, np.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_take_default_axis_native_replay(
    dtype: Any, empty: bool, tmp_path: Any
) -> None:
    data = np.array([] if empty else [-0.0, 2.0, -3.0], dtype=dtype)
    indices = () if empty else (2, 0, 2)
    compiled = xp.compile(lambda x: xp.take(x, indices), cache=tmp_path)
    try:
        for source in (data, data[::-1]):
            result = compiled(source)
            np.testing.assert_array_equal(result, np.take(source, indices))
            assert result.dtype == data.dtype
            if dtype is not np.bool_:
                np.testing.assert_array_equal(
                    np.signbit(result), np.signbit(np.take(source, indices))
                )
            assert not np.shares_memory(result, source)
    finally:
        compiled.close()


def test_take_default_axis_keeps_native_dtype_boundary(tmp_path: Any) -> None:
    compiled = xp.compile(lambda x: xp.take(x, (0,)), cache=tmp_path)
    try:
        with pytest.raises(ValueError, match="float64 or typed bool"):
            compiled(np.ones(3, dtype=np.float32))
    finally:
        compiled.close()

"""Uniform repeat values, static admission, ownership and derivative gates."""

from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import namespace


@pytest.mark.parametrize("dtype", [np.bool_, np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,count,axis",
    [
        ((2, 3), 2, None),
        ((2, 3), 3, 0),
        ((2, 3), 2, -1),
        ((2, 3), 0, 1),
        ((2, 3), 1, None),
        ((2, 3), 1, 0),
        ((), 3, None),
        ((), 0, None),
        ((2, 0, 3), 2, 1),
        ((0, 3), 2, -1),
        ((2, 3), 0, None),
    ],
)
def test_repeat_numpy_parity(dtype: Any, shape: Any, count: int, axis: Any) -> None:
    values = np.resize(np.array([-0.0, 0.0, 1.25, -2.5]), shape).astype(dtype)
    if values.ndim:
        values = values[::-1]
    expected = np.repeat(values, count, axis=axis)
    compiled = xp.compile(lambda x: xp.repeat(x, count, axis=axis), backend="reference")
    try:
        for result in (xp.repeat(values, count, axis=axis), compiled(values)):
            assert result.shape == expected.shape
            assert result.dtype == expected.dtype
            np.testing.assert_array_equal(result, expected)
            if dtype is not np.bool_:
                np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
            assert not np.shares_memory(result, values)
    finally:
        compiled.close()


@pytest.mark.parametrize(
    "count,axis,error",
    [
        (True, 0, TypeError),
        (1.0, 0, TypeError),
        ((1, 2), 0, TypeError),
        (np.array([1, 2]), 0, TypeError),
        (-1, 0, ValueError),
        (2, True, TypeError),
        (2, (0,), TypeError),
        (2, 2, ValueError),
        (2, -3, ValueError),
    ],
)
def test_repeat_rejects_invalid_static_arguments(
    count: Any, axis: Any, error: Any
) -> None:
    compiled = xp.compile(lambda x: xp.repeat(x, count, axis=axis), backend="reference")
    try:
        with pytest.raises(error):
            compiled.lower(np.ones((2, 3)))
    finally:
        compiled.close()


@pytest.mark.parametrize("count", [0, 1, 2])
def test_repeat_preserves_scientific_boundary(count: int) -> None:
    ao = xp.IndexSpace("ao", "ao", 2)
    value = xp.input_array("x", xp.TensorSpec((xp.Index("p", ao),), role="input"))
    with pytest.raises(ValueError, match="scientific"):
        xp.repeat(value, count)


def test_repeat_large_symbolic_shape_and_overflow() -> None:
    value = namespace.ones((65537,))
    assert xp.repeat(value, 2).shape == (131074,)
    assert xp.repeat(value, 0).shape == (0,)
    assert xp.repeat(value, 1).shape == value.shape
    with pytest.raises(ValueError):
        xp.repeat(value, 2**63)
    scalar = namespace.ones(())
    with pytest.raises(ValueError, match="axis"):
        xp.repeat(scalar, 2, axis=0)


@pytest.mark.parametrize("backend", ["reference", "cpu"])
@pytest.mark.parametrize("count,axis", [(3, None), (2, 0), (3, -1), (0, -1)])
def test_repeat_derivatives(
    backend: str, count: int, axis: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    if backend == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native repeat used reference execution or AD")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    rng = np.random.default_rng(2111)
    data, direction = (rng.normal(size=(2, 3)) for _ in range(2))
    expression = lambda x: xp.repeat(x, count, axis=axis)
    value, tangent = xp.jvp(
        expression, (data,), (direction,), backend=backend, cache=tmp_path
    )
    np.testing.assert_array_equal(value, np.repeat(data, count, axis=axis))
    step = 1e-5
    finite = (
        np.repeat(data + step * direction, count, axis=axis)
        - np.repeat(data - step * direction, count, axis=axis)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite, rtol=1e-9, atol=1e-9)
    weight = rng.normal(size=value.shape)
    _, pullback = xp.vjp(expression, data, backend=backend, cache=tmp_path)
    try:
        (gradient,) = pullback(weight)
        expected = np.empty_like(data)
        for index in np.ndindex(data.shape):
            plus, minus = data.copy(), data.copy()
            plus[index] += step
            minus[index] -= step
            expected[index] = np.sum(
                weight
                * (
                    np.repeat(plus, count, axis=axis)
                    - np.repeat(minus, count, axis=axis)
                )
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


@pytest.mark.parametrize("dtype", [np.bool_, np.float64])
@pytest.mark.parametrize(
    "shape,count,axis",
    [((2, 3), 2, -1), ((0, 3), 2, None), ((), 2, None), ((2, 3), 0, 0)],
)
def test_repeat_native_boundary(
    dtype: Any, shape: Any, count: int, axis: Any, tmp_path: Any
) -> None:
    values = (np.arange(np.prod(shape)).reshape(shape) % 2).astype(dtype)
    compiled = xp.compile(lambda x: xp.repeat(x, count, axis=axis), cache=tmp_path)
    try:
        result = compiled(values)
        np.testing.assert_array_equal(result, np.repeat(values, count, axis=axis))
        assert not np.shares_memory(result, values)
    finally:
        compiled.close()


def test_repeat_keeps_native_resource_limits(tmp_path: Any) -> None:
    compiled = xp.compile(lambda x: xp.repeat(x, 1000), cache=tmp_path, max_bytes=64)
    try:
        with pytest.raises(ValueError):
            compiled(np.ones(2))
    finally:
        compiled.close()

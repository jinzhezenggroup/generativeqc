"""Captured roll values, bounded metadata and independent derivative oracles."""

from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import namespace
from generativeqc_compiler.tensor import Index, IndexSpace, TensorSpec


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.bool_])
@pytest.mark.parametrize(
    ("shape", "shift", "axis"),
    [
        ((2, 3), 2, None),
        ((2, 3), -7, -1),
        ((2, 3), (1, -2), (0, 1)),
        ((2, 3), 1, (0, -1)),
        ((2, 3), (1, 2), -1),
        ((2, 3), (1, -4), (1, -1)),
        ((2, 3), (1, 2), None),
        ((2, 3), 1, ()),
        ((2, 3), (), ()),
        ((2, 3), 10**40, -1),
        ((), 3, None),
        ((2, 0, 3), 2, None),
        ((2, 0, 3), (1, 2), (0, 1)),
        ((0,), 1, 0),
    ],
)
def test_roll_numpy_parity(dtype: Any, shape: Any, shift: Any, axis: Any) -> None:
    data = np.arange(np.prod(shape, dtype=int)).reshape(shape).astype(dtype)
    compiled = xp.compile(lambda x: xp.roll(x, shift, axis=axis), backend="reference")
    expected = np.roll(data, shift, axis=axis)
    for result in (xp.roll(data, shift, axis=axis), compiled(data)):
        assert result.shape == data.shape
        assert result.dtype == data.dtype
        np.testing.assert_array_equal(result, expected)
        assert not np.shares_memory(result, data)


@pytest.mark.parametrize(
    ("shift", "axis", "error"),
    [
        (True, 0, TypeError),
        (1.5, 0, TypeError),
        ((1, False), 0, TypeError),
        ([1], 0, TypeError),
        (1, True, TypeError),
        (1, (False,), TypeError),
        (1, 2, ValueError),
        (1, -3, ValueError),
        ((1, 2), (0, 1, 0), ValueError),
    ],
)
def test_roll_invalid_static_arguments(shift: Any, axis: Any, error: Any) -> None:
    with pytest.raises(error):
        xp.compile(lambda x: xp.roll(x, shift, axis=axis), backend="reference").lower(
            np.ones((2, 3))
        )


def _symbolic(shape: tuple[int, ...]) -> Any:
    return xp.input_array(
        "x",
        TensorSpec(
            tuple(
                Index(f"d{i}", IndexSpace(f"_array_extent_{n}", "matrix", n))
                for i, n in enumerate(shape)
            ),
            "float64",
            role="input",
            differentiable=True,
        ),
    )


def test_roll_budget_and_noop_metadata() -> None:
    huge = _symbolic((10**12,))
    with pytest.raises(ValueError, match="budget"):
        namespace.roll(huge, 1)
    assert namespace.roll(huge, 10**12, axis=0).shape == huge.shape
    assert namespace.roll(huge, (1, -1), axis=(0, 0)).shape == huge.shape
    with pytest.raises(ValueError, match="budget"):
        namespace.roll(_symbolic((32769, 32768)), 1, axis=(0, 1))
    assert namespace.roll(_symbolic((65536,)), 1, axis=0).shape == (65536,)
    assert namespace.roll(_symbolic((0,)), 1).shape == (0,)


def test_roll_preserves_scientific_boundary() -> None:
    ao = Index("p", IndexSpace("ao", "ao", 3))
    x = xp.input_array(
        "x", TensorSpec((ao,), "float64", role="input", differentiable=True)
    )
    with pytest.raises(ValueError, match="scientific"):
        xp.roll(x, 1)


@pytest.mark.parametrize("backend", ["reference", "cpu"])
@pytest.mark.parametrize("axis", [None, (0, -1), (1, -1)])
def test_roll_derivatives(
    backend: str, axis: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    if backend == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native roll used reference execution")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    rng = np.random.default_rng(2111)
    data, direction, weight = (rng.normal(size=(2, 3)) for _ in range(3))
    shift = 1 if axis is None else (1, -2)
    expression = lambda x: xp.roll(x, shift, axis=axis)
    value, tangent = xp.jvp(
        expression, (data,), (direction,), backend=backend, cache=tmp_path
    )
    np.testing.assert_array_equal(value, np.roll(data, shift, axis=axis))
    step = 1e-5
    finite = (
        np.roll(data + step * direction, shift, axis=axis)
        - np.roll(data - step * direction, shift, axis=axis)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite, rtol=1e-9, atol=1e-9)
    _, pullback = xp.vjp(expression, data, backend=backend, cache=tmp_path)
    try:
        (gradient,) = pullback(weight)
        inverse = -shift if isinstance(shift, int) else tuple(-s for s in shift)
        np.testing.assert_array_equal(gradient, np.roll(weight, inverse, axis=axis))
        np.testing.assert_allclose(
            np.sum(weight * tangent),
            np.sum(gradient * direction),
            rtol=1e-13,
            atol=1e-13,
        )
    finally:
        pullback.close()


@pytest.mark.parametrize("dtype", [np.float64, np.bool_])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_roll_native_boundary(dtype: Any, shape: Any, tmp_path: Any) -> None:
    data = (np.arange(np.prod(shape, dtype=int)).reshape(shape) % 2).astype(dtype)
    compiled = xp.compile(lambda x: xp.roll(x, 1), cache=tmp_path)
    try:
        np.testing.assert_array_equal(compiled(data), np.roll(data, 1))
    finally:
        compiled.close()

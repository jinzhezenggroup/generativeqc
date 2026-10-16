"""Matrix-only T semantics, scientific metadata and native derivative contracts."""

from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from generativeqc.experimental import array_api as xp


@pytest.mark.parametrize("shape", [(), (0,), (3,), (2, 3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", [np.bool_, np.float32, np.float64])
def test_T_rejects_nonmatrices_in_eager_and_capture(shape: Any, dtype: Any) -> None:
    values = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError):
        _ = strict.asarray(values).T
    with pytest.raises(ValueError, match="two-dimensional"):
        _ = xp.asarray(values).T
    compiled = xp.compile(lambda x: x.T, backend="reference")
    try:
        with pytest.raises(ValueError, match="two-dimensional"):
            compiled.lower(values)
    finally:
        compiled.close()


@pytest.mark.parametrize("shape", [(2, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("dtype", [np.bool_, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_T_matrix_values_and_layouts(shape: Any, dtype: Any, layout: str) -> None:
    values = np.resize(np.array([-0.0, 0.0, 1.25, -2.5]), shape).astype(dtype)
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        buffer = np.empty((shape[0], shape[1] * 2), dtype=dtype)
        buffer[:, ::2] = values
        values = buffer[:, ::2]
    expected = np.transpose(values, (1, 0))
    compiled = xp.compile(lambda x: x.T, backend="reference")
    try:
        for result in (xp.asarray(values).T, compiled(values)):
            assert result.shape == expected.shape
            assert result.dtype == expected.dtype
            np.testing.assert_array_equal(result, expected)
            if dtype is not np.bool_:
                np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
        assert not np.shares_memory(compiled(values), values)
    finally:
        compiled.close()


def test_T_preserves_scientific_axis_metadata() -> None:
    ao = xp.IndexSpace("active_ao", "ao", 5, spin="alpha")
    occupied = xp.IndexSpace("occupied", "occupied", 4, spin="alpha")
    axes = (xp.Index("p", ao, 1, 4, (3, 1)), xp.Index("i", occupied, 1, 4))
    spec = xp.TensorSpec(axes, role="input", differentiable=True)
    symbolic = xp.input_array("x", spec)
    result = symbolic.T
    assert result.node.spec.indices == axes[::-1]
    assert result.dtype == spec.dtype
    assert result.node.spec.differentiable
    with pytest.raises(ValueError, match="two-dimensional"):
        _ = xp.input_array("x", xp.TensorSpec(axes[:1], role="input")).T


def test_matrix_stacks_retain_explicit_transpose_operations() -> None:
    values = np.arange(24.0).reshape(2, 3, 4)
    expression = xp.compile(
        lambda x: {"matrix": x.mT, "reverse": xp.permute_dims(x, (2, 1, 0))},
        backend="reference",
    )
    try:
        result = expression(values)
        np.testing.assert_array_equal(result["matrix"], np.swapaxes(values, -1, -2))
        np.testing.assert_array_equal(
            result["reverse"], np.transpose(values, (2, 1, 0))
        )
    finally:
        expression.close()


@pytest.mark.parametrize("backend", ["reference", "cpu"])
def test_T_values_and_derivatives(
    backend: str, tmp_path: Any, monkeypatch: Any
) -> None:
    if backend == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native transpose used reference execution or AD")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    rng = np.random.default_rng(2111)
    data, direction = (rng.normal(size=(2, 3)) for _ in range(2))
    weight = rng.normal(size=(3, 2))
    expression = lambda x: x.T
    compiled = xp.compile(expression, backend=backend, cache=tmp_path)
    try:
        result = compiled(data)
        np.testing.assert_array_equal(result, np.transpose(data))
        assert not np.shares_memory(result, data)
    finally:
        compiled.close()
    value, tangent = xp.jvp(
        expression, (data,), (direction,), backend=backend, cache=tmp_path
    )
    np.testing.assert_array_equal(value, np.transpose(data))
    step = 1e-5
    finite = (
        np.transpose(data + step * direction) - np.transpose(data - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite, rtol=1e-9, atol=1e-9)
    _, pullback = xp.vjp(expression, data, backend=backend, cache=tmp_path)
    try:
        (gradient,) = pullback(weight)
        np.testing.assert_array_equal(gradient, np.transpose(weight))
        np.testing.assert_allclose(
            np.sum(weight * tangent),
            np.sum(gradient * direction),
            rtol=1e-13,
            atol=1e-13,
        )
    finally:
        pullback.close()


@pytest.mark.parametrize("shape", [(2, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("dtype", [np.bool_, np.float64])
def test_T_native_boolean_and_empty_matrices(
    shape: Any, dtype: Any, tmp_path: Any
) -> None:
    values = (np.arange(np.prod(shape)).reshape(shape) % 2).astype(dtype)
    compiled = xp.compile(lambda x: x.T, cache=tmp_path)
    try:
        np.testing.assert_array_equal(compiled(values), np.transpose(values))
    finally:
        compiled.close()

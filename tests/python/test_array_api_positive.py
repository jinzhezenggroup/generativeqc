"""Unary positive capture values, dtype guards and native derivative contracts."""

import operator
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import namespace


@pytest.mark.parametrize("operation", [xp.positive, operator.pos])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_positive_eager_capture_numpy(operation: Any, dtype: Any, shape: Any) -> None:
    values = np.resize(np.array([-0.0, 0.0, -2.5, 1.25], dtype=dtype), shape)
    expected = np.positive(values)
    compiled = xp.compile(lambda x: operation(x), backend="reference")
    try:
        for result in (operation(xp.asarray(values)), compiled(values)):
            assert result.dtype == values.dtype
            assert result.shape == values.shape
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
        assert not np.shares_memory(compiled(values), values)
    finally:
        compiled.close()


@pytest.mark.parametrize("operation", [xp.positive, operator.pos])
def test_positive_rejects_boolean_eager_and_capture(operation: Any) -> None:
    values = xp.asarray([True, False])
    with pytest.raises(TypeError):
        operation(values)
    compiled = xp.compile(lambda x: operation(x), backend="reference")
    try:
        with pytest.raises(TypeError, match="float32 and float64"):
            compiled.lower(values)
    finally:
        compiled.close()


@pytest.mark.parametrize("dtype", ["bool", "int64"])
@pytest.mark.parametrize("operation", [xp.positive, operator.pos, namespace.positive])
def test_positive_rejects_symbolic_nonnumeric_data(dtype: str, operation: Any) -> None:
    value = xp.input_array("x", xp.TensorSpec(dtype=dtype, role="input"))
    with pytest.raises(TypeError, match="float32 and float64"):
        operation(value)


@pytest.mark.parametrize("operation", [xp.positive, operator.pos])
def test_positive_preserves_scientific_domain(operation: Any) -> None:
    space = xp.IndexSpace("ao", "ao", 2)
    spec = xp.TensorSpec((xp.Index("p", space),), role="input", differentiable=True)
    value = xp.input_array("x", spec)
    result = operation(value)
    assert result.node.spec == spec
    program = xp.trace(lambda x: operation(x), {"x": spec})
    assert program.outputs["output"].spec == spec


@pytest.mark.parametrize("operation", [xp.positive, operator.pos])
@pytest.mark.parametrize("backend", ["reference", "cpu"])
def test_positive_derivatives_and_native_execution(
    operation: Any, backend: str, tmp_path: Any, monkeypatch: Any
) -> None:
    if backend == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native positive used a reference evaluator")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    data = np.array([-0.0, -2.0, 3.0])
    direction = np.array([0.25, 1.0, -0.5])
    weight = np.array([-1.5, 2.0, 0.25])
    compiled = xp.compile(lambda x: operation(x), backend=backend, cache=tmp_path)
    try:
        result = compiled(data)
        np.testing.assert_array_equal(result, np.positive(data))
        np.testing.assert_array_equal(np.signbit(result), np.signbit(data))
        assert not np.shares_memory(result, data)
    finally:
        compiled.close()
    value, tangent = xp.jvp(
        lambda x: operation(x), (data,), (direction,), backend=backend, cache=tmp_path
    )
    np.testing.assert_array_equal(value, np.positive(data))
    step = 1e-5
    finite = (
        np.positive(data + step * direction) - np.positive(data - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite, rtol=1e-9, atol=1e-9)
    _, pullback = xp.vjp(lambda x: operation(x), data, backend=backend, cache=tmp_path)
    try:
        (gradient,) = pullback(weight)
        np.testing.assert_array_equal(gradient, weight)
        np.testing.assert_allclose(
            np.sum(weight * tangent), np.sum(gradient * direction), rtol=1e-14
        )
    finally:
        pullback.close()


@pytest.mark.parametrize("operation", [xp.positive, operator.pos])
@pytest.mark.parametrize("backend", ["reference", "cpu"])
@pytest.mark.parametrize("invalid", [np.inf, np.nan])
def test_positive_keeps_finite_input_admission(
    operation: Any, backend: str, invalid: float, tmp_path: Any
) -> None:
    compiled = xp.compile(lambda x: operation(x), backend=backend, cache=tmp_path)
    try:
        with pytest.raises(ValueError):
            compiled(np.array([invalid]))
    finally:
        compiled.close()

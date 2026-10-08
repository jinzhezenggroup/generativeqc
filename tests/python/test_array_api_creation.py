"""Array API 2025.12 uniform-creation subset: eager/captured contract tests."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from generativeqc.experimental import array_api as xp
from generativeqc.extensions import tensor
from generativeqc_compiler.array_api.trace import active_capture


@pytest.mark.parametrize("dtype", (xp.float32, xp.float64))
@pytest.mark.parametrize("shape", ((), (0,), (0, 3), (2, 3), 4))
def test_uniform_creators_match_numpy(dtype: object, shape: object) -> None:
    for operation, expected in (
        (xp.zeros, np.zeros),
        (xp.ones, np.ones),
    ):
        actual = operation(shape, dtype=dtype)
        reference = expected(shape, dtype=dtype)
        assert actual.dtype == reference.dtype
        np.testing.assert_array_equal(actual, reference)
    result = xp.full(shape, 0.25, dtype=dtype)
    np.testing.assert_array_equal(result, np.full(shape, 0.25, dtype=dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_uniform_and_like_creation_are_captured_as_tensorir(dtype: object) -> None:
    @xp.compile
    def expression(x: object) -> object:
        zero = xp.zeros(x.shape, dtype=x.dtype)
        ones = xp.ones_like(x)
        fourth = xp.full_like(x, Fraction(1, 4))
        extra = xp.full(x.shape, 0.5, dtype=x.dtype)
        return (x + zero + ones) * fourth + extra

    values = np.arange(6, dtype=dtype).reshape((2, 3))
    expected = (values + np.ones_like(values)) * dtype(0.25) + dtype(0.5)
    eager = (values + xp.zeros_like(values) + xp.ones_like(values)) * xp.full_like(
        values, 0.25
    ) + xp.full(values.shape, 0.5, dtype=dtype)
    result = expression(values)
    assert eager.dtype == result.dtype == values.dtype
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(result, expected)
    lowered = expression.lower(values)
    assert isinstance(lowered, xp.Program)
    assert any(node.op == "broadcast" and not node.spec.differentiable
               for node in lowered.nodes)
    assert all(node.op in {"input", "constant", "broadcast", "add", "multiply", "reshape"}
               for node in lowered.nodes)


def test_uniform_constant_is_one_scalar_payload_without_n_squared_expansion() -> None:
    # Symbolic payload remains one scalar even for large shapes.
    @xp.compile
    def expression(x: object) -> object:
        return x + xp.ones((1024, 1024), dtype=x.dtype)

    value = np.zeros((1024, 1024), dtype=np.float64)
    graph = expression.lower(value)
    literals = [node for node in graph.nodes if node.op == "constant"]
    assert literals
    assert all(node.spec.size == 1 for node in literals)
    np.testing.assert_array_equal(expression(value), np.ones_like(value))


def test_like_creation_is_dtype_stable_and_differentiation_safe() -> None:
    @xp.compile(differentiable=("x",))
    def expression(x: object) -> object:
        return xp.sum(x * xp.ones_like(x) + xp.zeros_like(x))

    x = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    graph = expression.lower(x)
    result = tensor.jvp(graph, {"x": x}, {"x": np.ones_like(x)})
    np.testing.assert_array_equal(result.output_tangents["output"], 3.0)


def test_create_like_rejects_scientific_relabeling() -> None:
    occupied = xp.IndexSpace("occupied", "occupied", 2)
    spec = xp.TensorSpec((xp.Index("i", occupied),), role="input")
    annotated = xp.input_array("x", spec)
    for operation in (xp.ones_like, xp.zeros_like):
        with pytest.raises(TypeError, match="explicit TensorIR index"):
            operation(annotated)
    with pytest.raises(TypeError, match="explicit TensorIR index"):
        xp.full_like(annotated, 0.5)


def test_capture_context_is_always_restored_on_trace_error() -> None:
    spec = xp.TensorSpec((xp.Index("i", xp.IndexSpace("ao", "ao", 2)),), role="input")
    assert not active_capture()

    def fail(x: object) -> object:
        assert active_capture()
        assert isinstance(xp.ones((2,), dtype=xp.float64), xp.VibeArray)
        raise RuntimeError("intentional failure")

    with pytest.raises(RuntimeError, match="intentional failure"):
        xp.trace(fail, {"x": spec})
    assert not active_capture()
    assert isinstance(xp.ones((2,), dtype=xp.float64), np.ndarray)


@pytest.mark.parametrize(
    ("create", "args"),
    (
        (xp.ones, ((2,),)),
        (xp.zeros, ((2,),)),
        (xp.full, ((2,), 1.0)),
    ),
)
def test_explicit_unknown_device_is_not_silently_accepted(
    create: object, args: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError, match="device=None"):
        create(*args, device="cuda:0")
    with pytest.raises(ValueError, match="device=None"):
        xp.zeros_like(np.ones(2, dtype=np.float64), device="cuda:0")


def test_uniform_creation_rejects_unsupported_implicit_integer_and_dtype() -> None:
    with pytest.raises(TypeError, match="not supported"):
        xp.full((2,), 3)
    with pytest.raises(TypeError, match="float32/float64"):
        xp.ones((2,), dtype=np.int32)
    with pytest.raises(TypeError, match="float32/float64"):
        xp.full_like(np.ones(2), 2, dtype=np.int32)
    with pytest.raises(TypeError, match="nonnegative"):
        xp.zeros((-1,))
    with pytest.raises(TypeError, match="nonnegative"):
        xp.zeros((True,))
    with pytest.raises(ValueError, match="finite"):
        xp.full((2,), float("inf"), dtype=xp.float64)

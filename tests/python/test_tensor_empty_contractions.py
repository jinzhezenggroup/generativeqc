"""Empty contraction identities must not depend on out-of-domain backing bytes."""

from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.tensor import autodiff
from generativeqc_compiler.tensor.interpreter import execute


def empty_with_inaccessible_nan(
    dtype: str, shape: tuple[int, ...], strides: tuple[int, ...]
) -> Any:
    backing = np.array([np.nan], dtype=dtype)
    value = np.ndarray(shape, dtype=dtype, buffer=backing, strides=strides)
    assert value.size == 0 and np.isfinite(value).all()
    return value


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("coefficient", [1, -2])
def test_public_and_reference_empty_forward_jvp_vjp(
    dtype: str, coefficient: int
) -> None:
    left = empty_with_inaccessible_nan(dtype, (2, 0), (0, 0))
    right = np.empty((1, 0), dtype=dtype)
    source = xp.compile(
        lambda a, b: xp.vecdot(a, b) * coefficient,
        backend="reference",
        differentiable=("a", "b"),
    )
    try:
        feeds = {"a": left, "b": right}
        program = source.lower(left, right)
        expected = np.zeros(2, dtype=dtype)
        np.testing.assert_array_equal(source(left, right), expected)
        np.testing.assert_array_equal(
            execute(program, feeds).outputs["output"], expected
        )
        forward = autodiff.jvp(program, feeds, feeds)
        np.testing.assert_array_equal(forward.output_tangents["output"], expected)
        reverse = autodiff.vjp(program, feeds, {"output": np.ones(2, dtype=dtype)})
        for name, value in feeds.items():
            assert reverse.input_cotangents[name].shape == value.shape
            assert reverse.input_cotangents[name].dtype == value.dtype
        primal, tangent = xp.jvp(source, (left, right), (left, right))
        np.testing.assert_array_equal(primal, expected)
        np.testing.assert_array_equal(tangent, expected)
        _, pullback = xp.vjp(source, left, right, argnums=(0, 1))
        try:
            result = pullback(np.ones(2, dtype=dtype))
            assert result[0].shape == left.shape and result[1].shape == right.shape
        finally:
            pullback.close()
    finally:
        source.close()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "equation,left_shape,right_shape,output_shape",
    [
        ("ij,jk->ik", (2, 0), (0, 3), (2, 3)),
        ("ij,ij->i", (0, 3), (0, 3), (0,)),
        ("i,i->", (0,), (0,), ()),
        ("i,j->ij", (0,), (3,), (0, 3)),
        ("ii,->", (0, 0), (), ()),
        ("ii,->i", (0, 0), (), (0,)),
        ("ij,j->i", (2, 0), (0,), (2,)),
    ],
)
def test_empty_output_and_reduction_shapes(
    dtype: str,
    equation: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    left, right = np.ones(left_shape, dtype=dtype), np.ones(right_shape, dtype=dtype)
    source = xp.compile(
        lambda a, b: xp.einsum(equation, a, b, coefficient=-2),
        backend="reference",
        differentiable=("a", "b"),
    )
    try:
        feeds = {"a": left, "b": right}
        program = source.lower(left, right)
        expected = np.zeros(output_shape, dtype=dtype) * -2
        actual = source(left, right)
        assert actual.shape == output_shape and actual.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
        forward = autodiff.jvp(program, feeds, feeds)
        np.testing.assert_array_equal(forward.output_tangents["output"], expected)
        reverse = autodiff.vjp(
            program, feeds, {"output": np.ones(output_shape, dtype=dtype)}
        )
        for name, value in feeds.items():
            np.testing.assert_array_equal(
                reverse.input_cotangents[name], np.zeros_like(value)
            )
            assert reverse.input_cotangents[name].dtype == value.dtype
    finally:
        source.close()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_empty_contraction_keeps_live_finite_validation(bad: float) -> None:
    source = xp.compile(lambda a, b: xp.einsum("ij,i->i", a, b), backend="reference")
    try:
        with pytest.raises(ValueError, match="finite"):
            source(np.empty((2, 0)), np.array([bad, 1.0]))
    finally:
        source.close()


def test_empty_contraction_keeps_output_budget() -> None:
    source = xp.compile(lambda a, b: xp.vecdot(a, b), backend="reference", max_bytes=32)
    try:
        with pytest.raises(ValueError, match="budget"):
            source(np.empty((16, 0)), np.empty((1, 0)))
    finally:
        source.close()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_empty_contraction_keeps_finite_tangent_and_cotangent_validation(
    dtype: str, bad: float
) -> None:
    left = np.empty((2, 0), dtype=dtype)
    right = np.ones(2, dtype=dtype)
    invalid = np.array([bad, 1.0], dtype=dtype)
    source = xp.compile(
        lambda a, b: xp.einsum("ij,i->i", a, b),
        backend="reference",
        differentiable=("a", "b"),
    )
    try:
        feeds = {"a": left, "b": right}
        program = source.lower(left, right)
        with pytest.raises(ValueError, match="finite"):
            autodiff.jvp(program, feeds, {"a": left, "b": invalid})
        with pytest.raises(ValueError, match="finite"):
            autodiff.vjp(program, feeds, {"output": invalid})
        with pytest.raises(ValueError, match="finite"):
            xp.jvp(source, (left, right), (left, invalid))
        _, pullback = xp.vjp(source, left, right, argnums=(0, 1))
        try:
            with pytest.raises(ValueError, match="finite"):
                pullback(invalid)
        finally:
            pullback.close()
    finally:
        source.close()


@pytest.mark.parametrize(
    "dtype,coefficient", [("float32", 10**100), ("float64", 10**400)]
)
def test_empty_contraction_keeps_coefficient_representability(
    dtype: str, coefficient: int
) -> None:
    left = np.empty((2, 0), dtype=dtype)
    right = np.empty((2, 0), dtype=dtype)
    source = xp.compile(
        lambda a, b: xp.einsum("ij,ij->i", a, b, coefficient=coefficient),
        backend="reference",
        differentiable=("a", "b"),
    )
    try:
        program = source.lower(left, right)
        feeds = {"a": left, "b": right}
        with pytest.raises(ValueError, match="finite"):
            execute(program, feeds)
        with pytest.raises(ValueError, match="finite"):
            autodiff.jvp(program, feeds, feeds)
        with pytest.raises(ValueError, match="finite"):
            autodiff.vjp(program, feeds, {"output": np.ones(2, dtype=dtype)})
    finally:
        source.close()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_empty_reduction_nonempty_operand_adjoint(dtype: str) -> None:
    empty = empty_with_inaccessible_nan(dtype, (2, 0), (0, 0))
    vector = np.array([2.0, 3.0], dtype=dtype)
    source = xp.compile(
        lambda a, b: xp.einsum("ij,i->i", a, b),
        backend="reference",
        differentiable=("a", "b"),
    )
    try:
        feeds = {"a": empty, "b": vector}
        program = source.lower(empty, vector)
        expected = np.zeros(2, dtype=dtype)
        np.testing.assert_array_equal(source(empty, vector), expected)
        forward = autodiff.jvp(program, feeds, feeds)
        np.testing.assert_array_equal(forward.output_tangents["output"], expected)
        reverse = autodiff.vjp(
            program, feeds, {"output": np.array([0.5, 0.75], dtype=dtype)}
        )
        np.testing.assert_array_equal(reverse.input_cotangents["b"], expected)
        assert reverse.input_cotangents["a"].shape == empty.shape
    finally:
        source.close()

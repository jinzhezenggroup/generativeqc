"""Experimental public Array API facade over the canonical TensorIR frontend."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from fractions import Fraction

import numpy as np
import pytest
from generativeqc.experimental import API_VERSION as EXPERIMENTAL_API_VERSION
from generativeqc.experimental import array_api as xp
from generativeqc.extensions import tensor
from generativeqc_compiler.array_api import namespace as compiler_xp
from generativeqc_compiler.array_api import trace as compiler_trace


def _vector_spec() -> xp.TensorSpec:
    ao = xp.IndexSpace("ao", "ao", 3)
    return xp.TensorSpec((xp.Index("p", ao),), role="input")


def test_experimental_package_keeps_array_surface_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import generativeqc.experimental as experimental

                assert experimental.API_VERSION == 1
                assert experimental.__all__ == ["API_VERSION", "array_api"]
                assert "array_api" in dir(experimental)
                assert "generativeqc.experimental.array_api" not in sys.modules
                assert not any(
                    name.startswith("generativeqc_compiler.array_api")
                    for name in sys.modules
                )
                """
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_preview_reuses_canonical_tensorir_identity_and_execution() -> None:
    spec = _vector_spec()
    public = xp.trace(
        lambda x: {"out": xp.sum(Fraction(1, 2) * x * x)},
        {"x": spec},
    )
    internal = compiler_trace(
        lambda x: {"out": compiler_xp.sum(Fraction(1, 2) * x * x)},
        {"x": spec},
    )

    assert isinstance(public, xp.Program)
    assert public.logical_hash == internal.logical_hash
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    execution = tensor.execute(public, {"x": values})
    assert execution.outputs["out"] == 7.0


def test_public_namespace_runs_eagerly_and_compiles_the_same_expression() -> None:
    def expression(x: object, y: object) -> object:
        return xp.sum(xp.sqrt((x + 0.5) * y))

    x = xp.asarray([0.5, 1.5, 3.5], dtype=xp.float64)
    y = xp.asarray([2.0, 3.0, 4.0], dtype=xp.float64)
    eager = expression(x, y)
    expected = np.sum(np.sqrt((x + 0.5) * y))
    np.testing.assert_allclose(eager, expected)

    compiled = xp.compile(expression)
    np.testing.assert_allclose(compiled(x, y), eager)


def test_eager_namespace_array_manipulation_matches_numpy() -> None:
    matrix = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float64)
    vector = xp.asarray([1.0, 2.0], dtype=xp.float64)

    np.testing.assert_array_equal(xp.reshape(vector, (1, 2)), vector.reshape(1, 2))
    np.testing.assert_array_equal(
        xp.broadcast_to(vector, (3, 2)), np.broadcast_to(vector, (3, 2))
    )
    np.testing.assert_array_equal(xp.permute_dims(matrix, (1, 0)), matrix.T)
    np.testing.assert_array_equal(xp.matrix_transpose(matrix), matrix.T)
    np.testing.assert_array_equal(xp.matmul(matrix, vector), matrix @ vector)
    np.testing.assert_array_equal(xp.take(matrix, (1, 0), axis=0), matrix[[1, 0]])


def test_compiled_observable_uses_array_syntax_without_tensor_specs() -> None:
    @xp.compile
    def observable(C: object, occupation: object, O: object) -> object:
        density = (C * occupation) @ C.T
        return xp.sum(density * O)

    coefficients = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    occupation = xp.asarray([2.0, 1.0], dtype=xp.float64)
    operator = np.eye(3, dtype=np.float64)

    actual = observable(coefficients, occupation, operator)
    weighted = coefficients * occupation
    expected = np.sum((weighted @ coefficients.T) * operator)
    np.testing.assert_allclose(actual, expected)

    program = observable.lower(coefficients, occupation, operator)
    assert isinstance(program, xp.Program)
    assert {node.attrs["name"] for node in program.nodes if node.op == "input"} == {
        "C",
        "occupation",
        "O",
    }


def test_generic_indexing_reshape_and_batched_matmul_match_numpy() -> None:
    @xp.compile
    def transform(x: object, y: object) -> object:
        column = x[::-1, None]
        flat = xp.reshape(column, (1, -1))
        return flat @ y

    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    actual = transform(x, y)
    expected = x[::-1, None].reshape(1, -1) @ y
    np.testing.assert_allclose(actual, expected)

    @xp.compile
    def batched(left: object, right: object) -> object:
        return left @ right

    left = np.arange(12.0, dtype=np.float64).reshape(2, 2, 3)
    right = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
    np.testing.assert_allclose(batched(left, right), left @ right)


def test_generic_empty_slices_match_numpy() -> None:
    @xp.compile
    def empty_slice(x: object) -> object:
        return x[2:1]

    values = np.arange(3.0, dtype=np.float64)
    np.testing.assert_array_equal(empty_slice(values), values[2:1])

    @xp.compile
    def empty_multiaxis(x: object) -> object:
        return x[1:1, ::-1]

    matrix = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    np.testing.assert_array_equal(empty_multiaxis(matrix), matrix[1:1, ::-1])


def test_compiled_lower_supports_explicit_differentiable_inputs() -> None:
    @xp.compile(differentiable=("x",))
    def norm2(x: object) -> object:
        return xp.sum(x * x)

    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    program = norm2.lower(values)
    x_input = next(
        node
        for node in program.nodes
        if node.op == "input" and node.attrs["name"] == "x"
    )
    assert x_input.spec.differentiable is True
    result = tensor.jvp(
        program,
        {"x": values},
        {"x": np.ones_like(values)},
    )
    assert result.output_tangents["output"] == 12.0

    @xp.compile
    def fixed(x: object) -> object:
        return xp.sum(x * x)

    fixed_program = fixed.lower(values)
    fixed_input = next(
        node
        for node in fixed_program.nodes
        if node.op == "input" and node.attrs["name"] == "x"
    )
    assert fixed_input.spec.differentiable is False


def test_generic_broadcasting_and_exact_scalar_operators_match_numpy() -> None:
    @xp.compile
    def expression(matrix: object, vector: object) -> object:
        return (matrix + 1) * vector - Fraction(1, 2)

    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = (matrix + 1.0) * vector - 0.5
    np.testing.assert_allclose(expression(matrix, vector), expected)


def test_public_preview_declares_experimental_nonconformance() -> None:
    assert EXPERIMENTAL_API_VERSION == 1
    assert xp.API_VERSION == 1
    report = xp.capabilities()
    assert report["public_api_version"] == 1
    assert report["surface"] == "array-api-shaped-experimental-public-preview"
    assert report["stability"] == "experimental"
    assert report["import_path"] == "generativeqc.experimental.array_api"
    assert report["array_api_version"] is None
    assert report["array_namespace_protocol"] is False
    assert report["implicit_broadcast"] is True
    assert report["reshape_requires_explicit_indices"] is False
    assert report["scientific_metadata_requires_explicit_indices"] is True
    assert report["compiled_call"] == "shape-dtype-specialized-tensorir-reference"
    assert report["compiled_differentiability"] == "explicit-parameter-names"
    assert report["namespace_dispatch"] == "numpy-eager-or-symbolic-tensorir"

    value = xp.input_array("x", _vector_spec())
    assert isinstance(value, xp.VibeArray)
    assert value.size == 3
    assert not hasattr(value, "__array_namespace__")


def test_public_preview_exports_array_style_operations() -> None:
    for name in (
        "add",
        "asarray",
        "broadcast_to",
        "compile",
        "divide",
        "einsum",
        "exp",
        "log",
        "matmul",
        "matrix_transpose",
        "mean",
        "multiply",
        "negative",
        "permute_dims",
        "pow",
        "reciprocal",
        "reshape",
        "slice",
        "sqrt",
        "square",
        "subtract",
        "sum",
        "take",
    ):
        assert callable(getattr(xp, name))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("axis", (None, 0, -1, (0, 2), ()))
@pytest.mark.parametrize("keepdims", (False, True))
def test_mean_and_sum_keepdims_eager_compiled_numpy_parity(
    dtype: object, axis: object, keepdims: bool
) -> None:
    values = np.arange(1, 25, dtype=dtype).reshape(2, 3, 4)
    for operation, oracle in ((xp.sum, np.sum), (xp.mean, np.mean)):
        def expression(x: object) -> object:
            return operation(x, axis=axis, keepdims=keepdims)

        eager = expression(values)
        compiled = xp.compile(expression)(values)
        expected = oracle(values, axis=axis, keepdims=keepdims)
        assert eager.dtype == compiled.dtype == expected.dtype
        assert eager.shape == compiled.shape == expected.shape
        np.testing.assert_allclose(eager, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(compiled, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_square_and_reciprocal_eager_compiled_negative_values(dtype: object) -> None:
    values = np.asarray([-4.0, -2.0, 0.5, 2.0], dtype=dtype)
    for operation, oracle in (
        (xp.square, np.square),
        (xp.reciprocal, np.reciprocal),
    ):
        expected = oracle(values)
        eager = operation(values)
        compiled = xp.compile(operation)(values)
        assert eager.dtype == compiled.dtype == values.dtype
        np.testing.assert_allclose(eager, expected)
        np.testing.assert_allclose(compiled, expected)


def test_mean_empty_reduction_is_explicitly_unsupported() -> None:
    values = np.empty((0, 2), dtype=np.float64)
    for operation in (
        lambda x: xp.mean(x, axis=0),
        lambda x: xp.mean(x, keepdims=True),
    ):
        with pytest.raises(ValueError, match="empty reduction"):
            operation(values)
        with pytest.raises(ValueError, match="empty reduction"):
            xp.compile(operation).lower(values)

    # The empty sum is well-defined and its singleton dimension is preserved.
    np.testing.assert_array_equal(
        xp.compile(lambda x: xp.sum(x, axis=0, keepdims=True))(values),
        np.zeros((1, 2), dtype=np.float64),
    )


def test_mean_keeps_tensorir_autodiff_and_scientific_guards() -> None:
    @xp.compile(differentiable=("x",))
    def expression(x: object) -> object:
        return xp.mean(xp.square(x), axis=-1, keepdims=True)

    values = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    tangent = np.asarray([[1.0, 0.0, -1.0], [1.0, 2.0, -2.0]])
    program = expression.lower(values)
    response = tensor.jvp(program, {"x": values}, {"x": tangent})
    np.testing.assert_allclose(
        response.output_tangents["output"],
        np.mean(2 * values * tangent, axis=-1, keepdims=True),
    )

    annotated = xp.input_array("x", _vector_spec())
    for operation in (
        lambda x: xp.sum(x, keepdims=True),
        lambda x: xp.mean(x, keepdims=True),
    ):
        with pytest.raises(ValueError, match="explicit TensorIR index metadata"):
            operation(annotated)
    with pytest.raises(TypeError, match="scientifically annotated"):
        xp.reciprocal(annotated)


def test_asarray_rejects_implicit_external_device_transfer() -> None:
    class External:
        def __dlpack_device__(self) -> tuple[int, int]:
            return (2, 0)

    with np.testing.assert_raises_regex(TypeError, "explicit handoff"):
        xp.asarray(External())


_EAGER_ARRAY_OPERATIONS = (
    xp.negative,
    xp.exp,
    xp.log,
    xp.sqrt,
    xp.square,
    xp.reciprocal,
    xp.mean,
    lambda x: xp.pow(x, 2),
    lambda x: xp.reshape(x, (4,)),
    lambda x: xp.broadcast_to(x, (3, 2, 2)),
    lambda x: xp.slice(x, ((0, 1), (0, 2))),
    lambda x: xp.take(x, (1, 0), axis=0),
    xp.sum,
    lambda x: xp.permute_dims(x, (1, 0)),
    xp.matrix_transpose,
    lambda x: xp.add(x, 0.5),
    lambda x: xp.subtract(0.5, x),
    lambda x: xp.multiply(x, 0.5),
    lambda x: xp.divide(0.5, x),
    lambda x: xp.matmul(x, np.eye(2)),
    lambda x: xp.matmul(np.eye(2), x),
    lambda x: xp.einsum("ij,jk->ik", x, np.eye(2)),
    lambda x: xp.einsum("ij,jk->ik", np.eye(2), x),
)


@pytest.mark.parametrize("operation", _EAGER_ARRAY_OPERATIONS)
@pytest.mark.parametrize("device", (1, 2))
def test_eager_namespace_rejects_foreign_arrays_before_conversion(
    operation: object, device: int
) -> None:
    class External:
        def __dlpack_device__(self) -> tuple[int, int]:
            return device, 0

        def __array__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("foreign conversion must not be invoked")

        def __array_ufunc__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("foreign ufunc must not be invoked")

        def __array_function__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("foreign array function must not be invoked")

    with pytest.raises(TypeError, match="explicit handoff"):
        operation(External())


@pytest.mark.parametrize("operation", _EAGER_ARRAY_OPERATIONS)
@pytest.mark.parametrize("dtype", (np.int64, np.float16, np.complex64, object))
def test_eager_namespace_rejects_unsupported_array_dtypes(
    operation: object, dtype: object
) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        operation(np.ones((2, 2), dtype=dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("scalar", (2, Fraction(1, 2), "1/2", 0.5))
@pytest.mark.parametrize("operation", (xp.add, xp.subtract, xp.multiply, xp.divide))
@pytest.mark.parametrize("reverse", (False, True))
def test_eager_exact_scalar_arithmetic_matches_compiled_dtype_and_values(
    dtype: object, scalar: object, operation: object, reverse: bool
) -> None:
    def expression(x: object) -> object:
        return operation(scalar, x) if reverse else operation(x, scalar)

    values = np.asarray([2.0, 8.0], dtype=dtype)
    eager = expression(values)
    compiled = xp.compile(expression)(values)
    numeric = dtype(float(Fraction(scalar)))
    numpy_operation = getattr(np, operation.__name__)
    expected = (
        numpy_operation(numeric, values)
        if reverse
        else numpy_operation(values, numeric)
    )
    assert eager.dtype == compiled.dtype == values.dtype
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(compiled, expected)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("exponent", (2, Fraction(1, 2), "1/2", 0.5))
def test_eager_exact_scalar_power_matches_compiled(
    dtype: object, exponent: object
) -> None:
    def expression(x: object) -> object:
        return xp.pow(x, exponent)

    values = np.asarray([2.0, 8.0], dtype=dtype)
    eager = expression(values)
    compiled = xp.compile(expression)(values)
    expected = np.power(values, dtype(float(Fraction(exponent))))
    assert eager.dtype == compiled.dtype == values.dtype
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(compiled, expected)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_eager_exact_scalar_composes_with_array_functions(dtype: object) -> None:
    def expression(x: object) -> object:
        return xp.sqrt(xp.multiply(x, Fraction(1, 2)))

    values = np.asarray([2.0, 8.0], dtype=dtype)
    expected = np.asarray([1.0, 2.0], dtype=dtype)
    np.testing.assert_array_equal(expression(values), expected)
    np.testing.assert_array_equal(xp.compile(expression)(values), expected)


@pytest.mark.parametrize(
    "operation",
    (
        xp.add,
        xp.subtract,
        xp.multiply,
        xp.divide,
        xp.matmul,
        lambda x, y: xp.einsum("ij,jk->ik", x, y),
    ),
)
def test_eager_namespace_rejects_implicit_array_dtype_promotion(
    operation: object,
) -> None:
    left = np.ones((2, 2), dtype=np.float32)
    right = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        operation(left, right)
    with pytest.raises(ValueError, match="dtype"):
        xp.compile(lambda x, y: operation(x, y))(left, right)


@pytest.mark.parametrize("scalar", (float("nan"), float("inf"), -float("inf"), -0.0))
@pytest.mark.parametrize(
    "operation", (xp.add, xp.subtract, xp.multiply, xp.divide, xp.pow)
)
def test_eager_and_compiled_reject_unrepresentable_scalar_literals(
    scalar: float, operation: object
) -> None:
    values = np.ones(2, dtype=np.float64)
    match = "negative-zero" if scalar == 0 else "finite"
    with pytest.raises(ValueError, match=match):
        operation(values, scalar)
    with pytest.raises(ValueError, match=match):
        xp.compile(lambda x: operation(x, scalar)).lower(values)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_finite_float_scalar_arithmetic_preserves_opt_in_ad(dtype: object) -> None:
    def expression(x: object) -> object:
        return xp.sum((x + 0.3) * 0.5 / 0.7)

    values = np.asarray([0.2, 0.7, 2.1], dtype=dtype)
    tangent = np.asarray([0.3, -0.4, 0.6], dtype=dtype)
    program = xp.compile(expression, differentiable=("x",)).lower(values)
    forward = tensor.jvp(program, {"x": values}, {"x": tangent})
    reverse = tensor.vjp(
        program,
        {"x": values},
        {"output": np.asarray(1.0, dtype=dtype)},
    )
    np.testing.assert_allclose(
        forward.output_tangents["output"], np.sum(tangent * (0.5 / 0.7)), rtol=2e-6
    )
    np.testing.assert_allclose(
        reverse.input_cotangents["x"], np.full_like(values, 0.5 / 0.7), rtol=2e-6
    )
    fixed = xp.compile(expression).lower(values)
    assert all(
        not node.spec.differentiable for node in fixed.nodes if node.op == "input"
    )


@pytest.mark.parametrize(
    "operation",
    (
        lambda x: xp.sum(x, keepdims=0),
        lambda x: xp.mean(x, keepdims=0),
        lambda x: xp.mean(x, axis=[0]),
        lambda x: xp.mean(x, axis=(0, 0)),
        lambda x: xp.sum(x, dtype=xp.float64),
        lambda x: xp.sum(x, axis=[0]),
        lambda x: xp.sum(x, axis=True),
        lambda x: xp.sum(x, axis=(0, 0)),
        lambda x: xp.take(x, (-1,), axis=0),
        lambda x: xp.take(x, (2,), axis=0),
        lambda x: xp.take(x, [0], axis=0),
        lambda x: xp.take(x, (True,), axis=0),
        lambda x: xp.take(x, (0,), axis=True),
        lambda x: xp.slice(x, ((-1, 2), (0, 2))),
        lambda x: xp.slice(x, ((1, 0), (0, 2))),
        lambda x: xp.slice(x, ((0, 3), (0, 2))),
        lambda x: xp.slice(x, ((0, 2),)),
        lambda x: xp.slice(x, ([0, 2], [0, 2])),
        lambda x: xp.reshape(x, [4]),
        lambda x: xp.broadcast_to(x, [2, 2]),
        lambda x: xp.permute_dims(x, (-1, -2)),
    ),
)
def test_eager_and_compiled_reject_unsupported_static_controls(
    operation: object,
) -> None:
    values = np.ones((2, 2), dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        operation(values)
    with pytest.raises((TypeError, ValueError)):
        xp.compile(lambda x: operation(x)).lower(values)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "operation",
    (
        lambda x: xp.sum(x, axis=-1),
        lambda x: xp.sum(x, axis=()),
        lambda x: xp.take(x, (), axis=-1),
        lambda x: xp.take(x, (1, 0, 1), axis=-1),
        lambda x: xp.slice(x, ((1, 1), (0, 2))),
    ),
)
def test_eager_and_compiled_preserve_supported_static_controls(
    operation: object, dtype: object
) -> None:
    values = np.arange(4.0, dtype=dtype).reshape(2, 2)
    eager = operation(values)
    compiled = xp.compile(lambda x: operation(x))(values)
    assert eager.dtype == compiled.dtype == values.dtype
    np.testing.assert_array_equal(eager, compiled)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    ("operation", "values"),
    (
        (lambda x: xp.pow(x, 2), [-2.0]),
        (lambda x: xp.pow(x, 2), [0.0]),
        (xp.log, [0.0]),
        (xp.log, [-1.0]),
        (xp.sqrt, [-1.0]),
        (lambda x: xp.divide(x, 0), [1.0]),
        (lambda x: xp.divide(1, x), [0.0]),
        (xp.exp, [1000.0]),
        (lambda x: xp.multiply(x, x), [1.0e30, 1.0e308]),
        (xp.negative, [float("nan")]),
        (xp.negative, [float("inf")]),
    ),
)
def test_eager_and_compiled_reject_invalid_real_domains_and_nonfinite_results(
    dtype: object, operation: object, values: object
) -> None:
    with np.errstate(all="ignore"):
        array = np.asarray(values, dtype=dtype)
        with pytest.raises(ValueError):
            operation(array)
        with pytest.raises(ValueError):
            xp.compile(lambda x: operation(x))(array)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_eager_einsum_exact_coefficient_matches_compiled(dtype: object) -> None:
    def expression(x: object) -> object:
        return xp.einsum("i,i->", x, x, coefficient="1/2")

    values = np.asarray([1.0, 2.0], dtype=dtype)
    eager = expression(values)
    compiled = xp.compile(expression)(values)
    assert eager.dtype == compiled.dtype == values.dtype
    np.testing.assert_array_equal(eager, dtype(2.5))
    np.testing.assert_array_equal(compiled, dtype(2.5))
    with pytest.raises(TypeError, match="exact"):
        xp.einsum("i->", values, coefficient=0.5)
    with pytest.raises(TypeError, match="exact"):
        xp.compile(lambda x: xp.einsum("i->", x, coefficient=0.5)).lower(values)


@pytest.mark.parametrize("exponent", (1.0e-100, -1.0e-100, 1.0e100))
def test_eager_and_compiled_reject_unrepresentable_float32_power_exponents(
    exponent: float,
) -> None:
    values = np.asarray([2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        xp.pow(values, exponent)
    with pytest.raises(ValueError):
        xp.compile(lambda x: xp.pow(x, exponent)).lower(values)


@pytest.mark.parametrize("protocol", ("dlpack", "cuda_array_interface"))
@pytest.mark.parametrize("container", ("list", "tuple", "nested", "object_array"))
@pytest.mark.parametrize("mode", ("asarray", "eager", "compiled"))
def test_nested_foreign_arrays_are_rejected_before_conversion(
    protocol: str, container: str, mode: str
) -> None:
    class External:
        def __array__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("nested foreign conversion must not be invoked")

    external = External()
    if protocol == "dlpack":
        external.__dlpack_device__ = lambda: (2, 0)
    else:
        external.__cuda_array_interface__ = {"version": 3}
    if container == "list":
        values = [external]
    elif container == "tuple":
        values = (external,)
    elif container == "nested":
        values = ([external],)
    else:
        values = np.empty(1, dtype=object)
        values[0] = external
    with pytest.raises(TypeError, match="explicit handoff"):
        if mode == "asarray":
            xp.asarray(values, dtype=xp.float64)
        elif mode == "eager":
            xp.negative(values)
        else:
            xp.compile(lambda x: -x)(values)


@pytest.mark.parametrize("operation", _EAGER_ARRAY_OPERATIONS)
def test_eager_namespace_rejects_cuda_array_interface_before_conversion(
    operation: object,
) -> None:
    class External:
        def __init__(self) -> None:
            self.__cuda_array_interface__ = {"version": 3}

        def __array__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("foreign conversion must not be invoked")

    with pytest.raises(TypeError, match="explicit handoff"):
        operation(External())


@pytest.mark.parametrize(
    "operation",
    (
        lambda x, y: xp.einsum("i", x),
        lambda x, y: xp.einsum("...->...", x),
        lambda x, y: xp.einsum("i,i->i", x, y),
        lambda x, y: xp.einsum("i,i->", x, y),
    ),
)
def test_eager_einsum_rejects_notation_and_broadcasting_outside_tensorir(
    operation: object,
) -> None:
    x = np.asarray([1.0, 2.0])
    y = np.ones(1, dtype=np.float64)
    with pytest.raises(ValueError):
        operation(x, y)
    with pytest.raises(ValueError):
        xp.compile(lambda x, y: operation(x, y)).lower(x, y)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("equation", ("ij->ji", "ii->i", "ii->", "ij->i"))
def test_eager_einsum_preserves_explicit_diagonals_and_axis_labels(
    dtype: object, equation: str
) -> None:
    values = np.arange(4.0, dtype=dtype).reshape(2, 2)
    eager = xp.einsum(equation, values)
    compiled = xp.compile(lambda x: xp.einsum(equation, x))(values)
    expected = np.einsum(equation, values, optimize=False)
    assert eager.dtype == compiled.dtype == values.dtype
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(compiled, expected)


def test_host_admission_preserves_nested_values_and_handles_cycles() -> None:
    values = ([1.0, 2.0], [3.0, 4.0])
    np.testing.assert_array_equal(xp.asarray(values), np.asarray(values))
    objects = np.asarray(values, dtype=object)
    np.testing.assert_array_equal(
        xp.asarray(objects, dtype=xp.float64), np.asarray(values)
    )
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError):
        xp.asarray(cyclic)

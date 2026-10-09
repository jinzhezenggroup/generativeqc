"""Finite real comparisons and first-class Boolean TensorIR data."""

from __future__ import annotations

import builtins
import typing

import array_api_strict as strict
import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import namespace as compiler_namespace
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.precision import PrecisionDirective
from generativeqc_compiler.tensor import compare as tensor_compare
from generativeqc_compiler.tensor import ir
from generativeqc_compiler.tensor.ad_program import linearize, transpose_program
from generativeqc_compiler.tensor.autodiff import dot_test, jvp, vjp
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.optimize import optimize, prepare_for_backend
from generativeqc_compiler.tensor.precision import describe_precision, lower_precision
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

COMPARISONS = (
    ("equal", np.equal),
    ("not_equal", np.not_equal),
    ("greater", np.greater),
    ("greater_equal", np.greater_equal),
    ("less", np.less),
    ("less_equal", np.less_equal),
)


@pytest.mark.parametrize("name,oracle", COMPARISONS)
@pytest.mark.parametrize(
    "left,right",
    (
        (np.array(1.0, dtype=np.float64), np.array(2.0, dtype=np.float32)),
        (
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[1.0, 3.0]], dtype=np.float64),
        ),
        (np.empty((0, 1), dtype=np.float64), np.ones((1, 3), dtype=np.float32)),
        (
            np.array([-0.0, 1.0], dtype=np.float64),
            np.array([0.0, -1.0], dtype=np.float64),
        ),
    ),
)
def test_eager_and_captured_comparisons_match_numpy(
    name: str,
    oracle: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    expected = oracle(left, right)
    operation = getattr(xp, name)
    eager = operation(left, right)
    compiled = xp.compile(lambda x, y: operation(x, y))
    program = compiled.lower(left, right)
    actual = compiled(left, right)
    assert eager.dtype == actual.dtype == expected.dtype == np.dtype("bool")
    assert program.outputs["output"].spec.itemsize == 1
    assert program.outputs["output"].spec.shape == expected.shape
    if left.dtype != right.dtype:
        assert any(node.op == "cast" for node in program.nodes)
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(actual, expected)


def test_operator_comparisons_remain_symbolic_and_immutable() -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    original = values.copy()
    operations = (
        (lambda x: x == 2, np.equal(values, 2)),
        (lambda x: x != 2, np.not_equal(values, 2)),
        (lambda x: x > 2, np.greater(values, 2)),
        (lambda x: x >= 2, np.greater_equal(values, 2)),
        (lambda x: x < 2, np.less(values, 2)),
        (lambda x: x <= 2, np.less_equal(values, 2)),
    )
    for expression, oracle in operations:
        program = xp.compile(expression).lower(values)
        result = execute(program, {"x": values}).outputs["output"]
        np.testing.assert_array_equal(result, oracle)
        assert result.dtype == np.dtype("bool")
        assert not np.shares_memory(result, values)
    np.testing.assert_array_equal(values, original)


def test_bool_input_constants_views_and_identity_roundtrip() -> None:
    spec = TensorSpec(
        (Index("i", IndexSpace("mask", "matrix", 3)),), dtype="bool", role="input"
    )
    mask = ir.input_tensor("mask", spec)
    literal = ir.constant(
        (True, False, True), TensorSpec(spec.indices, dtype="bool", role="constant")
    )
    viewed = ir.reshape(ir.transpose(mask, (0,)), spec.indices)
    program = Program({"input": viewed, "literal": literal})
    restored = Program.loads(program.dumps())
    assert program.logical_hash == restored.logical_hash
    assert program.dumps() == restored.dumps()
    assert all(node.spec.itemsize == 1 for node in program.nodes)
    bool_literal = ir.constant(True, TensorSpec(dtype="bool", role="constant"))
    real_literal = ir.constant(1, TensorSpec(role="constant"))
    assert bool_literal.attributes != real_literal.attributes
    assert (
        Program({"out": bool_literal}).logical_hash
        != Program({"out": real_literal}).logical_hash
    )
    values = np.asarray([True, False, True], dtype=np.bool_)
    result = execute(restored, {"mask": values})
    np.testing.assert_array_equal(result.outputs["input"], values)
    np.testing.assert_array_equal(result.outputs["literal"], values)
    assert not np.shares_memory(result.outputs["input"], values)
    assert result.outputs["input"].dtype == np.dtype("bool")
    with pytest.raises(ValueError, match="retained-byte budget"):
        execute(restored, {"mask": values}, max_bytes=11)


def test_public_bool_admission_and_bounded_creation() -> None:
    mask = xp.asarray([True, False], dtype=xp.bool)
    assert mask.dtype == xp.bool
    assert xp.isdtype(xp.bool, "bool")
    assert not xp.isdtype(xp.bool, "numeric")
    assert xp.can_cast(xp.bool, xp.bool)
    assert not xp.can_cast(xp.bool, xp.float64)
    assert xp.result_type(xp.bool, xp.bool) == xp.bool
    with pytest.raises(TypeError, match="promotion"):
        xp.result_type(xp.bool, xp.float64)
    for scalar in (1, 1.0):
        with pytest.raises(TypeError, match="promotion"):
            xp.result_type(xp.bool, scalar)
        with pytest.raises(TypeError, match="promotion"):
            xp.result_type(scalar, xp.bool)
    np.testing.assert_array_equal(xp.full((2,), True), [True, True])
    np.testing.assert_array_equal(xp.full((2,), True, dtype=xp.bool), [True, True])
    np.testing.assert_array_equal(xp.zeros((2,), dtype=xp.bool), [False, False])
    np.testing.assert_array_equal(xp.ones_like(mask), [True, True])
    compiled = xp.compile(lambda x: xp.full_like(x, True))
    program = compiled.lower(mask)
    assert program.outputs["output"].spec.dtype == "bool"
    np.testing.assert_array_equal(compiled(mask), [True, True])
    with pytest.raises(TypeError, match="bool fill"):
        xp.full((2,), 1, dtype=xp.bool)
    with pytest.raises(TypeError, match="cross-kind"):
        xp.asarray(mask, dtype=xp.float64)
    with pytest.raises(TypeError, match="floating arithmetic"):
        xp.add(mask, mask)
    with pytest.raises(TypeError, match="array operand"):
        xp.equal(1.0, 2.0)


def test_mixed_bool_real_host_containers_fail_before_numpy_promotion() -> None:
    mixed_values = (
        [True, 0.5],
        [[True, False], [0.25, 0.75]],
        [np.asarray([True, False]), np.asarray([0.25, 0.75])],
    )
    for value in mixed_values:
        with pytest.raises(TypeError, match="mixed bool/real"):
            xp.asarray(value)
        with pytest.raises(TypeError, match="mixed bool/real"):
            xp.asarray(value, dtype=xp.float64)
    with pytest.raises(TypeError, match="mixed bool/real"):
        xp.add([True, 0.5], np.ones(2, dtype=np.float64))

    np.testing.assert_array_equal(
        xp.asarray([True, False], dtype=xp.bool), [True, False]
    )
    np.testing.assert_array_equal(
        xp.add([0.25, 0.5], np.ones(2, dtype=np.float64)), [1.25, 1.5]
    )
    deep: object = True
    for _ in range(65):
        deep = [deep]
    with pytest.raises(ValueError, match="nesting"):
        xp.asarray(deep)


def test_host_array_like_uses_its_unforced_source_kind() -> None:
    class BooleanArrayLike:
        def __array__(
            self, dtype: object = None, copy: builtins.bool | None = None
        ) -> np.ndarray:
            values = np.asarray([True, False])
            return np.array(
                values,
                dtype=dtype,
                copy=True if copy is None else copy,
            )

    source = BooleanArrayLike()
    np.testing.assert_array_equal(xp.asarray(source), [True, False])
    np.testing.assert_array_equal(xp.asarray(source, dtype=xp.bool), [True, False])
    with pytest.raises(TypeError, match="cross-kind"):
        xp.asarray(source, dtype=xp.float64)

    class RealArrayLike:
        def __array__(
            self, dtype: object = None, copy: builtins.bool | None = None
        ) -> np.ndarray:
            values = np.asarray([0.25, 0.5])
            return np.array(
                values,
                dtype=dtype,
                copy=True if copy is None else copy,
            )

    real_source = RealArrayLike()
    np.testing.assert_array_equal(
        xp.asarray(real_source, dtype=xp.float64), [0.25, 0.5]
    )
    with pytest.raises(TypeError, match="cross-kind"):
        xp.asarray(real_source, dtype=xp.bool)
    with pytest.raises(TypeError, match="nested host"):
        xp.asarray([source], dtype=xp.bool)

    class ObjectArrayLike:
        def __init__(self, values: list[object]) -> None:
            self.values = values
            self.calls = 0

        def __array__(
            self, dtype: object = None, copy: builtins.bool | None = None
        ) -> np.ndarray:
            assert dtype is None
            assert copy is None
            self.calls += 1
            return np.asarray(self.values, dtype=object)

    object_bool = ObjectArrayLike([True, False])
    np.testing.assert_array_equal(xp.asarray(object_bool, dtype=xp.bool), [True, False])
    assert object_bool.calls == 1
    object_bool_to_real = ObjectArrayLike([True, False])
    with pytest.raises(TypeError, match="cross-kind"):
        xp.asarray(object_bool_to_real, dtype=xp.float64)
    assert object_bool_to_real.calls == 1
    object_mixed = ObjectArrayLike([True, 0.5])
    with pytest.raises(TypeError, match="mixed bool/real"):
        xp.asarray(object_mixed, dtype=xp.float64)
    assert object_mixed.calls == 1


def test_tensor_facade_exports_comparison_factory() -> None:
    x = ir.input_tensor("x", TensorSpec(dtype="float64", role="input"))
    predicate = tensor_compare("equal", x, x)
    assert predicate.op == "equal"
    assert predicate.spec.dtype == "bool"


@pytest.mark.parametrize(
    "operation,oracle",
    (
        (lambda x: xp.reshape(x, (6,)), lambda x: np.reshape(x, (6,))),
        (
            lambda x: xp.broadcast_to(x, (2, 4, 3)),
            lambda x: np.broadcast_to(x, (2, 4, 3)),
        ),
        (
            lambda x: xp.broadcast_arrays(x, xp.ones((1, 4, 1)))[0],
            lambda x: np.broadcast_arrays(x, np.ones((1, 4, 1)))[0],
        ),
        (lambda x: xp.expand_dims(x, (0, -1)), lambda x: np.expand_dims(x, (0, -1))),
        (lambda x: xp.squeeze(x, 1), lambda x: np.squeeze(x, 1)),
        (lambda x: xp.moveaxis(x, 0, 2), lambda x: np.moveaxis(x, 0, 2)),
        (lambda x: xp.flip(x, axis=(0, 2)), lambda x: np.flip(x, axis=(0, 2))),
        (lambda x: xp.slice(x, ((0, 1), (0, 1), (1, 3))), lambda x: x[:1, :1, 1:3]),
        (lambda x: xp.take(x, (2, 0), axis=2), lambda x: np.take(x, (2, 0), axis=2)),
        (
            lambda x: xp.permute_dims(x, (2, 1, 0)),
            lambda x: np.transpose(x, (2, 1, 0)),
        ),
        (xp.matrix_transpose, lambda x: np.swapaxes(x, -1, -2)),
    ),
)
def test_eager_and_captured_boolean_views_match_numpy(
    operation: typing.Callable[[typing.Any], typing.Any],
    oracle: typing.Callable[[np.ndarray], np.ndarray],
) -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 1, 3)
    mask = xp.greater(values, 2)[:, :, ::-1]
    original = mask.copy()
    expected = oracle(mask)
    eager = operation(mask)
    captured = xp.compile(operation)(mask)
    assert eager.dtype == captured.dtype == np.dtype("bool")
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(captured, expected)
    np.testing.assert_array_equal(mask, original)
    assert not np.shares_memory(captured, mask)


@pytest.mark.parametrize(
    "operation",
    (
        lambda x: xp.add(x, x),
        lambda x: xp.subtract(x, x),
        lambda x: xp.multiply(x, x),
        lambda x: xp.divide(x, x),
        xp.negative,
        xp.exp,
        xp.log,
        xp.sqrt,
        lambda x: xp.pow(x, 2),
        xp.square,
        xp.reciprocal,
        xp.sum,
        xp.mean,
        lambda x: xp.matmul(x, x),
        lambda x: xp.einsum("ij->ji", x),
    ),
)
def test_boolean_views_do_not_admit_floating_arithmetic(
    operation: typing.Callable[[typing.Any], typing.Any],
) -> None:
    mask = np.array([[True, False], [False, True]])
    with pytest.raises(TypeError, match="floating arithmetic"):
        operation(mask)
    with pytest.raises((TypeError, ValueError), match="bool|floating arithmetic"):
        xp.compile(lambda x: operation(x))(mask)


def test_compiler_namespace_bool_creation_uses_boolean_literals() -> None:
    seed = compiler_namespace.full((2,), True, dtype="bool")
    arrays = {
        "zeros": compiler_namespace.zeros((2,), dtype="bool"),
        "ones": compiler_namespace.ones((2,), dtype="bool"),
        "zeros_like": compiler_namespace.zeros_like(seed),
        "ones_like": compiler_namespace.ones_like(seed),
    }
    assert all(value.dtype == "bool" for value in arrays.values())
    outputs = execute(Program({name: value.node for name, value in arrays.items()}), {})
    np.testing.assert_array_equal(outputs.outputs["zeros"], [False, False])
    np.testing.assert_array_equal(outputs.outputs["ones"], [True, True])
    np.testing.assert_array_equal(outputs.outputs["zeros_like"], [False, False])
    np.testing.assert_array_equal(outputs.outputs["ones_like"], [True, True])


def test_public_bool_dtype_does_not_rebind_python_annotations() -> None:
    assert typing.get_type_hints(xp.asarray)["copy"] == builtins.bool | None
    assert typing.get_type_hints(xp.sum)["keepdims"] is builtins.bool
    assert typing.get_type_hints(xp.can_cast)["return"] is builtins.bool


def test_optimizer_cse_and_backend_gates() -> None:
    spec = TensorSpec(dtype="float64", role="input")
    x = ir.input_tensor("x", spec)
    program = Program({"a": ir.compare("equal", x, x), "b": ir.compare("equal", x, x)})
    optimized = optimize(program)
    assert optimized.outputs["a"] is optimized.outputs["b"]
    feeds = {"x": np.asarray(1.0, dtype=np.float64)}
    assert (
        execute(program, feeds).outputs["a"] == execute(optimized, feeds).outputs["a"]
    )
    for backend in ("cpu", "cuda", "portable", "scalar"):
        with pytest.raises(
            ValueError, match="does not support bool data or comparisons"
        ):
            prepare_for_backend(program, backend)
    with pytest.raises(ValueError, match="does not support bool data or comparisons"):
        emit_cpu(program)
    with pytest.raises(ValueError, match="does not support bool data or comparisons"):
        plan_cuda(program, cuda_target_info("sm_80"))

    mixed = Program({"real": x, "predicate": ir.compare("equal", x, x)})
    for backend in ("cpu", "cuda", "portable", "scalar"):
        prepared = prepare_for_backend(mixed, backend, requested_outputs=("real",))
        assert tuple(prepared.outputs) == ("real",)
        assert all(node.spec.dtype != "bool" for node in prepared.live_nodes)


def test_bool_nodes_are_outside_floating_precision_schedules() -> None:
    spec = TensorSpec(dtype="float64", role="input")
    x = ir.input_tensor("x", spec)
    predicate = ir.compare("equal", x, x)
    program = Program({"real": x, "predicate": predicate})
    predicate_name = program.debug_names[predicate]
    schedule = describe_precision(program)
    assert predicate_name not in {value.name for value in schedule.values}
    assert all(
        value.storage_dtype in ("float32", "float64") for value in schedule.values
    )
    with pytest.raises(ValueError, match="non-floating"):
        lower_precision(
            program,
            {predicate_name: PrecisionDirective("float64", "float64", "float64")},
        )
    assert lower_precision(program, {}).logical_hash == program.logical_hash


def test_real_precision_rewrite_restores_comparison_operand_dtypes() -> None:
    real = ir.input_tensor("real", TensorSpec(role="input"))
    squared = ir.multiply(real, real)
    predicate = ir.compare("greater", squared, real)
    viewed = ir.reshape(predicate, (Index("i", IndexSpace("unit", "matrix", 1)),))
    program = Program({"real": squared, "mask": viewed})
    lowered = lower_precision(
        program,
        {
            program.debug_names[squared]: PrecisionDirective(
                "float32", "float32", "float32"
            )
        },
    )
    source = np.asarray(1.00000001, dtype=np.float64)
    expected_real = np.asarray(source.astype(np.float32) ** 2, dtype=np.float64)
    expected_mask = np.greater(expected_real, source).reshape(1)
    actual = execute(lowered, {"real": source}).outputs
    np.testing.assert_array_equal(actual["real"], expected_real)
    np.testing.assert_array_equal(actual["mask"], expected_mask)
    assert actual["mask"].dtype == np.dtype("bool")
    assert all(
        node.spec.dtype in ("float32", "float64")
        for node in lowered.live_nodes
        if node.op == "cast"
    )
    assert describe_precision(lowered) == describe_precision(
        Program.loads(lowered.dumps())
    )
    with pytest.raises(ValueError, match="does not support bool"):
        prepare_for_backend(lowered, "cpu")


def test_scientific_domain_and_boolean_algebra_fail_closed() -> None:
    ao = Index("i", IndexSpace("ao", "ao", 2))
    occ = Index("i", IndexSpace("occ", "occupied", 2))
    left = xp.input_array("left", TensorSpec((ao,), role="input"))
    right = xp.input_array("right", TensorSpec((occ,), role="input"))
    with pytest.raises(ValueError, match="index domains"):
        xp.equal(left, right)
    represented = xp.input_array(
        "represented", TensorSpec((ao,), representation="spin_orbital", role="input")
    )
    with pytest.raises(ValueError, match="representation"):
        xp.equal(left, represented)
    bool_spec = TensorSpec((ao,), dtype="bool", role="input")
    value = ir.input_tensor("mask", bool_spec)
    assert ir.slice_tensor(value, ((0, 1),)).spec.dtype == "bool"
    for operation in (
        lambda: ir.add(value, value),
        lambda: ir.multiply(value, value),
        lambda: ir.reduce_sum(value, (0,)),
        lambda: ir.einsum("i->i", value),
        lambda: ir.cast(value, "float64"),
    ):
        with pytest.raises(ValueError, match="bool"):
            operation()
    with pytest.raises(ValueError, match="bool"):
        TensorSpec(dtype="bool", role="parameter", differentiable=True)
    with pytest.raises(ValueError, match="bool constants"):
        ir.constant(1, TensorSpec(dtype="bool", role="constant"))
    with pytest.raises(ValueError, match="dtype bool"):
        execute(Program({"mask": value}), {"mask": np.ones(2, dtype=np.int64)})


def test_boolean_ad_and_nonfinite_inputs_reject_explicitly() -> None:
    array = np.array([1.0, 2.0], dtype=np.float64)
    program = xp.compile(lambda x: x > 0).lower(array)
    for operation in (
        lambda: jvp(program, {"x": array}, {"x": np.ones_like(array)}),
        lambda: vjp(program, {"x": array}, {"output": np.ones(2, dtype=np.bool_)}),
        lambda: dot_test(
            program,
            {"x": array},
            {"x": np.ones_like(array)},
            {"output": np.ones(2, dtype=np.bool_)},
        ),
        lambda: linearize(program, ("x",)),
        lambda: transpose_program(program, ("output",)),
    ):
        with pytest.raises(ValueError, match="non-differentiable"):
            operation()
    bad = np.asarray([np.nan, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        xp.greater(bad, array)
    with pytest.raises(ValueError, match="non-finite"):
        xp.compile(lambda x, y: x > y)(bad, array)


@pytest.mark.parametrize("name", tuple(name for name, _ in COMPARISONS))
def test_array_api_strict_finite_reference(name: str) -> None:
    left = np.asarray([[1.0], [-2.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0]], dtype=np.float64)
    operation = getattr(xp, name)
    actual = xp.compile(lambda x, y: operation(x, y))(left, right)
    oracle = getattr(strict, name)(strict.asarray(left), strict.asarray(right))
    np.testing.assert_array_equal(actual, np.asarray(oracle))

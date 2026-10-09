"""Finite real comparisons and first-class Boolean TensorIR data."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import ir
from generativeqc_compiler.tensor.ad_program import linearize, transpose_program
from generativeqc_compiler.tensor.autodiff import jvp, vjp
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.optimize import optimize, prepare_for_backend
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


def test_array_api_strict_finite_reference_if_installed() -> None:
    strict = pytest.importorskip("array_api_strict")
    left = np.asarray([[1.0], [-2.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0]], dtype=np.float64)
    actual = xp.compile(lambda x, y: xp.less_equal(x, y))(left, right)
    oracle = strict.less_equal(strict.asarray(left), strict.asarray(right))
    np.testing.assert_array_equal(actual, np.asarray(oracle))

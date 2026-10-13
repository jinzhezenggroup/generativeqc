"""Independent finite-real selection oracles and captured endpoint boundaries."""

import typing
from dataclasses import replace
from pathlib import Path

import array_api_strict as strict
import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import input_array
from generativeqc_compiler.array_api import namespace as ns
from generativeqc_compiler.tensor import ir
from generativeqc_compiler.tensor.ad_program import linearize, transpose_program
from generativeqc_compiler.tensor.autodiff import jvp, vjp
from generativeqc_compiler.tensor.complexity import node_complexity
from generativeqc_compiler.tensor.cpu import describe_cpu_compilation
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.optimize import optimize, prepare_for_backend
from generativeqc_compiler.tensor.precision import (
    PrecisionDirective,
    conservative_precision_variants,
    lower_precision,
)
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, Symmetry, TensorSpec


@pytest.mark.parametrize(
    "d1,d2",
    [
        ("float32", "float32"),
        ("float64", "float64"),
        ("float32", "float64"),
        ("float64", "float32"),
    ],
)
@pytest.mark.parametrize(
    "shapes",
    [
        ((), (), ()),
        ((2, 1, 3), (1, 4, 1), (2, 1, 3)),
        ((0, 1), (1, 3), ()),
        ((3,), (3,), (3,)),
    ],
)
def test_complete_capture_matches_two_independent_oracles(
    d1: str, d2: str, shapes: tuple[tuple[int, ...], ...]
) -> None:
    condition = np.arange(np.prod(shapes[0], dtype=int)).reshape(shapes[0]) % 2 == 0
    left = np.arange(np.prod(shapes[1], dtype=int), dtype=d1).reshape(shapes[1])
    right = -np.arange(np.prod(shapes[2], dtype=int), dtype=d2).reshape(shapes[2])
    before = [value.copy() for value in (condition, left, right)]
    expected = np.where(condition, left, right)
    independent = np.asarray(
        strict.where(
            strict.asarray(condition), strict.asarray(left), strict.asarray(right)
        )
    )
    compiled = xp.compile(lambda c, a, b: xp.where(c, a, b), backend="reference")
    program = compiled.lower(condition, left, right)
    restored = Program.loads(program.dumps())
    assert restored.logical_hash == program.logical_hash
    assert restored.dumps() == program.dumps()
    assert program.outputs["output"].spec.dtype == expected.dtype.name
    assert program.outputs["output"].spec.shape == expected.shape
    for result in (
        compiled(condition, left, right),
        compiled(condition, left, right),
        execute(restored, {"c": condition, "a": left, "b": right}).outputs["output"],
    ):
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result, independent)
        for source in (condition, left, right):
            assert not np.shares_memory(result, source)
    for source, original in zip((condition, left, right), before, strict=True):
        np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_noncontiguous_extreme_finite_and_signed_zero(dtype: str) -> None:
    limits = np.finfo(dtype)
    left = np.array([-0.0, limits.max, limits.tiny, 1.0], dtype=dtype)[::-1]
    right = np.array([0.0, -limits.max, -limits.tiny, -1.0], dtype=dtype)[::-1]
    condition = np.array([True, False, True, False])[::-1]
    result = xp.compile(lambda c, a, b: xp.where(c, a, b), backend="reference")(
        condition, left, right
    )
    expected = np.where(condition, left, right)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    result[:] = 7
    np.testing.assert_array_equal(
        left, np.array([-0.0, limits.max, limits.tiny, 1.0], dtype=dtype)[::-1]
    )


@pytest.mark.parametrize(
    "condition",
    [
        lambda x: x > 0,
        lambda x: xp.full(x.shape, True, dtype=xp.bool),
        lambda x: True,
        lambda x: xp.zeros((), dtype=xp.bool),
    ],
)
def test_condition_sources_and_python_branch_scalars(
    condition: typing.Callable,
) -> None:
    values = np.array([-1.0, 2.0], dtype=np.float32)
    compiled = xp.compile(lambda x: xp.where(condition(x), x, 0.5), backend="reference")
    expected = np.where(condition(values), values, np.float32(0.5))
    result = compiled(values)
    assert result.dtype == np.dtype("float32")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "bad", [np.array([1, 0], dtype=np.int64), np.array([1.0, 0.0])]
)
def test_numeric_condition_rejected(bad: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="bool|float32|float64"):
        xp.compile(lambda c, x: xp.where(c, x, x), backend="reference")(bad, np.ones(2))


def test_cross_kind_shape_and_scientific_admission() -> None:
    mask = ns.full((2,), True, dtype="bool")
    real = ns.full((2,), 1)
    with pytest.raises(TypeError, match="real float"):
        ns.where(mask, mask, real)
    with pytest.raises((TypeError, ValueError)):
        ns.where(mask, real, False)
    with pytest.raises(ValueError, match="broadcast"):
        ns.where(mask, real, ns.ones((3,)))
    for kind in ("ao", "occupied", "virtual", "spin"):
        spec = TensorSpec((Index("i", IndexSpace(kind, kind, 2)),), role="input")
        value = input_array("x", spec)
        with pytest.raises((TypeError, ValueError), match="generic|scientific"):
            ns.where(mask, value, value)
        with pytest.raises(ValueError, match="generic|scientific"):
            ir.where(mask.node, value.node, value.node)
    for spec in (
        replace(real.node.spec, representation="spin_orbital"),
        replace(ns.ones((2, 2)).node.spec, symmetries=(Symmetry((1, 0)),)),
    ):
        value = ir.input_tensor("x", replace(spec, role="input"))
        predicate = ns.full(spec.shape, True, dtype="bool").node
        with pytest.raises(ValueError, match="scientific"):
            ir.where(predicate, value, value)
        with pytest.raises(TypeError, match="scientific"):
            ns.where(True, input_array("x", value.spec), 0)
    spin_spec = TensorSpec(
        (Index("i", IndexSpace("_array_extent_2", "matrix", 2, "alpha")),),
        role="input",
    )
    spin_value = input_array("spin", spin_spec)
    with pytest.raises(TypeError, match="scientific"):
        ns.where(True, spin_value, 0)
    with pytest.raises(ValueError, match="scientific"):
        ir.where(mask.node, spin_value.node, spin_value.node)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_unselected_nonfinite_branch_remains_live_after_optimization(
    bad: float,
) -> None:
    compiled = xp.compile(lambda a, b: xp.where(True, a, b), backend="reference")
    program = optimize(compiled.lower(np.ones(2), np.ones(2)))
    with pytest.raises(ValueError, match="non-finite"):
        execute(program, {"a": np.ones(2), "b": np.full(2, bad)})


@pytest.mark.parametrize(
    "bad_branch", [lambda x: xp.log(x), lambda x: xp.divide(x, 0), lambda x: xp.exp(x)]
)
def test_unselected_invalid_arithmetic_is_not_short_circuited(
    bad_branch: typing.Callable,
) -> None:
    values = np.array([-1.0, 1000.0])
    compiled = xp.compile(
        lambda x: xp.where(True, x, bad_branch(x)), backend="reference"
    )
    with pytest.raises(ValueError, match="domain|zero|non-finite"):
        compiled(values)


def test_where_identity_cse_and_bounded_retention() -> None:
    compiled = xp.compile(
        lambda c, a, b: {"first": xp.where(c, a, b), "second": xp.where(c, a, b)},
        backend="reference",
    )
    c, a, b = np.array([True, False]), np.ones(2), np.zeros(2)
    program = compiled.lower(c, a, b)
    prepared = optimize(program)
    assert prepared.outputs["first"] is prepared.outputs["second"]
    opposite = Program(
        {
            "first": ir.where(
                prepared.outputs["first"].inputs[0],
                prepared.outputs["first"].inputs[2],
                prepared.outputs["first"].inputs[1],
            )
        }
    )
    assert (
        opposite.logical_hash
        != Program({"first": prepared.outputs["first"]}).logical_hash
    )
    result = execute(prepared, {"c": c, "a": a, "b": b})
    assert node_complexity(prepared.outputs["first"]).concrete_work == 2
    with pytest.raises(ValueError, match="retained-byte budget"):
        execute(
            prepared,
            {"c": c, "a": a, "b": b},
            max_bytes=result.logical_retained_bytes - 1,
        )
    execute(prepared, {"c": c, "a": a, "b": b}, max_bytes=result.logical_retained_bytes)
    with pytest.raises(ValueError, match="overflow|size|64-bit"):
        ns.where(True, ns.ones((2**62, 8)), 1)


def test_serialized_where_cannot_bypass_operand_validation() -> None:
    c = ns.full((2,), True, dtype="bool").node
    a = ns.ones((2,)).node
    b = ns.ones((3,)).node
    with pytest.raises(ValueError, match="domains"):
        ir.Node("where", (c, a, b), a.spec)
    with pytest.raises(ValueError, match="bool"):
        ir.Node("where", (a, a, a), a.spec)
    with pytest.raises(ValueError, match="three"):
        ir.Node("where", (c, a), a.spec)


def test_ad_refuses_where_but_unrelated_diagnostic_can_be_pruned() -> None:
    values = np.ones(2)
    compiled = xp.compile(
        lambda x: {"selected": xp.where(x > 0, x, 0), "real": xp.sum(x * x)},
        backend="reference",
        differentiable=("x",),
    )
    program = compiled.lower(values)
    for action in (
        lambda: linearize(program, ["x"]),
        lambda: transpose_program(program, ["selected"]),
        lambda: jvp(program, {"x": values}, {"x": values}),
        lambda: vjp(
            program, {"x": values}, {"selected": values, "real": np.array(1.0)}
        ),
    ):
        with pytest.raises(ValueError, match="Boolean|non-differentiable|AD"):
            action()
    derivative = linearize(program, ["x"], outputs=["real"]).program
    assert execute(derivative, {"x": values, "d_x": values}).outputs["d_real"] == 4


@pytest.mark.parametrize("backend", ["cpu", "cuda", "portable", "scalar"])
def test_native_preparation_refuses_where(backend: str) -> None:
    program = xp.compile(lambda x: xp.where(x > 0, x, 0), backend="reference").lower(
        np.ones(2)
    )
    with pytest.raises(ValueError, match="where"):
        prepare_for_backend(program, backend, allow_cpu_boolean=backend == "cpu")
    if backend == "cpu":
        with pytest.raises(ValueError, match="where"):
            describe_cpu_compilation(program)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_public_native_endpoint_refuses_before_toolchain_or_device(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("unsupported where reached toolchain/device/allocation")

    import generativeqc_compiler.common.compiler_cache as cache
    from generativeqc import profiles
    from generativeqc_compiler.common import cpp_adapter
    from generativeqc_compiler.tensor import cpu, cuda_execute

    monkeypatch.setattr(cpu, "CppCompilerAdapter", forbidden)
    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)
    monkeypatch.setattr(cache, "resolve_compiler_cache", forbidden)
    monkeypatch.setattr(profiles, "find_nvcc", forbidden)
    monkeypatch.setattr(cuda_execute, "PreparedCuda", forbidden)
    compiled = xp.compile(lambda x: xp.where(x > 0, x, 0), backend=backend)
    with pytest.raises(ValueError, match="where"):
        compiled(np.ones(2))


@pytest.mark.parametrize("backend", ["reference", "cpu", "cuda"])
def test_public_derivative_endpoints_refuse_selection(backend: str) -> None:
    def energy(x: typing.Any) -> typing.Any:
        return xp.sum(xp.where(x > 0, x, 0))

    values = np.ones(2)
    for action in (
        lambda: xp.grad(energy, backend=backend)(values),
        lambda: xp.value_and_grad(energy, backend=backend)(values),
        lambda: xp.jvp(energy, (values,), (values,), backend=backend),
        lambda: xp.vjp(energy, values, backend=backend),
    ):
        with pytest.raises(ValueError, match="Boolean|non-differentiable"):
            action()


def test_native_bundle_rejects_before_cache_allocation(tmp_path: Path) -> None:
    from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
    from generativeqc_compiler.tensor.cpu_bundle import NativeTensorProgramBundle

    # No adapter discovery/initialization is needed to test IR admission.
    adapter = object.__new__(CppCompilerAdapter)
    program = xp.compile(lambda x: xp.where(True, x, 0), backend="reference").lower(
        np.ones(2)
    )
    cache = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match="bool|where"):
        NativeTensorProgramBundle((program,), compiler=adapter, cache=cache)
    assert not cache.exists()


def test_direct_public_where_capture_and_two_scalar_branches() -> None:
    mask = np.array([True, False])
    result = xp.compile(xp.where, backend="reference")(mask, np.ones(2), np.zeros(2))
    np.testing.assert_array_equal(result, [1, 0])
    result = xp.compile(lambda c: xp.where(c, 1.5, 2.5), backend="reference")(mask)
    np.testing.assert_array_equal(result, [1.5, 2.5])
    assert result.dtype == np.dtype("float64")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("comparison_condition", [False, True])
def test_precision_identity_preserves_boolean_condition(
    dtype: str, comparison_condition: bool
) -> None:
    mask = np.array([True, False])
    values = np.array([-0.0, 1.00000006], dtype=dtype)
    compiled = xp.compile(
        lambda c, x: xp.where(x > 0 if comparison_condition else c, x, 2 * x),
        backend="reference",
    )
    program = compiled.lower(mask, values)
    lowered = lower_precision(program, {})
    assert lowered.logical_hash == program.logical_hash
    expected = np.where(
        values > 0 if comparison_condition else mask, values, 2 * values
    )
    result = execute(lowered, {"c": mask, "x": values}).outputs["output"]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    assert (
        next(node for node in lowered.live_nodes if node.op == "where")
        .inputs[0]
        .spec.dtype
        == "bool"
    )


def test_precision_casts_only_real_selection_branches() -> None:
    mask = np.array([True, False])
    values = np.array([1.00000006, -0.0])
    program = xp.compile(lambda c, x: xp.where(c, x, 2 * x), backend="reference").lower(
        mask, values
    )
    selected = program.outputs["output"]
    lowered = lower_precision(
        program,
        {
            program.debug_names[selected]: PrecisionDirective(
                "float32", "float32", "float32"
            )
        },
    )
    result = execute(lowered, {"c": mask, "x": values}).outputs["output"]
    expected = np.where(
        mask, values.astype(np.float32), (2 * values).astype(np.float32)
    ).astype(np.float64)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    predicate = next(node for node in lowered.live_nodes if node.op == "where").inputs[
        0
    ]
    assert predicate.spec.dtype == "bool"
    variants = conservative_precision_variants(program)
    assert len(variants) == 2
    for variant in variants:
        actual = execute(variant, {"c": mask, "x": values}).outputs["output"]
        oracle = np.where(
            mask,
            values,
            2 * values if variant is program else 2 * values.astype(np.float32),
        )
        np.testing.assert_array_equal(actual, oracle)
        with pytest.raises(ValueError, match="Boolean|non-differentiable"):
            linearize(variant, [])
        with pytest.raises(ValueError, match="where"):
            prepare_for_backend(variant, "cpu", allow_cpu_boolean=True)

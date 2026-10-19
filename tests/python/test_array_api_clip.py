"""Finite-real clip capture, independent oracles and explicit domain boundaries."""

from dataclasses import replace
from fractions import Fraction
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import VibeArray, input_array
from generativeqc_compiler.array_api import namespace as ns
from generativeqc_compiler.tensor import autodiff, ir
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.optimize import optimize
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, Symmetry, TensorSpec


def symbolic(name: str, shape: tuple, dtype: str = "float64") -> VibeArray:
    return input_array(
        name, TensorSpec(ns._generic_indices(shape), dtype=dtype, role="input")
    )


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("shape", ((), (0, 3), (2, 1, 3)))
@pytest.mark.parametrize(
    "bounds", ((None, None), (-1, None), (None, 1), (-1, 1), (1, 1), (-3, -1), (1, 3))
)
def test_scalar_bounds_two_oracles_and_serialization(
    dtype: str, shape: tuple, bounds: tuple
) -> None:
    x = (np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape) - 2) / 2
    lo, hi = bounds
    expected = np.asarray(strict.clip(strict.asarray(x), lo, hi))
    numpy = x.copy() if lo is hi is None else np.clip(x, lo, hi)
    np.testing.assert_array_equal(expected, numpy)
    source = xp.compile(lambda x: xp.clip(x, min=lo, max=hi), backend="reference")
    try:
        program = source.lower(x)
        assert source.lower(x).dumps() == program.dumps()
        restored = Program.loads(program.dumps())
        assert restored.dumps() == program.dumps()
        assert restored.logical_hash == program.logical_hash
        before = x.copy()
        for result in (
            xp.clip(xp.asarray(x), lo, hi),
            source(x),
            source(x),
            execute(restored, {"x": x}).outputs["output"],
            execute(optimize(program), {"x": x}).outputs["output"],
        ):
            assert result.dtype == x.dtype and result.shape == x.shape
            np.testing.assert_array_equal(result, expected)
            assert not np.shares_memory(result, x)
        np.testing.assert_array_equal(x, before)
    finally:
        source.close()


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("shape", ((2, 1, 3), (0, 1, 3)))
def test_multi_axis_array_bounds(dtype: str, shape: tuple) -> None:
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape) - 2
    lo = np.array([[[-1], [0], [1], [2]]], dtype=dtype)
    hi = np.array([3, 4, 5], dtype=dtype)
    wanted = np.clip(x, lo, hi)
    independent = np.asarray(strict.clip(*(strict.asarray(v) for v in (x, lo, hi))))
    source = xp.compile(lambda x, lo, hi: xp.clip(x, lo, hi), backend="reference")
    try:
        for result in (
            source(x, lo, hi),
            xp.clip(*(xp.asarray(v) for v in (x, lo, hi))),
        ):
            assert result.shape == (shape[0], 4, 3) and result.dtype == x.dtype
            np.testing.assert_array_equal(result, wanted)
            np.testing.assert_array_equal(result, independent)
    finally:
        source.close()


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_exact_fraction_boundary_and_typed_rounding(dtype: str) -> None:
    x = np.array([-2, -0.5, 0, 0.5, 2], dtype=dtype)
    lo, hi = Fraction(-1, 2), Fraction(1, 2)
    manual = np.array(
        [float(min(max(Fraction(float(v)), lo), hi)) for v in x], dtype=dtype
    )
    source = xp.compile(lambda x: xp.clip(x, lo, hi), backend="reference")
    rounded = xp.compile(lambda x: xp.clip(x, 0.1, 0.3), backend="reference")
    try:
        np.testing.assert_array_equal(source(x), manual)
        np.testing.assert_array_equal(
            rounded(x), np.asarray(strict.clip(strict.asarray(x), 0.1, 0.3))
        )
    finally:
        source.close()
        rounded.close()


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("side", ("min", "max"))
def test_one_sided_array_bound(dtype: str, side: str) -> None:
    x = np.array([[-2.0], [2.0]], dtype=dtype)
    bound = np.array([-1.0, 0.0, 1.0], dtype=dtype)
    source = xp.compile(
        lambda x, bound: xp.clip(x, **{side: bound}), backend="reference"
    )
    try:
        result = source(x, bound)
        expected = strict.clip(strict.asarray(x), **{side: strict.asarray(bound)})
        np.testing.assert_array_equal(result, np.asarray(expected))
    finally:
        source.close()


def test_reversed_bounds_choose_upper_without_standard_conformance_claim() -> None:
    x, lo, hi = (
        np.array([-4.0, 0.0, 4.0]),
        np.array([2.0, -1.0, 3.0]),
        np.array([1.0, 1.0, 0.0]),
    )
    source = xp.compile(lambda x, lo, hi: xp.clip(x, lo, hi), backend="reference")
    try:
        np.testing.assert_array_equal(source(x, lo, hi), [1, 0, 0])
        np.testing.assert_array_equal(source(x, lo, hi), np.clip(x, lo, hi))
    finally:
        source.close()


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_array_signed_zero_ties_and_extreme_finite(dtype: str) -> None:
    limit = np.finfo(dtype)
    x = np.array([-limit.max, -0.0, 0.0, limit.max], dtype=dtype)[::-1]
    lo, hi = np.array(-0.0, dtype=dtype), np.array(0.0, dtype=dtype)
    source = xp.compile(lambda x, lo, hi: xp.clip(x, lo, hi), backend="reference")
    try:
        result, expected = source(x, lo, hi), np.clip(x, lo, hi)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    finally:
        source.close()


@pytest.mark.parametrize("bound", (True, 1j, object(), [0], np.array(0.0)))
def test_invalid_static_bounds(bound: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ns.clip(symbolic("x", (2,)), bound)


@pytest.mark.parametrize("bound", (float("nan"), float("inf"), -float("inf"), -0.0))
def test_unsupported_scalar_domains(bound: float) -> None:
    with pytest.raises(ValueError):
        ns.clip(symbolic("x", (0,)), bound)


def test_dtype_shape_and_scientific_rejections() -> None:
    x = symbolic("x", (2, 3))
    with pytest.raises(TypeError, match="same dtype"):
        ns.clip(x, symbolic("lo", (), dtype="float32"))
    with pytest.raises(TypeError, match="real float"):
        ns.clip(symbolic("mask", (2,), dtype="bool"))
    with pytest.raises(ValueError, match="broadcast"):
        ns.clip(x, symbolic("lo", (4,)), symbolic("hi", (2, 1)))
    annotated = VibeArray(
        ir.input_tensor(
            "ao", TensorSpec((Index("a", IndexSpace("ao", "ao", 3)),), role="input")
        )
    )
    for bounds in ((None, None), (0, None), (None, x)):
        with pytest.raises(TypeError, match="scientific annotations"):
            ns.clip(annotated, *bounds)
    for spec in (
        replace(x.node.spec, representation="spin_orbital"),
        replace(symbolic("square", (2, 2)).node.spec, symmetries=(Symmetry((1, 0)),)),
        TensorSpec(
            (Index("i", IndexSpace("_array_extent_2", "matrix", 2, "alpha")),),
            role="input",
        ),
    ):
        with pytest.raises(TypeError, match="scientific annotations"):
            ns.clip(VibeArray(ir.input_tensor("annotated", spec)))
    with pytest.raises(TypeError, match="scientific annotations"):
        ns.clip(x, annotated)


@pytest.mark.parametrize("dtype,bound", (("float32", 1e100), ("float64", 10**1000)))
def test_scalar_overflow_rejected_even_for_empty_capture(
    dtype: str, bound: object
) -> None:
    with pytest.raises(ValueError, match="finite"):
        ns.clip(symbolic("x", (0,), dtype), max=bound)


def test_masked_invalid_computation_and_resource_budgets() -> None:
    source = xp.compile(lambda x: xp.clip(xp.log(x), 0, 0), backend="reference")
    try:
        with pytest.raises(ValueError, match="domain|positive|finite"):
            source(np.array([-1.0, 1.0]))
    finally:
        source.close()
    for backend, limits in (
        ("reference", {"max_bytes": 1}),
        ("cpu", {"max_work": 0}),
        ("cpu", {"max_nodes": 1}),
    ):
        source = xp.compile(lambda x: xp.clip(x, 0, 1), backend=backend, **limits)
        try:
            with pytest.raises(ValueError, match="budget"):
                source(np.ones(3))
        finally:
            source.close()


@pytest.mark.parametrize("position", (0, 1, 2))
@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_all_runtime_operands_checked_even_when_masked(
    position: int, bad: float
) -> None:
    source = xp.compile(lambda x, lo, hi: xp.clip(x, lo, hi), backend="reference")
    args = [np.array([-4.0, 4.0]), np.array(0.0), np.array(1.0)]
    args[position] = np.full_like(args[position], bad)
    try:
        with pytest.raises(ValueError, match="finite"):
            source(*args)
    finally:
        source.close()


def test_eager_standard_nan_integer_complex_and_discovery_preserved() -> None:
    for x, lo, hi in (
        ([np.nan, -2.0, 3.0], 0.0, 1.0),
        ([1.0, 2.0], np.nan, 4.0),
        ([1.0, 2.0], 0.0, np.nan),
    ):
        np.testing.assert_array_equal(
            np.asarray(xp.clip(xp.asarray(x), lo, hi)), np.clip(x, lo, hi)
        )
    np.testing.assert_array_equal(xp.clip(xp.asarray([-2, 0, 2]), -1, 1), [-1, 0, 1])
    assert xp.__array_api_version__ == "2025.12"
    with pytest.raises((TypeError, ValueError)):
        xp.compile(lambda x: xp.clip(x), backend="reference")(np.array([1j]))


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_reference_interior_jvp_vjp_dot_and_finite_difference(dtype: str) -> None:
    feeds = {
        "x": np.array([[-2.0], [0.0], [2.0]], dtype=dtype),
        "lo": np.array([-1.0, -0.5], dtype=dtype),
        "hi": np.array(1.0, dtype=dtype),
    }
    directions = {
        key: np.full_like(value, v)
        for (key, value), v in zip(feeds.items(), (0.2, -0.3, 0.4), strict=True)
    }
    source = xp.compile(
        lambda x, lo, hi: xp.clip(x, lo, hi),
        backend="reference",
        differentiable=tuple(feeds),
    )
    try:
        program = source.lower(*feeds.values())
        weights = np.arange(1, 7, dtype=dtype).reshape(3, 2) / 8
        expected = np.array([[-0.3, -0.3], [0.2, 0.2], [0.4, 0.4]], dtype=dtype)
        forward = autodiff.jvp(program, feeds, directions, allow_comparison_where=True)
        reverse = autodiff.vjp(
            program, feeds, {"output": weights}, allow_comparison_where=True
        )
        tol = 2e-6 if dtype == "float32" else 2e-13
        np.testing.assert_allclose(
            forward.output_tangents["output"], expected, atol=tol, rtol=tol
        )
        for key, wanted in (
            ("x", [[0.0], [weights[1].sum()], [0.0]]),
            ("lo", weights[0]),
            ("hi", weights[2].sum()),
        ):
            np.testing.assert_allclose(
                reverse.input_cotangents[key], wanted, atol=tol, rtol=tol
            )
        assert autodiff.dot_test(
            program,
            feeds,
            directions,
            {"output": weights},
            rtol=tol,
            atol=tol,
            allow_comparison_where=True,
        ).passed
        step = 2e-3 if dtype == "float32" else 1e-5
        plus = [v + step * directions[k] for k, v in feeds.items()]
        minus = [v - step * directions[k] for k, v in feeds.items()]
        np.testing.assert_allclose(
            (np.clip(*plus) - np.clip(*minus)) / (2 * step),
            expected,
            atol=2e-4 if dtype == "float32" else 2e-10,
            rtol=2e-3 if dtype == "float32" else 2e-9,
        )
    finally:
        source.close()


def test_boundary_selected_branch_convention_and_default_ad_gate() -> None:
    function = lambda x, lo, hi: xp.clip(x, lo, hi)
    args = (np.array([-1.0, 0.0, 1.0]), np.array(-1.0), np.array(1.0))
    directions = (np.ones(3), np.array(2.0), np.array(3.0))
    with pytest.raises(ValueError, match="Boolean"):
        xp.jvp(function, args, directions, backend="reference")
    _, tangent = xp.jvp(
        function, args, directions, backend="reference", allow_comparison_where=True
    )
    np.testing.assert_array_equal(tangent, [1, 1, 1])
    # Equality keeps x; no differentiability claim at the tie.
    _, tangent = xp.jvp(
        function,
        (np.zeros(3), np.array(0.0), np.array(0.0)),
        directions,
        backend="reference",
        allow_comparison_where=True,
    )
    np.testing.assert_array_equal(tangent, 1)


def test_native_fp64_forward_and_selected_branch_ad(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native clip used the reference interpreter")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    function = lambda x, lo, hi: xp.clip(x, lo, hi)
    args = (np.array([[-2.0], [0.0], [2.0]]), np.array([-1.0, -0.5]), np.array(1.0))
    source = xp.compile(function, backend="cpu", cache=tmp_path)
    try:
        np.testing.assert_array_equal(source(*args), np.clip(*args))
        _, tangent = xp.jvp(
            source,
            args,
            tuple(np.ones_like(a) for a in args),
            allow_comparison_where=True,
        )
        np.testing.assert_array_equal(tangent, 1)
        _, pullback = xp.vjp(
            source, *args, argnums=(0, 1, 2), allow_comparison_where=True
        )
        try:
            bx, blo, bhi = pullback(np.ones((3, 2)))
            np.testing.assert_array_equal(bx, [[0], [2], [0]])
            np.testing.assert_array_equal(blo, [1, 1])
            np.testing.assert_array_equal(bhi, 2)
        finally:
            pullback.close()
    finally:
        source.close()


def test_capabilities_separate_reference_from_unqualified_native() -> None:
    matrix = xp.capabilities()["operation_matrix"]["clip"]
    assert matrix["capture"] == "tensorir"
    assert matrix["native_cpu"] is False and matrix["native_cuda"] is False
    assert matrix["native_cpu_notes"] == "not-qualified"
    assert "nondifferentiable-at-ties" in matrix["ad"]


def test_float32_native_and_cuda_admission_remain_closed() -> None:
    from generativeqc_compiler.tensor.cpu import describe_cpu_compilation
    from generativeqc_compiler.tensor.optimize import prepare_for_backend

    source = xp.compile(lambda x: xp.clip(x, 0, 1), backend="reference")
    try:
        with pytest.raises(ValueError, match="float64"):
            describe_cpu_compilation(source.lower(np.ones(2, dtype=np.float32)))
        with pytest.raises(ValueError):
            prepare_for_backend(source.lower(np.ones(2)), backend="cuda")
    finally:
        source.close()

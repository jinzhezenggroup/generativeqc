"""Finite minimum/maximum capture against independent NumPy and strict oracles."""

from dataclasses import replace
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


@pytest.mark.parametrize("name", ("minimum", "maximum"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("shape", ((), (0, 3), (2, 1, 3)))
def test_direct_callable_two_oracles_and_serialization(
    name: str, dtype: str, shape: tuple
) -> None:
    left = (np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape) - 2) / 2
    right = np.array([-1, 0, 2], dtype=dtype) if shape else np.array(1, dtype=dtype)
    operation = getattr(xp, name)
    expected = getattr(np, name)(left, right)
    independent = np.asarray(
        getattr(strict, name)(strict.asarray(left), strict.asarray(right))
    )
    source = xp.compile(operation, backend="reference")
    try:
        program = source.lower(left, right)
        assert source.lower(left, right).dumps() == program.dumps()
        restored = Program.loads(program.dumps())
        assert restored.dumps() == program.dumps()
        assert restored.logical_hash == program.logical_hash
        before = left.copy(), right.copy()
        for result in (
            operation(xp.asarray(left), xp.asarray(right)),
            source(left, right),
            source(left, right),
            source(x1=left, x2=right),
            execute(restored, {"x1": left, "x2": right}).outputs["output"],
            execute(optimize(program), {"x1": left, "x2": right}).outputs["output"],
        ):
            assert result.dtype == expected.dtype and result.shape == expected.shape
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(result, independent)
            assert not np.shares_memory(result, left) and not np.shares_memory(
                result, right
            )
        np.testing.assert_array_equal(left, before[0])
        np.testing.assert_array_equal(right, before[1])
    finally:
        source.close()


@pytest.mark.parametrize("name", ("minimum", "maximum"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("scalar", (-3, 0, 0.1, 2.5))
@pytest.mark.parametrize("swapped", (False, True))
def test_weak_scalars_both_sides(
    name: str, dtype: str, scalar: float, swapped: bool
) -> None:
    operation = getattr(xp, name)
    x = np.array([-4, -1, 0, 1, 4], dtype=dtype)
    function = (
        (lambda x: operation(scalar, x))
        if swapped
        else (lambda x: operation(x, scalar))
    )
    source = xp.compile(function, backend="reference")
    args = (scalar, strict.asarray(x)) if swapped else (strict.asarray(x), scalar)
    try:
        expected = np.asarray(getattr(strict, name)(*args))
        np.testing.assert_array_equal(expected, getattr(np, name)(x, scalar))
        for result in (source(x), function(xp.asarray(x))):
            assert result.dtype == x.dtype
            np.testing.assert_array_equal(result, expected)
    finally:
        source.close()


@pytest.mark.parametrize("name", ("minimum", "maximum"))
@pytest.mark.parametrize("swapped", (False, True))
def test_mixed_real_promotion_and_multiaxis_broadcast(name: str, swapped: bool) -> None:
    args = (
        np.array([[[-2], [2]]], dtype="float32"),
        np.array([-1, 0, 3], dtype="float64"),
    )
    if swapped:
        args = args[::-1]
    source = xp.compile(getattr(xp, name), backend="reference")
    try:
        result = source(*args)
        assert result.dtype == np.dtype("float64") and result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result, getattr(np, name)(*args))
        np.testing.assert_array_equal(
            result,
            np.asarray(getattr(strict, name)(*(strict.asarray(v) for v in args))),
        )
    finally:
        source.close()


@pytest.mark.parametrize("name", ("minimum", "maximum"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_finite_extremes_and_first_operand_zero_ties(name: str, dtype: str) -> None:
    limit = np.finfo(dtype)
    left = np.array(
        [-limit.max, -limit.tiny, -0.0, 0.0, limit.tiny, limit.max], dtype=dtype
    )
    right = left[::-1].copy()
    source = xp.compile(getattr(xp, name), backend="reference")
    try:
        for a, b in ((left, right), (right, left), (left, left)):
            result = source(a, b)
            np.testing.assert_array_equal(result, getattr(np, name)(a, b))
            np.testing.assert_array_equal(
                result,
                np.asarray(getattr(strict, name)(strict.asarray(a), strict.asarray(b))),
            )
            # The standard permits either zero sign; capture retains x1 on ties.
            np.testing.assert_array_equal(
                np.signbit(result[a == b]), np.signbit(a[a == b])
            )
    finally:
        source.close()


@pytest.mark.parametrize("name", ("minimum", "maximum"))
def test_invalid_static_and_annotated_domains(name: str) -> None:
    operation = getattr(ns, name)
    x = symbolic("x", (2, 3))
    for bad in (True, 1j, object(), [0], np.array(0.0)):
        for args in ((x, bad), (bad, x)):
            with pytest.raises((TypeError, ValueError)):
                operation(*args)
    for bad in (float("nan"), float("inf"), -float("inf"), -0.0, 10**1000):
        with pytest.raises(ValueError):
            operation(symbolic("empty", (0,)), bad)
    with pytest.raises(ValueError, match="finite"):
        operation(symbolic("empty", (0,), "float32"), 1e100)
    with pytest.raises(TypeError, match="at least one"):
        operation(1, 2)
    with pytest.raises(TypeError, match="real float"):
        operation(x, symbolic("mask", (), "bool"))
    with pytest.raises(ValueError, match="broadcast"):
        operation(x, symbolic("badshape", (4,)))
    specs = [
        TensorSpec((Index("i", IndexSpace(space, space, 3)),), role="input")
        for space in ("ao", "occupied", "virtual", "auxiliary", "spin")
    ] + [
        replace(x.node.spec, representation="spin_orbital"),
        replace(symbolic("square", (2, 2)).node.spec, symmetries=(Symmetry((1, 0)),)),
        TensorSpec(
            (Index("i", IndexSpace("_array_extent_3", "matrix", 3, "alpha")),),
            role="input",
        ),
    ]
    for spec in specs:
        annotated = VibeArray(ir.input_tensor("annotated", spec))
        for args in ((x, annotated), (annotated, x)):
            with pytest.raises(TypeError, match="scientific annotations"):
                operation(*args)


@pytest.mark.parametrize("name", ("minimum", "maximum"))
def test_runtime_rejection_and_eager_ieee_preserved(name: str) -> None:
    operation = getattr(xp, name)
    source = xp.compile(operation, backend="reference")
    try:
        for position in (0, 1):
            for bad in (np.nan, np.inf, -np.inf):
                args = [np.array([-1.0, 2.0]), np.array([0.0, 1.0])]
                args[position] = np.full(2, bad)
                with pytest.raises(ValueError, match="finite"):
                    source(*args)
                np.testing.assert_array_equal(
                    operation(*(xp.asarray(a) for a in args)), getattr(np, name)(*args)
                )
        for bad in (np.array([1, 2]), np.array([True, False]), np.array([1j, 2j])):
            with pytest.raises((ValueError, TypeError)):
                source(bad, bad)

        class Device:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (2, 0)

        with pytest.raises((ValueError, TypeError)):
            source(Device(), np.ones(2))
        with pytest.raises((ValueError, TypeError)):
            source(1, 2)
    finally:
        source.close()
    assert xp.__array_api_version__ == "2025.12"


@pytest.mark.parametrize("name", ("minimum", "maximum"))
@pytest.mark.parametrize("dtype,budget", (("float32", 2048), ("float64", 4096)))
@pytest.mark.parametrize("scalar", (False, True))
def test_logical_resource_budgets(
    name: str, dtype: str, budget: int, scalar: bool
) -> None:
    args = (
        np.arange(8, dtype=dtype).reshape(8, 1),
        np.arange(8, dtype=dtype).reshape(1, 8),
    )
    operation = getattr(xp, name)
    function = (lambda x1, x2: operation(x1 + x2, 0.5)) if scalar else operation
    source = xp.compile(function, backend="reference", max_bytes=budget)
    try:
        expected = (
            getattr(np, name)(args[0] + args[1], 0.5)
            if scalar
            else getattr(np, name)(*args)
        )
        np.testing.assert_array_equal(source(*args), expected)
    finally:
        source.close()
    for backend, limits in (
        ("reference", {"max_bytes": 128}),
        ("cpu", {"max_work": 0}),
        ("cpu", {"max_nodes": 1}),
    ):
        source = xp.compile(function, backend=backend, **limits)
        try:
            with pytest.raises(ValueError, match="budget"):
                source(*(v.astype("float64") for v in args))
        finally:
            source.close()


@pytest.mark.parametrize("name", ("minimum", "maximum"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_interior_jvp_vjp_dot_and_finite_difference(name: str, dtype: str) -> None:
    feeds = {
        "x1": np.array([[-2.0], [2.0]], dtype=dtype),
        "x2": np.array([-1.0, 0.5, 1.0], dtype=dtype),
    }
    directions = {
        "x1": np.array([[0.2], [-0.4]], dtype=dtype),
        "x2": np.array([-0.3, 0.6, 0.8], dtype=dtype),
    }
    weights = np.arange(1, 7, dtype=dtype).reshape(2, 3) / 8
    choose = (
        feeds["x1"] < feeds["x2"] if name == "minimum" else feeds["x1"] > feeds["x2"]
    )
    expected = np.where(choose, directions["x1"], directions["x2"])
    source = xp.compile(
        getattr(xp, name), backend="reference", differentiable=tuple(feeds)
    )
    try:
        program = source.lower(*feeds.values())
        forward = autodiff.jvp(program, feeds, directions, allow_comparison_where=True)
        reverse = autodiff.vjp(
            program, feeds, {"output": weights}, allow_comparison_where=True
        )
        tol = 2e-6 if dtype == "float32" else 2e-13
        np.testing.assert_allclose(
            forward.output_tangents["output"], expected, atol=tol, rtol=tol
        )
        np.testing.assert_allclose(
            reverse.input_cotangents["x1"],
            np.where(choose, weights, 0).sum(axis=1, keepdims=True),
            atol=tol,
            rtol=tol,
        )
        np.testing.assert_allclose(
            reverse.input_cotangents["x2"],
            np.where(choose, 0, weights).sum(axis=0),
            atol=tol,
            rtol=tol,
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
            (getattr(np, name)(*plus) - getattr(np, name)(*minus)) / (2 * step),
            expected,
            atol=2e-4 if dtype == "float32" else 2e-10,
            rtol=2e-3 if dtype == "float32" else 2e-9,
        )
    finally:
        source.close()


@pytest.mark.parametrize("name", ("minimum", "maximum"))
def test_tie_branch_is_only_an_opt_in_convention(name: str) -> None:
    args = np.zeros(3), np.array(0.0)
    directions = np.ones(3), np.array(2.0)
    with pytest.raises(ValueError, match="Boolean"):
        xp.jvp(getattr(xp, name), args, directions, backend="reference")
    _, tangent = xp.jvp(
        getattr(xp, name),
        args,
        directions,
        backend="reference",
        allow_comparison_where=True,
    )
    np.testing.assert_array_equal(tangent, 1)


@pytest.mark.parametrize("name", ("minimum", "maximum"))
def test_native_fp64_forward_jvp_vjp_without_reference(
    name: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native extrema used the reference interpreter")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)
    args = np.array([[-2.0], [2.0]]), np.array([-1.0, 0.5, 1.0])
    directions = np.array([[0.2], [-0.4]]), np.array([-0.3, 0.6, 0.8])
    weights = np.arange(1, 7).reshape(2, 3) / 8
    choose = args[0] < args[1] if name == "minimum" else args[0] > args[1]
    source = xp.compile(getattr(xp, name), backend="cpu", cache=tmp_path)
    try:
        np.testing.assert_array_equal(source(*args), getattr(np, name)(*args))
        zeros = np.array([[-0.0], [0.0]]), np.array([0.0, -0.0, 0.0])
        np.testing.assert_array_equal(
            np.signbit(source(*zeros)), np.broadcast_to(np.signbit(zeros[0]), (2, 3))
        )
        _, tangent = xp.jvp(source, args, directions, allow_comparison_where=True)
        np.testing.assert_allclose(
            tangent, np.where(choose, *directions), atol=2e-13, rtol=2e-13
        )
        _, pullback = xp.vjp(source, *args, argnums=(0, 1), allow_comparison_where=True)
        try:
            bx, by = pullback(weights)
            np.testing.assert_array_equal(
                bx, np.where(choose, weights, 0).sum(axis=1, keepdims=True)
            )
            np.testing.assert_array_equal(by, np.where(choose, 0, weights).sum(axis=0))
        finally:
            pullback.close()
    finally:
        source.close()


@pytest.mark.parametrize("name", ("minimum", "maximum"))
def test_unqualified_capabilities_and_backend_admission(name: str) -> None:
    from generativeqc_compiler.tensor.cpu import describe_cpu_compilation
    from generativeqc_compiler.tensor.optimize import prepare_for_backend

    matrix = xp.capabilities()["operation_matrix"][name]
    assert (
        matrix["capture"] == "tensorir" and "nondifferentiable-at-ties" in matrix["ad"]
    )
    assert matrix["native_cpu"] is False and matrix["native_cuda"] is False
    assert matrix["native_cpu_notes"] == "not-qualified"
    source = xp.compile(getattr(xp, name), backend="reference")
    try:
        with pytest.raises(ValueError, match="float64"):
            describe_cpu_compilation(
                source.lower(np.ones(2, dtype="float32"), np.ones(2, dtype="float32"))
            )
        with pytest.raises(ValueError):
            prepare_for_backend(source.lower(np.ones(2), np.ones(2)), backend="cuda")
    finally:
        source.close()

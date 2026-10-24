"""Finite static product capture: independent polynomial and Array API oracles."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
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


def symbolic(shape: tuple[int, ...], dtype: str = "float64") -> VibeArray:
    return input_array(
        "x", TensorSpec(ns._generic_indices(shape), dtype=dtype, role="input")
    )


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize(
    "shape,axis",
    (
        ((), None),
        ((), ()),
        ((1,), 0),
        ((2, 3, 5), None),
        ((2, 3, 5), 1),
        ((2, 3, 5), -1),
        ((2, 3, 5), (0, -1)),
        ((2, 3, 5), (-1, 0)),
        ((2, 3, 5), ()),
        ((0, 3, 5), 1),
        ((2, 0, 5), (0, 2)),
        ((0, 3), ()),
        ((2, 0, 5), 1),
        ((0, 0), 0),
        ((0,), None),
    ),
)
@pytest.mark.parametrize("keepdims", (False, True))
def test_oracles_serialization_and_replay(
    dtype: str, shape: tuple, axis: Any, keepdims: bool
) -> None:
    x = np.random.default_rng(302).uniform(-1.5, 1.5, size=shape).astype(dtype)
    if x.ndim:
        x = x[..., ::-1]
    before = x.copy()
    function = lambda x: xp.prod(x, axis=axis, keepdims=keepdims)
    expected = np.prod(x, axis=axis, keepdims=keepdims)
    independent = np.asarray(
        strict.prod(strict.asarray(x.copy()), axis=axis, keepdims=keepdims)
    )
    source = xp.compile(function, backend="reference")
    try:
        program = source.lower(x)
        restored = Program.loads(program.dumps())
        assert restored.dumps() == program.dumps() == source.lower(x).dumps()
        for result in (
            source(x),
            source(x),
            function(xp.asarray(x)),
            execute(restored, {"x": x}).outputs["output"],
            execute(optimize(program), {"x": x}).outputs["output"],
        ):
            assert result.shape == expected.shape and result.dtype == x.dtype
            tolerance = 3e-6 if dtype == "float32" else 3e-13
            np.testing.assert_allclose(result, expected, rtol=tolerance, atol=1e-15)
            np.testing.assert_allclose(result, independent, rtol=tolerance, atol=1e-15)
            assert not np.shares_memory(result, x)
        np.testing.assert_array_equal(x, before)
    finally:
        source.close()


@pytest.mark.parametrize("length", (2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 4097))
def test_even_and_odd_tail_exact_polynomial(length: int) -> None:
    # Distinct nonunit factors at both ends catch dropped/repeated odd tails.
    x = np.resize(np.array([-1.0, 1.0, -1.0, -1.0]), length)
    x[0], x[-1] = 2.0, -3.0
    expected = 1.0
    for item in x:
        expected *= float(item)
    source = xp.compile(lambda x: xp.prod(x), backend="reference")
    try:
        assert source(x) == expected
    finally:
        source.close()


def polynomial_oracle(
    x: np.ndarray,
    direction: np.ndarray,
    weights: np.ndarray,
    axes: tuple[int, ...],
    keepdims: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate each scalar monomial by omitting one factor, without division."""
    retained = tuple(i for i in range(x.ndim) if i not in axes)
    output = tuple(x.shape[i] for i in retained)
    tangent, adjoint = np.zeros(output, dtype=x.dtype), np.zeros_like(x)
    for out in np.ndindex(output):
        indices = []
        for reduced in np.ndindex(tuple(x.shape[i] for i in axes)):
            index = [0] * x.ndim
            for i, v in zip(retained, out, strict=True):
                index[i] = v
            for i, v in zip(axes, reduced, strict=True):
                index[i] = v
            indices.append(tuple(index))
        for index in indices:
            derivative = 1.0
            for other in indices:
                if other != index:
                    derivative *= float(x[other])
            tangent[out] += derivative * direction[index]
            adjoint[index] = derivative * weights.reshape(output)[out]
    target = (
        tuple(1 if i in axes else n for i, n in enumerate(x.shape))
        if keepdims
        else output
    )
    return tangent.reshape(target), adjoint


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("axes,keepdims", (((1,), False), ((0, 2), True), ((), False)))
@pytest.mark.parametrize("zeros", (0, 1, 2))
def test_reference_zero_safe_first_order(
    dtype: str, axes: tuple, keepdims: bool, zeros: int
) -> None:
    x = np.linspace(-1.2, 1.3, 30, dtype=dtype).reshape(2, 3, 5)
    x.reshape(-1)[-zeros:] = 0 if zeros else x.reshape(-1)
    direction = np.linspace(-0.4, 0.7, 30, dtype=dtype).reshape(x.shape)
    expected = np.prod(x, axis=axes, keepdims=keepdims)
    weights = np.full(expected.shape, 0.75, dtype=dtype)
    tangent, adjoint = polynomial_oracle(x, direction, weights, axes, keepdims)
    source = xp.compile(
        lambda x: xp.prod(x, axis=axes, keepdims=keepdims),
        backend="reference",
        differentiable=("x",),
    )
    try:
        program = source.lower(x)
        feeds, directions = {"x": x}, {"x": direction}
        tol = 3e-6 if dtype == "float32" else 3e-13
        np.testing.assert_allclose(
            autodiff.jvp(program, feeds, directions).output_tangents["output"],
            tangent,
            rtol=tol,
            atol=tol,
        )
        np.testing.assert_allclose(
            autodiff.vjp(program, feeds, {"output": weights}).input_cotangents["x"],
            adjoint,
            rtol=tol,
            atol=tol,
        )
        assert autodiff.dot_test(
            program, feeds, directions, {"output": weights}, rtol=tol, atol=tol
        ).passed
        step = 0.002 if dtype == "float32" else 1e-5
        difference = (
            np.prod(x + step * direction, axis=axes, keepdims=keepdims)
            - np.prod(x - step * direction, axis=axes, keepdims=keepdims)
        ) / (2 * step)
        np.testing.assert_allclose(
            difference, tangent, rtol=2e-3, atol=3e-5 if dtype == "float32" else 1e-9
        )
    finally:
        source.close()


@pytest.mark.parametrize(
    "shape,axes,keepdims",
    (
        ((2, 3, 5), (1,), False),
        ((2, 3, 5), (0, 2), True),
        ((2, 3, 5), (), False),
        ((0,), (0,), False),
        ((2, 0, 3), (1,), True),
        ((2, 0, 3), (0, 2), True),
        ((2, 1, 3), (1,), True),
        ((5,), (0,), False),
        ((), (), False),
    ),
)
@pytest.mark.parametrize("zeros", (0, 1, 2))
def test_native_fp64_no_reference(
    shape: tuple,
    axes: tuple,
    keepdims: bool,
    zeros: int,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native prod used reference execution/AD")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)
    x = np.linspace(-1.1, 1.3, np.prod(shape, dtype=int)).reshape(shape)
    if zeros and x.size > 2:
        x.reshape(-1)[-zeros:] = 0.0
    direction = np.full(shape, 0.3)
    expected = np.prod(x, axis=axes, keepdims=keepdims)
    weights = np.full(expected.shape, 0.7)
    tangent, adjoint = polynomial_oracle(x, direction, weights, axes, keepdims)
    function = (
        xp.prod
        if axes == tuple(range(x.ndim)) and not keepdims
        else lambda x: xp.prod(x, axis=axes, keepdims=keepdims)
    )
    source = xp.compile(
        function,
        backend="cpu",
        cache=tmp_path,
    )
    try:
        for _ in range(2):
            np.testing.assert_allclose(source(x), expected, rtol=3e-13, atol=3e-13)
        value, actual = xp.jvp(source, (x,), (direction,))
        np.testing.assert_allclose(value, expected, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(actual, tangent, rtol=3e-13, atol=3e-13)
        _, pullback = xp.vjp(source, x, argnums=0)
        try:
            (actual_adjoint,) = pullback(weights)
            np.testing.assert_allclose(actual_adjoint, adjoint, rtol=3e-13, atol=3e-13)
        finally:
            pullback.close()
    finally:
        source.close()


def test_validation_dtype_metadata_and_nonfinite() -> None:
    for axis in (True, 1.0, [0], (True,), (0.0,)):
        with pytest.raises(TypeError):
            ns.prod(symbolic((2, 3)), axis=axis)
    for axis in (2, -3, (0, 0), (1, -1)):
        with pytest.raises(ValueError):
            ns.prod(symbolic((2, 3)), axis=axis)
    for keepdims in (0, 1, None, np.bool_(True)):
        with pytest.raises(TypeError, match="keepdims"):
            ns.prod(symbolic((2, 3)), keepdims=keepdims)
    with pytest.raises(TypeError, match="real float"):
        ns.prod(symbolic((3,), "bool"), axis=())
    base = symbolic((2, 2)).node.spec
    specs = [
        replace(base, representation="spin_orbital"),
        replace(base, symmetries=(Symmetry((1, 0)),)),
    ]
    specs += [
        TensorSpec((Index("i", IndexSpace("annotated", kind, 2)),), role="input")
        for kind in ("ao", "occupied", "virtual", "auxiliary", "spin")
    ]
    specs.append(
        TensorSpec(
            (Index("i", IndexSpace("_array_extent_2", "matrix", 2, "alpha")),),
            role="input",
        )
    )
    for spec in specs:
        for axis in (None, ()):
            with pytest.raises(TypeError, match="scientific annotations"):
                ns.prod(VibeArray(ir.input_tensor("x", spec)), axis=axis)
    for dtype in ("float32", "float64"):

        def factory(dtype: str) -> Any:
            return lambda x: xp.prod(x, dtype=getattr(xp, dtype))

        source = xp.compile(factory(dtype), backend="reference")
        try:
            assert source(np.array([2, 3], dtype=dtype)).dtype == np.dtype(dtype)
        finally:
            source.close()
    source = xp.compile(lambda x: xp.prod(x, dtype=xp.float32), backend="reference")
    try:
        result = source(np.array([1.25, -2.5, 3.75]))
        assert result.dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            result, np.prod(np.array([1.25, -2.5, 3.75], dtype="float32"))
        )
    finally:
        source.close()
    source = xp.compile(lambda x: xp.prod(x), backend="reference")
    try:
        for bad in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match="finite"):
                source(np.array([1.0, bad]))
            np.testing.assert_array_equal(
                xp.prod(xp.asarray([1.0, bad])), np.prod([1.0, bad])
            )
        for dtype in ("int64", "complex128"):
            with pytest.raises((ValueError, TypeError)):
                source(np.ones(3, dtype=dtype))
    finally:
        source.close()


def test_rank64_and_signed_zero() -> None:
    x = np.array([-2.0, 3.0]).reshape((1,) * 63 + (2,))
    source = xp.compile(
        lambda x: xp.prod(x, axis=(-1, 0), keepdims=True), backend="reference"
    )
    try:
        np.testing.assert_array_equal(
            source(x), np.prod(x, axis=(-1, 0), keepdims=True)
        )
    finally:
        source.close()
    source = xp.compile(lambda x: xp.prod(x), backend="reference")
    try:
        for x in (
            np.array([0.0, -2.0, 3.0]),
            np.array([-0.0, -2.0, 3.0]),
            np.array([-0.0, 0.0, -2.0, 3.0, 1.0]),
        ):
            assert np.signbit(source(x)) == np.signbit(np.prod(x))
    finally:
        source.close()


def test_resource_preflight_and_runtime_budgets() -> None:
    for shape in ((65_536,), (2,) * 33, (4096, 4096)):
        with pytest.raises(ValueError, match="graph budget"):
            ns.prod(symbolic(shape))
    for shape, axis in (((34_000_000,), ()), ((0, 34_000_000), 0), ((2, 6_000_000), 0)):
        with pytest.raises(ValueError, match="graph budget"):
            ns.prod(symbolic(shape), axis=axis)
    assert ns.prod(symbolic((0, 100_000)), axis=1).shape == (0,)
    assert ns.prod(symbolic((2, 5_000_000)), axis=0).shape == (5_000_000,)
    assert ns.prod(symbolic((3, 3_500_000)), axis=0).shape == (3_500_000,)
    with pytest.raises(ValueError, match="graph budget"):
        ns.prod(symbolic((3, 3_800_000)), axis=0)
    # Compare the conservative planner with public live graph storage, including
    # all gather canonical views and the final reshape, on unequal/odd axes.
    for shape, axes in (
        ((3, 5, 7), (0, 2)),
        ((2, 9), (1,)),
        ((0, 3), (0,)),
        ((0, 3), (1,)),
        ((1,), ()),
        ((2, 5_000_000), (0,)),
        ((3, 3_500_000), (0,)),
    ):
        x = symbolic(shape)
        result = ns.prod(x, axis=axes)
        program = Program({"output": result.node})
        new_nodes = [n for n in program.live_nodes if n is not x.node]
        assert (
            sum(n.spec.size * n.spec.itemsize for n in new_nodes) <= 256 * 1024 * 1024
        )
        assert len(new_nodes) <= 512
        assert (
            sum(len(n.attrs["positions"]) for n in new_nodes if n.op == "gather")
            <= 65_536
        )
    for limits in ({"max_bytes": 1}, {"max_nodes": 1}, {"max_work": 0}):
        source = xp.compile(lambda x: xp.prod(x), backend="cpu", **limits)
        try:
            with pytest.raises(ValueError, match="budget"):
                source(np.arange(15.0).reshape(3, 5))
        finally:
            source.close()


def test_capabilities_and_unqualified_backends() -> None:
    from generativeqc_compiler.array_api.capabilities import capabilities
    from generativeqc_compiler.tensor.cpu import describe_cpu_compilation

    matrix = capabilities()["operation_matrix"]["prod"]
    assert capabilities()["array_api_version"] is None
    assert matrix["capture"] == "tensorir" and "zero-safe" in matrix["ad"]
    assert matrix["native_cpu"] is False and matrix["native_cuda"] is False
    source = xp.compile(lambda x: xp.prod(x), backend="reference")
    try:
        with pytest.raises(ValueError, match="float64"):
            describe_cpu_compilation(source.lower(np.ones(3, dtype="float32")))
    finally:
        source.close()


def test_cuda_public_admission_before_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc.experimental._array_api_execution import _Executor
    from generativeqc_compiler.tensor.optimize import prepare_for_backend

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unqualified prod reached native executor")

    monkeypatch.setattr(_Executor, "_build", forbidden)
    x = np.array([2.0, 0.0, -3.0])

    def helper(x: Any) -> Any:
        return xp.prod(x)

    for function in (xp.prod, lambda x: xp.prod(x), lambda x: helper(x)):
        source = xp.compile(function, backend="cuda")
        try:
            for action in (
                lambda source=source: source.lower(x),
                lambda source=source: source(x),
                lambda source=source: xp.jvp(source, (x,), (np.ones_like(x),)),
                lambda source=source: xp.vjp(source, x),
            ):
                with pytest.raises(ValueError, match="prod.*not qualified"):
                    action()
            gradient = xp.grad(source)
            try:
                with pytest.raises(ValueError, match="prod.*not qualified"):
                    gradient(x)
            finally:
                gradient.close()
        finally:
            source.close()
    # Generic compiler IR has no frontend-operation identity. Its existing CUDA
    # primitive admission remains independent of the unqualified public API.
    raw = Program({"output": ns.prod(symbolic((3,))).node})
    assert prepare_for_backend(raw, backend="cuda").outputs["output"].spec.shape == ()
    ordinary = xp.compile(lambda x: x * x, backend="cuda")
    try:
        prepare_for_backend(ordinary.lower(x), backend="cuda")
    finally:
        ordinary.close()


def test_capture_backend_nested_and_thread_isolation() -> None:
    x = np.array([2.0, 3.0])
    reference = xp.compile(lambda x: xp.prod(x), backend="reference")
    cuda = xp.compile(lambda x: xp.prod(x), backend="cuda")

    def nested_reference(x: Any) -> Any:
        with pytest.raises(ValueError, match="not qualified"):
            cuda.lower(np.array([2.0, 3.0]))
        return xp.prod(x)

    def nested_cuda(x: Any) -> Any:
        assert reference(np.array([2.0, 3.0])) == 6.0
        return xp.prod(x)

    sources = [
        xp.compile(nested_reference, backend="reference"),
        xp.compile(nested_cuda, backend="cuda"),
    ]
    try:
        assert sources[0](x) == 6.0
        with pytest.raises(ValueError, match="not qualified"):
            sources[1].lower(x)
        barrier = Barrier(2)

        def run(backend: str) -> bool:
            def simultaneous(x: Any) -> Any:
                barrier.wait(timeout=10)
                return xp.prod(x)

            source = xp.compile(simultaneous, backend=backend)
            try:
                if backend == "cuda":
                    with pytest.raises(ValueError, match="not qualified"):
                        source.lower(x)
                else:
                    assert source(x) == 6.0
                return True
            finally:
                source.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(run, ("reference", "cuda"))) == [True, True]
        assert reference(x) == 6.0
    finally:
        reference.close()
        cuda.close()
        for source in sources:
            source.close()
    source = xp.compile(lambda x: xp.prod(x), backend="cuda")
    try:
        with pytest.raises(ValueError, match="not qualified"):
            source.lower(np.ones(3))
    finally:
        source.close()
    # A rejected capture must not leak its backend into the next capture.
    source = xp.compile(xp.prod, backend="reference")
    try:
        assert source(np.array([2.0, 3.0])) == 6.0
    finally:
        source.close()

"""Public fixed-mask scalar gradients against independent NumPy mathematics."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp


@pytest.fixture(params=("reference", "cpu"))
def backend(request: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "cpu":
        from generativeqc_compiler.tensor import autodiff, interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native gradient used reference execution or AD")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    return request.param


def energy(x: Any, y: Any, mask: Any) -> Any:
    selected = xp.where(mask, xp.take(x, (2, 0, 2), axis=0) ** 2, y**2)
    return xp.sum(selected)


def expected(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple:
    gathered = x[[2, 0, 2]]
    value = np.sum(np.where(mask, gathered**2, y**2))
    dx = np.zeros_like(x)
    np.add.at(dx, [2, 0, 2], np.where(mask, 2 * gathered, 0))
    dy = np.where(mask, 0, 2 * y).sum(axis=0)
    return value, dx, dy


@pytest.mark.parametrize("with_value", (False, True))
def test_replay_gather_broadcast_order_and_finite_difference(
    backend: str, with_value: bool, tmp_path: Path
) -> None:
    x = np.asarray([[0.4, 0.8], [1.1, 0.3], [0.5, 1.3]])
    y = np.asarray([0.7, 1.2])
    mask = np.asarray([[True], [False], [True]])
    source = xp.compile(energy, backend=backend, cache=tmp_path)
    transform = xp.value_and_grad if with_value else xp.grad
    derivative = transform(source, argnums=(-2, 0), allow_external_mask_where=True)
    try:
        for selector in (mask, ~mask, np.ones_like(mask), np.zeros_like(mask)):
            value, dx, dy = expected(x, y, selector)
            result = derivative(x=x, y=y, mask=selector)
            if with_value:
                actual_value, result = result
                np.testing.assert_allclose(actual_value, value, rtol=2e-14, atol=2e-14)
            actual_dy, actual_dx = result
            np.testing.assert_allclose(actual_dx, dx, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(actual_dy, dy, rtol=2e-14, atol=2e-14)
            direction = np.asarray([[0.1, -0.2], [0.3, 0.1], [-0.4, 0.2]])
            step = 1e-5
            fd = (
                expected(x + step * direction, y, selector)[0]
                - expected(x - step * direction, y, selector)[0]
            ) / (2 * step)
            np.testing.assert_allclose(
                fd, np.sum(actual_dx * direction), rtol=2e-9, atol=2e-9
            )
        if backend == "cpu":
            assert derivative.compilation_info()["native_executables"] == 1
        derivative.close()
        with pytest.raises(RuntimeError, match="closed"):
            derivative(x, y, mask)
        np.testing.assert_allclose(source(x, y, mask), expected(x, y, mask)[0])
    finally:
        derivative.close()
        source.close()


@pytest.mark.parametrize("shape", ((), (0, 2), (2, 3)))
@pytest.mark.parametrize("with_value", (False, True))
def test_constant_and_external_masks_disconnected_inputs(
    backend: str, shape: tuple, with_value: bool, tmp_path: Path
) -> None:
    def expression(x: Any, unused: Any, mask: Any) -> Any:
        return xp.sum(xp.where(mask, x * x, x))

    x, unused = np.full(shape, 0.5), np.ones((3,))
    mask = np.full(shape, True)
    transform = xp.value_and_grad if with_value else xp.grad
    derivative = transform(
        expression,
        backend=backend,
        argnums=(0, 1),
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    try:
        result = derivative(x, unused, mask)
        if with_value:
            value, result = result
            np.testing.assert_allclose(value, np.sum(x * x))
        np.testing.assert_array_equal(result[0], 2 * x)
        np.testing.assert_array_equal(result[1], np.zeros_like(unused))
        with pytest.raises(ValueError, match="non-finite compiled input: unused"):
            derivative(x, np.full_like(unused, np.nan), mask)
    finally:
        derivative.close()
    constant = xp.grad(
        lambda x: xp.sum(xp.where(True, x * x, x)),
        backend=backend,
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    try:
        np.testing.assert_array_equal(constant(x), 2 * x)
    finally:
        constant.close()


@pytest.mark.parametrize("with_value", (False, True))
@pytest.mark.parametrize("argnums", (0, 1))
@pytest.mark.parametrize(
    "operation,bad",
    ((xp.log, -1.0), (xp.sqrt, 0.0), (xp.reciprocal, 0.0), (xp.exp, 1000.0)),
)
def test_unselected_domains_survive_gradient_only_and_disconnection(
    backend: str,
    with_value: bool,
    argnums: int,
    operation: Any,
    bad: float,
    tmp_path: Path,
) -> None:
    def expression(x: Any, unused: Any, mask: Any) -> Any:
        return xp.sum(xp.where(mask, x * x, operation(x)))

    transform = xp.value_and_grad if with_value else xp.grad
    derivative = transform(
        expression,
        argnums=argnums,
        backend=backend,
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    try:
        with pytest.raises(ValueError):
            derivative(np.asarray([bad]), np.ones(2), np.asarray([True]))
        # A failed call must not poison a cached specialization or owner.
        result = derivative(np.asarray([2.0]), np.ones(2), np.asarray([True]))
        if with_value:
            value, result = result
            assert float(value) == 4.0
        np.testing.assert_array_equal(result, [4.0] if argnums == 0 else [0.0, 0.0])
    finally:
        derivative.close()


@pytest.mark.parametrize("transform", (xp.grad, xp.value_and_grad))
def test_default_and_invalid_opt_in_fail_before_toolchain(
    transform: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common import cpp_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid request reached native compiler discovery")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)
    x, y, mask = np.ones((3, 2)), np.ones(2), np.ones((3, 1), dtype=bool)
    for flag in (False,):
        derivative = transform(
            energy, argnums=(0, 1), cache=tmp_path, allow_external_mask_where=flag
        )
        try:
            with pytest.raises(ValueError, match="Boolean"):
                derivative(x, y, mask)
        finally:
            derivative.close()
    for flag in (1, None, "yes", np.bool_(True)):
        with pytest.raises(TypeError, match="must be Boolean"):
            transform(energy, allow_external_mask_where=flag)
    with pytest.raises(ValueError, match="reference or CPU"):
        transform(energy, backend="cuda", allow_external_mask_where=True)
    for dtype in (np.float32,):
        derivative = transform(
            energy, argnums=(0, 1), cache=tmp_path, allow_external_mask_where=True
        )
        try:
            with pytest.raises(ValueError, match="float64"):
                derivative(x.astype(dtype), y.astype(dtype), mask)
        finally:
            derivative.close()
    derivative = transform(energy, argnums=2, allow_external_mask_where=True)
    try:
        with pytest.raises((TypeError, ValueError)):
            derivative(x, y, mask)
    finally:
        derivative.close()
    for options in ({"max_bytes": 1}, {"max_work": 0}, {"max_nodes": 1}):
        derivative = transform(
            energy,
            argnums=(0, 1),
            cache=tmp_path,
            allow_external_mask_where=True,
            **options,
        )
        try:
            with pytest.raises(ValueError, match="budget"):
                derivative(x, y, mask)
        finally:
            derivative.close()


def test_computed_predicate_and_scientific_domains_reject(backend: str) -> None:
    derivative = xp.grad(
        lambda x: xp.sum(xp.where(x > 0, x, x * x)),
        backend=backend,
        allow_external_mask_where=True,
    )
    try:
        with pytest.raises(ValueError, match="external Boolean mask"):
            derivative(np.ones(2))
    finally:
        derivative.close()
    ao = xp.IndexSpace("ao", "ao", 2)
    spec = xp.TensorSpec((xp.Index("p", ao),), role="input")
    source = xp.compile(
        lambda x, mask: xp.sum(xp.where(mask, x, x * x)),
        backend=backend,
        specs={"x": spec},
    )
    derivative = xp.grad(source, allow_external_mask_where=True)
    try:
        with pytest.raises((TypeError, ValueError), match="scientific"):
            derivative(np.ones(2), np.ones(2, dtype=bool))
    finally:
        derivative.close()
        source.close()


def test_capability_reports_separate_scalar_opt_in() -> None:
    capability = xp.capabilities()["external_mask_gradients"]
    assert capability["transforms"] == ("grad", "value_and_grad")
    assert capability["backends"] == ("reference", "cpu")
    assert capability["default"] is False
    assert capability["real_dtype"] == "float64"


@pytest.mark.parametrize("with_value", (False, True))
def test_generated_execution_boundary_and_detached_results(
    backend: str, with_value: bool, tmp_path: Path
) -> None:
    from generativeqc_compiler.tensor import Program, transpose_program

    transform = xp.value_and_grad if with_value else xp.grad
    derivative = transform(
        lambda x, mask: xp.sum(xp.where(mask, x * x, x)),
        backend=backend,
        allow_external_mask_where=True,
        cache=tmp_path,
    )
    x, mask = np.asarray([0.2, 0.7]), np.asarray([True, False])
    try:
        program = derivative.lower(x, mask)
        replay = Program.loads(program.dumps())
        assert all(
            not node.spec.differentiable for node in replay.nodes if node.op == "input"
        )
        # Removing provenance must not reopen the typed execution-only boundary.
        stripped = Program(replay.outputs, replay.nodes)
        with pytest.raises(ValueError):
            transpose_program(
                stripped,
                tuple(stripped.outputs),
                inputs=("x",),
                allow_external_mask_where=True,
            )
        first = derivative(x, mask)
        gradient = first[1] if with_value else first
        assert not np.shares_memory(gradient, x)
        second = derivative(x, mask)
        second_gradient = second[1] if with_value else second
        assert not np.shares_memory(gradient, second_gradient)
        np.testing.assert_allclose(second_gradient, [0.4, 1.0])
        np.testing.assert_array_equal(x, [0.2, 0.7])
        np.testing.assert_array_equal(mask, [True, False])
    finally:
        derivative.close()


def test_mask_views_and_keyword_only_selector(backend: str, tmp_path: Path) -> None:
    def expression(x: Any, *, mask: Any) -> Any:
        return xp.sum(xp.where(xp.flip(mask, axis=0), x * x, x))

    x, mask = np.asarray([0.2, 0.7]), np.asarray([True, False])
    derivative = xp.value_and_grad(
        expression, backend=backend, allow_external_mask_where=True, cache=tmp_path
    )
    try:
        value, gradient = derivative(x, mask=mask)
        np.testing.assert_allclose(value, 0.2 + 0.7**2, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(gradient, [1.0, 1.4], rtol=2e-14, atol=2e-14)
    finally:
        derivative.close()

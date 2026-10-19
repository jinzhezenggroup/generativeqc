"""Public fixed-mask JVP with explicit non-differentiated primal placeholders."""

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
            raise AssertionError("native JVP delegated to a reference oracle")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    return request.param


def expression(x: Any, y: Any, mask: Any, unused: Any) -> dict:
    gathered = xp.take(x, (2, 0, 2), axis=0)
    selected = xp.where(xp.flip(mask, axis=0), gathered * gathered, y * y)
    return {
        "vector": selected,
        "norm": xp.einsum("ij,ij->", selected, selected),
        "constant": xp.ones_like(unused),
    }


def values(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple:
    selected = np.where(mask[::-1], x[[2, 0, 2]] ** 2, y**2)
    return selected, np.sum(selected**2)


@pytest.mark.parametrize("omit_y", (False, True))
def test_nonlinear_broadcast_gather_oracle_finite_difference_and_dot(
    backend: str, omit_y: bool, tmp_path: Path
) -> None:
    x = np.asarray([[0.4, 0.8], [1.1, 0.3], [0.5, 1.3]])
    y = np.asarray([0.7, 1.2])
    dx = np.asarray([[0.1, -0.2], [0.3, 0.1], [-0.4, 0.2]])
    dy = np.asarray([0.3, -0.2])
    direction_y = np.zeros_like(y) if omit_y else dy
    unused = np.ones(4)
    source = xp.compile(expression, backend=backend, cache=tmp_path)
    try:
        for mask in (
            np.asarray([[True], [False], [True]]),
            np.zeros((3, 1), dtype=bool),
            np.ones((3, 1), dtype=bool),
        ):
            primal, tangent = xp.jvp(
                source,
                (x, y, mask, unused),
                (dx, None if omit_y else dy, None, np.ones(4)),
                allow_external_mask_where=True,
            )
            selected, norm = values(x, y, mask)
            expected = np.where(
                mask[::-1], 2 * x[[2, 0, 2]] * dx[[2, 0, 2]], 2 * y * direction_y
            )
            expected_norm = 2 * np.sum(selected * expected)
            np.testing.assert_allclose(
                primal["vector"], selected, rtol=2e-13, atol=2e-13
            )
            np.testing.assert_allclose(primal["norm"], norm, rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(
                tangent["vector"], expected, rtol=2e-13, atol=2e-13
            )
            np.testing.assert_allclose(
                tangent["norm"], expected_norm, rtol=2e-13, atol=2e-13
            )
            np.testing.assert_array_equal(tangent["constant"], np.zeros(4))
            step = 1e-5
            plus = values(x + step * dx, y + step * direction_y, mask)
            minus = values(x - step * dx, y - step * direction_y, mask)
            for actual, a, b in zip(
                (tangent["vector"], tangent["norm"]), plus, minus, strict=True
            ):
                np.testing.assert_allclose(
                    actual, (a - b) / (2 * step), rtol=2e-8, atol=2e-8
                )
            weights = np.asarray([[0.2, 0.8], [-0.3, 0.5], [0.4, -0.2]])
            weight = weights + 0.6 * selected
            bar_x = np.zeros_like(x)
            np.add.at(
                bar_x, [2, 0, 2], np.where(mask[::-1], 2 * x[[2, 0, 2]] * weight, 0)
            )
            bar_y = np.where(mask[::-1], 0, 2 * y * weight).sum(axis=0)
            lhs = np.sum(tangent["vector"] * weights) + 0.3 * tangent["norm"]
            rhs = np.sum(bar_x * dx) + np.sum(bar_y * direction_y)
            np.testing.assert_allclose(lhs, rhs, rtol=2e-13, atol=2e-13)
            assert not np.shares_memory(primal["vector"], tangent["vector"])
            assert not np.shares_memory(tangent["vector"], x)
        assert not source._execution.closed
    finally:
        source.close()


@pytest.mark.parametrize("shape", ((), (0, 2), (2, 3)))
@pytest.mark.parametrize("omit_x", (False, True))
def test_empty_scalar_disconnected_and_defaulted_primals(
    backend: str, shape: tuple, omit_x: bool, tmp_path: Path
) -> None:
    default_scale = np.asarray(2.0)

    def function(x: Any, unused: Any, mask: Any, scale: Any = default_scale) -> dict:
        return {
            "out": xp.where(mask, x * x, x * scale),
            "constant": xp.ones_like(unused),
        }

    x, mask = np.full(shape, 0.5), np.full(shape, True)
    primal, tangent = xp.jvp(
        function,
        (x, np.ones(3), mask),
        (None if omit_x else np.ones(shape), np.ones(3), None),
        backend=backend,
        cache=tmp_path,
        allow_external_mask_where=True,
    )
    np.testing.assert_array_equal(primal["out"], x * x)
    np.testing.assert_array_equal(tangent["out"], np.zeros(shape) if omit_x else 2 * x)
    np.testing.assert_array_equal(tangent["constant"], np.zeros(3))


@pytest.mark.parametrize(
    "operation,bad",
    ((xp.log, -1.0), (xp.sqrt, 0.0), (xp.reciprocal, 0.0), (xp.exp, 1000.0)),
)
@pytest.mark.parametrize("shape", ((1,), (0,)))
def test_unselected_forward_domains_survive_omitted_directions_and_empty_outputs(
    backend: str, operation: Any, bad: float, shape: tuple, tmp_path: Path
) -> None:
    def function(x: Any, bad_branch: Any, unused: Any, mask: Any) -> Any:
        return xp.where(mask, x * x, operation(bad_branch))

    options = {"backend": backend, "cache": tmp_path, "allow_external_mask_where": True}
    tangents = (None, None, np.ones(3), None)
    with pytest.raises(ValueError):
        xp.jvp(
            function,
            (np.ones(shape), np.asarray(bad), np.ones(3), np.full(shape, True)),
            tangents,
            **options,
        )
    _, response = xp.jvp(
        function,
        (np.ones(shape), np.asarray(2.0), np.ones(3), np.full(shape, True)),
        tangents,
        **options,
    )
    np.testing.assert_array_equal(response, np.zeros(shape))


@pytest.mark.parametrize("selector", ("input", True, False))
def test_selected_direction_bits_survive_public_combination(
    backend: str, selector: Any, tmp_path: Path
) -> None:
    largest = np.finfo(np.float64).max
    tiny = np.nextafter(0.0, 1.0)
    dx = np.asarray([-0.0, 0.0, -tiny, tiny, -largest, largest])
    dy = -dx
    mask = np.asarray([True, False, False, True, True, False])

    def function(x: Any, y: Any, mask: Any) -> Any:
        return xp.where(mask if selector == "input" else selector, x, y)

    _, tangent = xp.jvp(
        function,
        (np.ones(6), np.ones(6), mask),
        (dx, dy, None),
        backend=backend,
        cache=tmp_path,
        allow_external_mask_where=True,
    )
    expected = np.where(mask if selector == "input" else selector, dx, dy)
    np.testing.assert_array_equal(
        np.asarray(tangent).view(np.uint64), expected.view(np.uint64)
    )


@pytest.mark.parametrize("domain", ("float32", "scientific"))
def test_requested_disconnected_input_domains_reject(backend: str, domain: str) -> None:
    dtype = np.float32 if domain == "float32" else np.float64
    specs = {}
    if domain == "scientific":
        specs["unused"] = xp.TensorSpec(
            (xp.Index("p", xp.IndexSpace("ao", "ao", 3)),), role="input"
        )
    source = xp.compile(
        lambda x, unused, mask: xp.where(mask, x, x * x), backend=backend, specs=specs
    )
    try:
        with pytest.raises(ValueError, match="float64|generic"):
            xp.jvp(
                source,
                (np.ones(2), np.ones(3, dtype=dtype), np.ones(2, dtype=bool)),
                (None, np.ones(3, dtype=dtype), None),
                allow_external_mask_where=True,
            )
    finally:
        source.close()


@pytest.mark.parametrize("domain", ("float32", "scientific"))
def test_canonical_generation_closes_disconnected_admission_hole(domain: str) -> None:
    from generativeqc_compiler.tensor import linearize

    specs = {}
    dtype = np.float32 if domain == "float32" else np.float64
    if domain == "scientific":
        specs["unused"] = xp.TensorSpec(
            (xp.Index("p", xp.IndexSpace("ao", "ao", 3)),), role="input"
        )
    source = xp.compile(
        lambda mask, x, unused: xp.where(mask, x, x * x),
        backend="reference",
        specs=specs,
        differentiable=("unused",),
    )
    try:
        program = source.lower(
            np.ones(2, dtype=bool), np.ones(2), np.ones(3, dtype=dtype)
        )
        with pytest.raises(ValueError, match="float64|generic"):
            linearize(program, ("unused",), allow_external_mask_where=True)
    finally:
        source.close()


def test_default_and_invalid_admission_before_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from generativeqc_compiler.common import cpp_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid JVP reached compiler discovery")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)

    def function(x: Any, y: Any, mask: Any) -> Any:
        return xp.where(mask, x * x, y * y)

    primals = (np.ones(2), np.ones(2), np.ones(2, dtype=bool))
    directions = (np.ones(2), np.ones(2), None)
    with pytest.raises((TypeError, ValueError)):
        xp.jvp(function, primals, directions)
    for flag in (1, None, "true", np.bool_(True)):
        with pytest.raises(TypeError, match="must be Boolean"):
            xp.jvp(function, primals, directions, allow_external_mask_where=flag)
    with pytest.raises(ValueError, match="reference or CPU"):
        xp.jvp(
            function,
            primals,
            directions,
            backend="cuda",
            allow_external_mask_where=True,
        )
    with pytest.raises(ValueError, match="at least one"):
        xp.jvp(function, primals, (None, None, None), allow_external_mask_where=True)
    for invalid in (
        (np.ones(3), np.ones(2), None),
        (np.ones(2, dtype=np.float32), np.ones(2), None),
        (np.ones(2), np.ones(2), np.zeros(2, dtype=bool)),
        (np.ones(2), np.ones(2)),
    ):
        with pytest.raises((TypeError, ValueError)):
            xp.jvp(function, primals, invalid, allow_external_mask_where=True)
    with pytest.raises(ValueError, match="external Boolean mask"):
        xp.jvp(
            lambda x: xp.where(x > 0, x, x * x),
            (np.ones(2),),
            (np.ones(2),),
            allow_external_mask_where=True,
        )
    for options in ({"max_bytes": 1}, {"max_work": 0}, {"max_nodes": 1}):
        source = xp.compile(function, cache=tmp_path, **options)
        try:
            with pytest.raises(ValueError, match="budget"):
                xp.jvp(source, primals, directions, allow_external_mask_where=True)
            assert not source._execution.closed
        finally:
            source.close()


@pytest.mark.parametrize("bad", ("primal", "direction", "dtype"))
def test_disconnected_direction_and_primal_validation(backend: str, bad: str) -> None:
    def function(x: Any, unused: Any, mask: Any) -> Any:
        return xp.where(mask, x * x, x)

    primals = (np.ones(2), np.ones(3), np.ones(2, dtype=bool))
    tangents = (None, np.ones(3), None)
    if bad == "primal":
        primals = (primals[0], np.full(3, np.inf), primals[2])
    elif bad == "direction":
        tangents = (None, np.full(3, np.nan), None)
    else:
        tangents = (None, np.ones(3, dtype=np.float32), None)
    with pytest.raises(ValueError):
        xp.jvp(
            function, primals, tangents, backend=backend, allow_external_mask_where=True
        )


def test_serialized_combination_preserves_typed_first_order_boundary(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc.experimental._array_api_execution import _Executor
    from generativeqc_compiler.tensor import Program, linearize

    original = _Executor.execute

    def checked(self: Any, program: Any, feeds: Any) -> Any:
        assert program.provenance.get("external_mask_where_jvp")
        replay = Program.loads(program.dumps())
        assert all(
            not node.spec.differentiable for node in replay.nodes if node.op == "input"
        )
        stripped = Program(replay.outputs, replay.nodes)
        with pytest.raises(ValueError):
            linearize(stripped, ("x",), allow_external_mask_where=True)
        return original(self, replay, feeds)

    monkeypatch.setattr(_Executor, "execute", checked)
    primal, tangent = xp.jvp(
        lambda x, mask: xp.sum(xp.where(mask, x * x, x)),
        (np.asarray([0.2, 0.7]), np.asarray([True, False])),
        (np.ones(2), None),
        backend=backend,
        cache=tmp_path,
        allow_external_mask_where=True,
    )
    np.testing.assert_allclose(primal, 0.74, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(tangent, 1.4, rtol=2e-13, atol=2e-13)


def test_compiler_failure_releases_temporary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc.experimental._array_api_execution import _Executor

    source = xp.compile(lambda x, mask: xp.where(mask, x * x, x))
    owners = []

    def fail(self: Any, program: Any) -> Any:
        owners.append(self)
        raise RuntimeError("injected JVP compiler failure")

    monkeypatch.setattr(_Executor, "_build", fail)
    try:
        with pytest.raises(RuntimeError, match="injected JVP compiler failure"):
            xp.jvp(
                source,
                (np.ones(2), np.ones(2, dtype=bool)),
                (np.ones(2), None),
                allow_external_mask_where=True,
            )
        assert owners and all(owner.closed for owner in owners)
        assert not source._execution.closed
    finally:
        source.close()


def test_capability_separates_explicit_none_direction_contract() -> None:
    capability = xp.capabilities()["external_mask_jvp"]
    assert capability["default"] is False
    assert capability["backends"] == ("reference", "cpu")
    assert capability["real_dtype"] == "float64"
    assert (
        capability["non_differentiated_primal"]
        == "explicit-None-tangent-in-matching-tuple"
    )

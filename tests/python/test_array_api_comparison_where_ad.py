"""Real-branch derivatives with primal comparison selectors, without Boolean AD."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.tensor import autodiff
from generativeqc_compiler.tensor.ad_program import linearize, transpose_program
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.optimize import optimize
from generativeqc_compiler.tensor.program import Program

COMPARISONS = ("equal", "not_equal", "greater", "greater_equal", "less", "less_equal")


@pytest.fixture(params=("reference", "cpu"))
def backend(request: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "cpu":
        from generativeqc_compiler.tensor import interpreter

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("native selected-branch AD used a reference oracle")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(autodiff, "jvp", forbidden)
        monkeypatch.setattr(autodiff, "vjp", forbidden)
    return request.param


def capture(function: Any, feeds: dict) -> Program:
    names = tuple(name for name, value in feeds.items() if value.dtype.kind == "f")
    source = xp.compile(function, backend="reference", differentiable=names)
    try:
        return source.lower(*feeds.values())
    finally:
        source.close()


@pytest.mark.parametrize("comparison", COMPARISONS)
def test_public_nonlinear_broadcast_scatter_fd_dot_snapshot(
    backend: str, comparison: str, tmp_path: Path
) -> None:
    order = [3, 0, 3, 1]
    x = np.array([[-1.4], [-0.6], [0.5], [1.8]])
    y = np.array([0.7, 1.1])
    threshold = np.array([0.0, 1.2])
    dx = np.array([[0.1], [-0.2], [0.3], [0.2]])
    dy = np.array([0.3, -0.1])
    dt = np.array([0.4, 0.2])
    weight = np.array([[0.2, -0.3], [0.5, 0.1], [-0.2, 0.4], [0.3, 0.2]])
    compare = getattr(np, comparison)

    def function(x: Any, y: Any, threshold: Any) -> dict:
        a = xp.take(x, tuple(order), axis=0)
        mask = getattr(xp, comparison)(a, threshold)
        selected = xp.where(mask, a * a, y * y)
        return {"vector": selected, "norm": xp.einsum("ij,ij->", selected, selected)}

    def oracle(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> tuple:
        selected = np.where(compare(a[order], t), a[order] ** 2, b**2)
        return selected, np.sum(selected**2)

    source = xp.compile(function, backend=backend, cache=tmp_path)
    try:
        primal, tangent = xp.jvp(
            source, (x, y, threshold), (dx, dy, dt), allow_comparison_where=True
        )
        selected, norm = oracle(x, y, threshold)
        mask = compare(x[order], threshold)
        expected = np.where(mask, 2 * x[order] * dx[order], 2 * y * dy)
        expected_norm = 2 * np.sum(selected * expected)
        for actual, wanted in (
            (primal["vector"], selected),
            (primal["norm"], norm),
            (tangent["vector"], expected),
            (tangent["norm"], expected_norm),
        ):
            np.testing.assert_allclose(actual, wanted, rtol=2e-13, atol=2e-13)
        step = 1e-5
        plus = oracle(x + step * dx, y + step * dy, threshold + step * dt)
        minus = oracle(x - step * dx, y - step * dy, threshold - step * dt)
        for actual, a, b in zip(
            (tangent["vector"], tangent["norm"]), plus, minus, strict=True
        ):
            np.testing.assert_allclose(
                actual, (a - b) / (2 * step), rtol=2e-8, atol=2e-8
            )
        _, pullback = xp.vjp(
            source, x, y, threshold, argnums=(0, 1, 2), allow_comparison_where=True
        )
        try:
            total_weight = weight + 0.6 * selected
            bar_x = np.zeros_like(x)
            np.add.at(
                bar_x,
                order,
                np.where(mask, 2 * x[order] * total_weight, 0).sum(
                    axis=1, keepdims=True
                ),
            )
            bar_y = np.where(mask, 0, 2 * y * total_weight).sum(axis=0)
            # Predicate and real branch primals both belong to the immutable VJP snapshot.
            x[:] = 8
            y[:] = 9
            threshold[:] = 10
            bx, by, bt = pullback({"vector": weight, "norm": np.array(0.3)})
            for actual, wanted in (
                (bx, bar_x),
                (by, bar_y),
                (bt, np.zeros_like(threshold)),
            ):
                np.testing.assert_allclose(actual, wanted, rtol=2e-13, atol=2e-13)
            lhs = np.sum(weight * tangent["vector"]) + 0.3 * tangent["norm"]
            rhs = np.sum(bx * dx) + np.sum(by * dy) + np.sum(bt * dt)
            np.testing.assert_allclose(lhs, rhs, rtol=2e-13, atol=2e-13)
            assert not np.shares_memory(bx, x)
        finally:
            pullback.close()
        assert not source._execution.closed
    finally:
        source.close()


@pytest.mark.parametrize("comparison", COMPARISONS)
def test_selected_branch_convention_at_switching_points(
    backend: str, comparison: str, tmp_path: Path
) -> None:
    # Equality is a branch convention, not a claim of differentiability there.
    x = np.array([-0.0, 0.0, 1.0])
    y = x.copy()
    function = lambda a, b: xp.where(getattr(xp, comparison)(a, b), a * a, 3 * b)
    mask = getattr(np, comparison)(x, y)
    _, tangent = xp.jvp(
        function,
        (x, y),
        (np.ones(3), np.ones(3)),
        backend=backend,
        cache=tmp_path,
        allow_comparison_where=True,
    )
    np.testing.assert_array_equal(tangent, np.where(mask, 2 * x, 3))


@pytest.mark.parametrize("shape", ((), (0, 2), (2, 3)))
def test_predicate_only_singular_slope_is_not_differentiated(
    backend: str, shape: tuple, tmp_path: Path
) -> None:
    def function(x: Any, p: Any) -> Any:
        return xp.where(xp.sqrt(p) > 0, x * x, x)

    x, p = np.ones(shape), np.array(0.0)
    _, tangent = xp.jvp(
        function,
        (x, p),
        (np.full(shape, 0.2), np.array(1.0)),
        backend=backend,
        cache=tmp_path,
        allow_comparison_where=True,
    )
    np.testing.assert_array_equal(tangent, np.full(shape, 0.2))
    _, pullback = xp.vjp(
        function, x, p, backend=backend, cache=tmp_path, allow_comparison_where=True
    )
    try:
        bx, bp = pullback(np.ones(shape))
        np.testing.assert_array_equal(bx, np.ones(shape))
        np.testing.assert_array_equal(bp, 0.0)
    finally:
        pullback.close()
    with pytest.raises(ValueError, match="domain|non-finite|failed"):
        xp.jvp(
            function,
            (x, np.array(-1.0)),
            (np.ones(shape), np.array(1.0)),
            backend=backend,
            cache=tmp_path,
            allow_comparison_where=True,
        )


@pytest.mark.parametrize("transform", ("grad", "value_and_grad"))
def test_scalar_transforms_reuse_selected_branch_rules(
    backend: str, transform: str, tmp_path: Path
) -> None:
    x = np.array([-1.0, 0.5, 2.0])
    function = lambda a: xp.sum(xp.where(a > 0, a * a, 3 * a))
    derivative = getattr(xp, transform)(
        function, backend=backend, cache=tmp_path, allow_comparison_where=True
    )
    try:
        result = derivative(x)
        gradient = result[1] if transform == "value_and_grad" else result
        np.testing.assert_array_equal(gradient, [3.0, 1.0, 4.0])
        if transform == "value_and_grad":
            np.testing.assert_allclose(result[0], 1.25, rtol=2e-13, atol=2e-13)
    finally:
        derivative.close()


def test_invalid_options_domains_and_legacy_gates_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.common import cpp_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid selected-branch AD reached compiler discovery")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)
    function = lambda x: xp.where(x > 0, x * x, x)
    for flag in (1, None, "true", np.bool_(True)):
        with pytest.raises(TypeError, match="must be Boolean"):
            xp.jvp(function, (np.ones(2),), (np.ones(2),), allow_comparison_where=flag)
        with pytest.raises(TypeError, match="must be Boolean"):
            xp.vjp(function, np.ones(2), allow_comparison_where=flag)
        with pytest.raises(TypeError, match="must be Boolean"):
            xp.grad(function, allow_comparison_where=flag)
    for options in ({}, {"allow_external_mask_where": True}):
        with pytest.raises(ValueError, match="Boolean"):
            xp.jvp(function, (np.ones(2),), (np.ones(2),), **options)
    with pytest.raises(ValueError, match="reference or CPU"):
        xp.jvp(
            function,
            (np.ones(2),),
            (np.ones(2),),
            backend="cuda",
            allow_comparison_where=True,
        )
    with pytest.raises(ValueError, match="float64"):
        xp.jvp(
            function,
            (np.ones(2, dtype=np.float32),),
            (np.ones(2, dtype=np.float32),),
            allow_comparison_where=True,
        )
    with pytest.raises(ValueError, match="at least one"):
        xp.jvp(function, (np.ones(2),), (None,), allow_comparison_where=True)
    with pytest.raises(ValueError, match="Boolean"):
        xp.jvp(
            lambda x: x > 0, (np.ones(2),), (np.ones(2),), allow_comparison_where=True
        )


def test_failed_compilation_closes_temporary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc.experimental._array_api_execution import _Executor

    source = xp.compile(lambda x: xp.where(x > 0, x * x, x))
    owners = []

    def fail(self: Any, program: Any) -> Any:
        owners.append(self)
        raise RuntimeError("injected comparison-where compiler failure")

    monkeypatch.setattr(_Executor, "_build", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            xp.jvp(source, (np.ones(2),), (np.ones(2),), allow_comparison_where=True)
        assert owners and all(owner.closed for owner in owners)
        assert not source._execution.closed
    finally:
        source.close()


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("comparison", COMPARISONS)
def test_reference_analytic_finite_difference_and_adjoint(
    dtype: Any, comparison: str
) -> None:
    feeds = {
        "x": np.array([-1.4, 0.5, 1.8], dtype=dtype),
        "y": np.array(0.7, dtype=dtype),
        "t": np.array(0.0, dtype=dtype),
    }
    directions = {name: np.full_like(value, 0.2) for name, value in feeds.items()}
    function = lambda x, y, t: xp.where(getattr(xp, comparison)(x, t), x * x, y * y)
    program = capture(function, feeds)
    mask = getattr(np, comparison)(feeds["x"], feeds["t"])
    expected = np.where(
        mask, 2 * feeds["x"] * directions["x"], 2 * feeds["y"] * directions["y"]
    )
    weights = np.array([0.2, -0.3, 0.5], dtype=dtype)
    forward = autodiff.jvp(program, feeds, directions, allow_comparison_where=True)
    reverse = autodiff.vjp(
        program, feeds, {"output": weights}, allow_comparison_where=True
    )
    atol = 2e-6 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(
        forward.output_tangents["output"], expected, rtol=atol, atol=atol
    )
    np.testing.assert_allclose(
        reverse.input_cotangents["x"],
        np.where(mask, 2 * feeds["x"] * weights, 0),
        rtol=atol,
        atol=atol,
    )
    np.testing.assert_allclose(
        reverse.input_cotangents["y"],
        np.sum(np.where(mask, 0, 2 * feeds["y"] * weights)),
        rtol=atol,
        atol=atol,
    )
    np.testing.assert_array_equal(reverse.input_cotangents["t"], 0)
    assert autodiff.dot_test(
        program,
        feeds,
        directions,
        {"output": weights},
        rtol=atol,
        atol=atol,
        allow_comparison_where=True,
    ).passed
    step = 2e-3 if dtype == np.float32 else 1e-5
    plus = {name: value + step * directions[name] for name, value in feeds.items()}
    minus = {name: value - step * directions[name] for name, value in feeds.items()}
    oracle = lambda f: np.where(
        getattr(np, comparison)(f["x"], f["t"]), f["x"] ** 2, f["y"] ** 2
    )
    finite_difference = (oracle(plus) - oracle(minus)) / (2 * step)
    np.testing.assert_allclose(
        expected,
        finite_difference,
        rtol=2e-3 if dtype == np.float32 else 2e-8,
        atol=2e-4 if dtype == np.float32 else 2e-8,
    )


def test_serialized_generated_predicate_boundary_and_requested_feeds() -> None:
    feeds = {"x": np.array([-1.0, 2.0]), "p": np.array(0.0)}
    program = capture(lambda x, p: xp.where(xp.sqrt(p) > 0, x * x, x), feeds)
    forward = linearize(program, ("x", "p"), allow_comparison_where=True).program
    reverse = transpose_program(
        program, ("output",), inputs=("x", "p"), allow_comparison_where=True
    ).program
    for graph in (forward, optimize(forward), Program.loads(forward.dumps())):
        valid = {**feeds, "d_x": np.ones(2), "d_p": np.array(0.3)}
        np.testing.assert_array_equal(execute(graph, valid).outputs["d_output"], 1)
        for bad in (np.array(np.nan), np.array(1.0, dtype=np.float32), np.ones(2)):
            with pytest.raises(ValueError):
                execute(graph, {**valid, "d_p": bad})
        for replay in (graph, Program(graph.outputs, graph.definitions)):
            with pytest.raises(ValueError):
                linearize(replay, ("x",), allow_comparison_where=True)
            with pytest.raises(ValueError):
                transpose_program(
                    replay,
                    tuple(replay.outputs),
                    inputs=("x",),
                    allow_comparison_where=True,
                )
    for graph in (reverse, optimize(reverse), Program.loads(reverse.dumps())):
        outputs = execute(graph, {**feeds, "bar_output": np.array([0.2, 0.3])}).outputs
        np.testing.assert_array_equal(outputs["bar_x"], [0.2, 0.3])
        np.testing.assert_array_equal(outputs["bar_p"], 0)
        assert all(not n.spec.differentiable for n in graph.nodes if n.op == "input")


@pytest.mark.parametrize("transform", ("jvp", "vjp"))
def test_unselected_branch_derivative_domains_remain_checked(
    backend: str, transform: str, tmp_path: Path
) -> None:
    function = lambda x: xp.where(x >= 0, x, xp.sqrt(x))
    x = np.zeros(2)
    with pytest.raises(ValueError, match="zero|domain|non-finite|failed"):
        if transform == "jvp":
            xp.jvp(
                function,
                (x,),
                (np.ones(2),),
                backend=backend,
                cache=tmp_path,
                allow_comparison_where=True,
            )
        else:
            _, pullback = xp.vjp(
                function,
                x,
                backend=backend,
                cache=tmp_path,
                allow_comparison_where=True,
            )
            try:
                pullback(np.ones(2))
            finally:
                pullback.close()


def test_comparison_ad_capabilities_are_explicit() -> None:
    public = xp.capabilities()["comparison_where_ad"]
    assert public["option"] == "allow_comparison_where" and public["default"] is False
    assert public["backends"] == ("reference", "cpu")
    assert public["real_dtype"] == "float64"
    assert autodiff.capabilities()["reference_comparison_where"]["default"] is False


@pytest.mark.parametrize("mode", ("jvp", "vjp"))
def test_resource_budgets_reject_before_toolchain(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from generativeqc_compiler.common import cpp_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("resource rejection reached compiler discovery")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", forbidden)
    for options in ({"max_bytes": 1}, {"max_work": 0}, {"max_nodes": 1}):
        source = xp.compile(
            lambda x: xp.where(x > 0, x * x, x), cache=tmp_path, **options
        )
        try:
            with pytest.raises(ValueError, match="budget"):
                if mode == "jvp":
                    xp.jvp(
                        source,
                        (np.ones(2),),
                        (np.ones(2),),
                        allow_comparison_where=True,
                    )
                else:
                    xp.vjp(source, np.ones(2), allow_comparison_where=True)
            assert not source._execution.closed
        finally:
            source.close()


def test_reference_predicate_only_slopes_and_boolean_budgets() -> None:
    feeds = {"x": np.array([-1.0, 2.0]), "p": np.array(0.0)}
    program = capture(lambda x, p: xp.where(xp.sqrt(p) > 0, x * x, x), feeds)
    tangents = {"x": np.ones(2), "p": np.array(1.0)}
    weights = {"output": np.array([0.2, 0.3])}
    forward = autodiff.jvp(program, feeds, tangents, allow_comparison_where=True)
    reverse = autodiff.vjp(program, feeds, weights, allow_comparison_where=True)
    np.testing.assert_array_equal(forward.output_tangents["output"], 1)
    np.testing.assert_array_equal(reverse.input_cotangents["p"], 0)
    assert autodiff.dot_test(
        program, feeds, tangents, weights, allow_comparison_where=True
    ).passed
    with pytest.raises(ValueError, match="budget"):
        autodiff.jvp(program, feeds, tangents, max_bytes=1, allow_comparison_where=True)
    for graph in (program, optimize(program), Program.loads(program.dumps())):
        with pytest.raises(ValueError, match="external Boolean mask"):
            autodiff.jvp(graph, feeds, tangents)


def test_typed_comparison_views_nested_with_external_selector(
    backend: str, tmp_path: Path
) -> None:
    def function(x: Any, y: Any, t: Any, external: Any) -> Any:
        viewed = xp.flip(x > t, axis=0)
        inner = xp.where(viewed, x * x, 3 * x)
        return xp.where(external, inner, y)

    x, y, t = np.array([-1.0, 0.5, 2.0]), np.array(0.7), np.array(0.0)
    external = np.array([True, False, True])
    _, tangent = xp.jvp(
        function,
        (x, y, t, external),
        (np.ones(3), np.array(0.2), np.array(0.3), None),
        backend=backend,
        cache=tmp_path,
        allow_comparison_where=True,
    )
    mask = (x > t)[::-1]
    np.testing.assert_array_equal(
        tangent, np.where(external, np.where(mask, 2 * x, 3), 0.2)
    )
    _, pullback = xp.vjp(
        function,
        x,
        y,
        t,
        external,
        argnums=(0, 1, 2),
        backend=backend,
        cache=tmp_path,
        allow_comparison_where=True,
    )
    try:
        bx, by, bt = pullback(np.ones(3))
        np.testing.assert_array_equal(
            bx, np.where(external, np.where(mask, 2 * x, 3), 0)
        )
        np.testing.assert_array_equal(by, 1)
        np.testing.assert_array_equal(bt, 0)
    finally:
        pullback.close()


@pytest.mark.parametrize("domain", ("float32", "scientific"))
def test_disconnected_requested_input_domain_remains_checked(domain: str) -> None:
    from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

    indices = (Index("a", IndexSpace("ao", "ao", 2)),) if domain == "scientific" else ()
    spec = TensorSpec(
        indices,
        dtype="float64" if domain == "scientific" else "float32",
        role="parameter",
        differentiable=True,
    )
    source = xp.compile(
        lambda x, z: xp.where(x > 0, x * x, x), backend="reference", specs={"z": spec}
    )
    x = np.array([-1.0, 2.0])
    z = np.ones(spec.shape, dtype=spec.dtype)
    try:
        with pytest.raises(ValueError, match="generic|float64"):
            xp.jvp(source, (x, z), (None, np.ones_like(z)), allow_comparison_where=True)
        with pytest.raises(ValueError, match="generic|float64"):
            xp.vjp(source, x, z, argnums=(1,), allow_comparison_where=True)
    finally:
        source.close()


def test_pure_real_derivative_identity_is_unchanged() -> None:
    program = capture(lambda x: x * x, {"x": np.array([-1.0, 2.0])})
    for mode in ("jvp", "vjp"):
        if mode == "jvp":
            ordinary = linearize(program, ("x",)).program
            opted = linearize(program, ("x",), allow_comparison_where=True).program
        else:
            ordinary = transpose_program(program, ("output",)).program
            opted = transpose_program(
                program, ("output",), allow_comparison_where=True
            ).program
        assert ordinary.dumps() == opted.dumps()

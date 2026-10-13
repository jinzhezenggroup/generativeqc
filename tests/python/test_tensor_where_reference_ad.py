"""Independent fixed-mask reference derivatives; no conditional/native AD claim."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api import namespace as ns
from generativeqc_compiler.array_api.array import VibeArray
from generativeqc_compiler.tensor import Program, dot_test, execute, ir, jvp, vjp
from generativeqc_compiler.tensor.ad_program import linearize, transpose_program
from generativeqc_compiler.tensor.autodiff import AD_PRIMITIVES, capabilities


def capture(function: Any, feeds: dict, names: tuple = ("x", "y")) -> Program:
    return xp.compile(function, backend="reference", differentiable=names).lower(
        *feeds.values()
    )


def check(
    program: Program,
    feeds: dict,
    directions: dict,
    weights: dict,
    expected: dict,
    slopes: dict,
    bars: dict,
) -> None:
    """Check actual full reference endpoints against independently supplied math."""
    dtype = next(iter(expected.values())).dtype
    rtol, atol, step = (
        (3e-6, 2e-6, 2e-3) if dtype == np.float32 else (2e-12, 2e-12, 1e-5)
    )
    saved = {name: value.copy() for name, value in feeds.items()}
    primal = execute(program, feeds).outputs
    forward = jvp(program, feeds, directions)
    reverse = vjp(program, feeds, weights)
    for name, expected_value in expected.items():
        np.testing.assert_allclose(primal[name], expected_value, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            forward.output_tangents[name], slopes[name], rtol=rtol, atol=atol
        )
        plus, minus = dict(feeds), dict(feeds)
        for key, direction in directions.items():
            plus[key], minus[key] = (
                feeds[key] + step * direction,
                feeds[key] - step * direction,
            )
        # The mask is held fixed; this does not qualify comparison switching points.
        finite_difference = (
            execute(program, plus).outputs[name] - execute(program, minus).outputs[name]
        ) / (2 * step)
        np.testing.assert_allclose(
            forward.output_tangents[name],
            finite_difference,
            rtol=2e-3 if dtype == np.float32 else 2e-7,
            atol=2e-4 if dtype == np.float32 else 2e-9,
        )
    for name, expected_bar in bars.items():
        np.testing.assert_allclose(
            reverse.input_cotangents[name], expected_bar, rtol=rtol, atol=atol
        )
    inner = lambda a, b: float(
        np.sum(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))
    )
    lhs = sum(inner(weights[name], slopes[name]) for name in weights)
    rhs = sum(inner(bars[name], directions[name]) for name in directions)
    result = dot_test(program, feeds, directions, weights, rtol=rtol, atol=atol)
    assert result.passed, result
    np.testing.assert_allclose(
        [result.lhs, result.rhs], [lhs, rhs], rtol=rtol, atol=atol
    )
    for name, value in feeds.items():
        np.testing.assert_array_equal(value, saved[name])
        for output in (
            *primal.values(),
            *forward.output_tangents.values(),
            *reverse.input_cotangents.values(),
        ):
            assert not np.shares_memory(output, value)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["true", "false", "sparse", "dense"])
def test_polynomial_exp_log_broadcast_oracles(dtype: Any, mode: str) -> None:
    rng = np.random.default_rng(2137)
    x = np.linspace(0.4, 1.1, 6, dtype=dtype).reshape(2, 1, 3)
    y = np.linspace(0.8, 1.3, 4, dtype=dtype).reshape(1, 4, 1)
    mask = (
        np.full((2, 4, 3), mode == "true")
        if mode in ("true", "false")
        else rng.random((2, 4, 3)) < (0.2 if mode == "sparse" else 0.8)
    )
    feeds = {"mask": mask, "x": x, "y": y}
    program = capture(
        lambda mask, x, y: ns.where(mask, x * x + ns.exp(x), ns.log(y)), feeds
    )
    dx, dy = np.full_like(x, 0.3), np.full_like(y, -0.2)
    w = rng.normal(size=mask.shape).astype(dtype)
    left_slope = 2 * x + np.exp(x)
    check(
        program,
        feeds,
        {"x": dx, "y": dy},
        {"output": w},
        {"output": np.where(mask, x * x + np.exp(x), np.log(y))},
        {"output": np.where(mask, left_slope * dx, dy / y)},
        {
            "x": np.sum(np.where(mask, w, 0), axis=1, keepdims=True) * left_slope,
            "y": np.sum(np.where(mask, 0, w), axis=(0, 2), keepdims=True) / y,
        },
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_scalar_empty_noncontiguous_nested_replay(dtype: Any, shape: tuple) -> None:
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape) / dtype(10) + dtype(0.5)
    if shape == (2, 3):
        x = np.ascontiguousarray(x.T).T
        assert not x.flags.c_contiguous
    y = x.copy() + dtype(0.3)
    mask = np.ones(shape, dtype=bool)
    feeds = {"mask": mask, "x": x, "y": y}
    program = capture(
        lambda mask, x, y: ns.where(mask, ns.where(False, y, x * x), y * y), feeds
    )
    replay = Program.loads(program.dumps())
    assert replay.dumps() == program.dumps()
    for predicate in (mask, ~mask):
        feeds["mask"] = predicate
        dx, dy, w = (
            np.ones_like(x) * dtype(0.2),
            np.ones_like(y) * dtype(-0.1),
            np.ones_like(x),
        )
        check(
            replay,
            feeds,
            {"x": dx, "y": dy},
            {"output": w},
            {"output": np.where(predicate, x * x, y * y)},
            {"output": np.where(predicate, 2 * x * dx, 2 * y * dy)},
            {"x": np.where(predicate, 2 * x, 0), "y": np.where(predicate, 0, 2 * y)},
        )
    assert (
        jvp(program, feeds, {"x": dx}).derivative_hash
        == jvp(replay, feeds, {"x": dx}).derivative_hash
    )


def test_mask_view_chain_and_real_scatter_adjoint() -> None:
    feeds = {
        "mask": np.array([[True, False], [False, True]]),
        "x": np.array([0.5, 0.7, 0.9]),
        "y": np.array([1.0, 2.0, 3.0]),
    }

    def function(mask: Any, x: Any, y: Any) -> Any:
        m = ns.permute_dims(mask, (1, 0))
        m = ns.reshape(m, (4,))
        m = VibeArray(ir.slice_tensor(m.node, ((1, 4),)))
        m = VibeArray(ir.gather(m.node, 0, (2, 0, 2)))
        m = ns.broadcast_to(m, (2, 3))
        gathered = VibeArray(ir.gather(x.node, 0, (2, 0, 2)))
        return ns.where(m, gathered * gathered, y)

    program = capture(function, feeds)
    mask = np.broadcast_to(feeds["mask"].T.reshape(4)[1:4][[2, 0, 2]], (2, 3))
    positions = np.array([2, 0, 2])
    xg = feeds["x"][positions]
    dx, dy = np.array([0.2, -0.3, 0.4]), np.array([-0.1, 0.3, 0.5])
    w = np.arange(6, dtype=np.float64).reshape(2, 3) + 1
    bx = np.zeros(3)
    np.add.at(bx, positions, np.sum(np.where(mask, w, 0), axis=0) * 2 * xg)
    check(
        program,
        feeds,
        {"x": dx, "y": dy},
        {"output": w},
        {"output": np.where(mask, xg * xg, feeds["y"])},
        {"output": np.where(mask, 2 * xg * dx[positions], dy)},
        {"x": bx, "y": np.sum(np.where(mask, 0, w), axis=0)},
    )


def test_promotions_partial_selection_alias_and_budget() -> None:
    feeds = {
        "mask": np.array([True, False]),
        "x": np.array([1.0, 2.0], dtype=np.float32),
        "y": np.array([3.0, 4.0]),
    }
    program = capture(
        lambda mask, x, y: {"selected": ns.where(mask, x * x, y * y), "real": y * y},
        feeds,
    )
    forward = jvp(
        program, feeds, {"x": np.ones(2, dtype=np.float32)}, outputs=["selected"]
    )
    np.testing.assert_array_equal(forward.output_tangents["selected"], [2.0, 0.0])
    assert forward.output_tangents["selected"].dtype == np.float64
    reverse = vjp(program, feeds, {"selected": np.ones(2)}, inputs=["x"])
    np.testing.assert_array_equal(reverse.input_cotangents["x"], [2.0, 0.0])
    assert reverse.input_cotangents["x"].dtype == np.float32
    for endpoint in (
        lambda budget: jvp(
            program, feeds, {"x": np.ones(2, dtype=np.float32)}, max_bytes=budget
        ),
        lambda budget: vjp(program, feeds, {"selected": np.ones(2)}, max_bytes=budget),
        lambda budget: dot_test(
            program,
            feeds,
            {"x": np.ones(2, dtype=np.float32)},
            {"selected": np.ones(2)},
            max_bytes=budget,
        ),
    ):
        with pytest.raises(ValueError, match="budget"):
            endpoint(1)
        endpoint(10000)
    shared = np.array([1.0, 2.0])
    alias = {"mask": feeds["mask"], "x": shared, "y": shared}
    p = capture(lambda mask, x, y: ns.where(mask, x, y), alias)
    check(
        p,
        alias,
        {"x": shared, "y": shared},
        {"output": shared},
        {"output": shared},
        {"output": shared},
        {
            "x": np.where(alias["mask"], shared, 0),
            "y": np.where(alias["mask"], 0, shared),
        },
    )


@pytest.mark.parametrize(
    "bad",
    [
        "tangent_mask",
        "bar_bool",
        "wrt_mask",
        "bool_output",
        "comparison",
        "unrelated_bool",
    ],
)
def test_non_differentiable_requests_fail_closed(bad: str) -> None:
    feeds = {"mask": np.array([True, False]), "x": np.ones(2), "y": np.ones(2)}
    function = lambda mask, x, y: ns.where(mask, x, y)
    if bad == "bool_output":
        function = lambda mask, x, y: {"real": ns.where(mask, x, y), "bool": mask}
    if bad == "comparison":
        function = lambda mask, x, y: ns.where(x > 0, x, y)
    if bad == "unrelated_bool":
        function = lambda mask, x, y: x > y
    program = capture(function, feeds)
    if bad == "tangent_mask":
        action = lambda: jvp(program, feeds, {"mask": feeds["mask"]})
    elif bad == "bar_bool":
        action = lambda: vjp(program, feeds, {"output": feeds["mask"]})
    elif bad == "wrt_mask":
        action = lambda: vjp(program, feeds, {"output": np.ones(2)}, inputs=["mask"])
    else:
        action = lambda: jvp(program, feeds, {"x": np.ones(2)})
    with pytest.raises(ValueError, match="Boolean|differentiable|dtype|input"):
        action()


@pytest.mark.parametrize("endpoint", ["jvp", "vjp", "dot"])
@pytest.mark.parametrize("failure", ["nan", "log", "divide", "singular", "overflow"])
def test_unselected_errors_are_retained_and_recover(
    endpoint: str, failure: str
) -> None:
    funcs = {
        "nan": lambda x, y: y,
        "log": lambda x, y: ns.log(y),
        "divide": lambda x, y: x / y,
        "singular": lambda x, y: ns.sqrt(y),
        "overflow": lambda x, y: ns.exp(y),
    }
    feeds = {
        "x": np.ones(2),
        "y": np.full(
            2,
            {
                "nan": np.nan,
                "log": -1.0,
                "divide": 0.0,
                "singular": 0.0,
                "overflow": 1000.0,
            }[failure],
        ),
    }
    program = capture(
        lambda x, y: ns.where(True, x, funcs[failure](x, y)),
        {"x": np.ones(2), "y": np.ones(2)},
    )
    actions = {
        "jvp": lambda: jvp(program, feeds, {"y": np.ones(2)}),
        "vjp": lambda: vjp(program, feeds, {"output": np.ones(2)}),
        "dot": lambda: dot_test(
            program, feeds, {"y": np.ones(2)}, {"output": np.ones(2)}
        ),
    }
    with pytest.raises(ValueError, match="finite|positive|zero|derivative|domain"):
        actions[endpoint]()
    feeds["y"] = np.ones(2)
    actions[endpoint]()


def test_generated_and_public_ad_remain_rejected_before_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.tensor import cpu, cuda_execute

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("where AD reached native compiler/device allocation")

    monkeypatch.setattr(cpu, "CppCompilerAdapter", forbidden)
    monkeypatch.setattr(cuda_execute, "PreparedCuda", forbidden)
    values = np.ones(2)
    program = capture(lambda x, y: ns.where(True, x, y), {"x": values, "y": values})
    for action in (
        lambda: linearize(program, ["x"]),
        lambda: transpose_program(program, ["output"]),
    ):
        with pytest.raises(ValueError, match="Boolean|differentiable"):
            action()
    for backend in ("reference", "cpu", "cuda"):

        def energy(x: Any) -> Any:
            return xp.sum(xp.where(True, x, 0))

        for action in (
            lambda backend=backend: xp.grad(energy, backend=backend)(values),
            lambda backend=backend: xp.value_and_grad(energy, backend=backend)(values),
            lambda backend=backend: xp.jvp(
                energy, (values,), (values,), backend=backend
            ),
            lambda backend=backend: xp.vjp(energy, values, backend=backend),
        ):
            with pytest.raises(ValueError, match="Boolean|differentiable"):
                action()
    assert "where" not in AD_PRIMITIVES
    assert (
        capabilities()["reference_external_mask_where"]["generated_or_native"] is False
    )


def test_malformed_mask_views_and_scientific_ancestors_reject() -> None:
    feeds = {"mask": np.ones(2, dtype=bool), "x": np.ones(2), "y": np.ones(2)}
    p = capture(lambda mask, x, y: ns.where(mask, x, y), feeds)
    mask = next(n for n in p.live_nodes if n.op == "input" and n.spec.dtype == "bool")
    with pytest.raises(ValueError, match="shape|elements|reshape"):
        ir.reshape(mask, ())
    with pytest.raises(ValueError, match="axes|axis|permutation"):
        ir.transpose(mask, (0, 0))
    with pytest.raises(ValueError, match="range|position|bounds"):
        ir.gather(mask, 0, (9,))
    feeds["mask"] = np.ones(2, dtype=np.int64)
    with pytest.raises(ValueError, match="dtype"):
        jvp(p, feeds, {"x": np.ones(2)})
    # Even a separate scientific output is outside the qualified mixed graph.
    x = next(n for n in p.live_nodes if n.op == "input" and n.attrs["name"] == "x")
    scientific = ir.input_tensor("s", replace(x.spec, representation="spin_orbital"))
    mixed = Program({"output": p.outputs["output"], "scientific": scientific})
    with pytest.raises(ValueError, match="generic"):
        jvp(mixed, {}, {})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_explicit_cast_and_duplicate_parameter_consumers(dtype: Any) -> None:
    feeds = {
        "mask": np.array([True, False]),
        "x": np.array([0.5, 0.75]),
        "y": np.array([0.8, 1.2]),
    }
    program = capture(
        lambda mask, x, y: ns.where(
            mask, ns.astype(x, dtype.__name__), ns.astype(y, dtype.__name__)
        ),
        feeds,
    )
    direction = np.array([0.2, -0.3])
    weight = np.array([2.0, 3.0], dtype=dtype)
    result = jvp(program, feeds, {"x": direction})
    np.testing.assert_array_equal(
        result.output_tangents["output"],
        np.where(feeds["mask"], direction.astype(dtype), dtype(0)),
    )
    bars = vjp(program, feeds, {"output": weight}).input_cotangents
    np.testing.assert_array_equal(bars["x"], np.array([2.0, 0.0]))
    np.testing.assert_array_equal(bars["y"], np.array([0.0, 3.0]))
    assert bars["x"].dtype == np.float64
    # Distinct SSA input nodes with one public name must sum their adjoints.
    x = next(
        node
        for node in program.live_nodes
        if node.op == "input" and node.attrs["name"] == "x"
    )
    another = ir.input_tensor("x", x.spec)
    mask = next(
        node
        for node in program.live_nodes
        if node.op == "input" and node.attrs["name"] == "mask"
    )
    repeated = Program(
        {"output": ir.where(mask, ir.add(x, another), ir.multiply(x, another))}
    )
    np.testing.assert_array_equal(
        vjp(
            repeated, {"mask": feeds["mask"], "x": feeds["x"]}, {"output": np.ones(2)}
        ).input_cotangents["x"],
        [2.0, 1.5],
    )


def test_existing_pruning_and_derivative_overflow_boundary() -> None:
    feeds = {"x": np.ones(2), "y": np.zeros(2)}
    program = capture(lambda x, y: ns.where(True, x, ns.sqrt(y)), feeds)
    np.testing.assert_array_equal(
        vjp(program, feeds, {"output": np.ones(2)}, inputs=["x"]).input_cotangents["x"],
        np.ones(2),
    )
    with pytest.raises(ValueError, match="derivative"):
        jvp(program, feeds, {"x": np.ones(2)})
    # Positive finite primals do not qualify overflowing tangent arithmetic.
    feeds["y"] = np.full(2, 1e-300)
    with pytest.raises(ValueError, match="finite JVP"):
        jvp(program, feeds, {"y": np.full(2, 1e308)})
    feeds["y"] = np.ones(2)
    jvp(program, feeds, {"y": np.ones(2)})


def test_comparison_of_non_wrt_real_input_is_still_not_external() -> None:
    feeds = {"x": np.ones(2), "y": np.ones(2)}
    program = capture(lambda x, y: ns.where(y > 0, x, 0), feeds, names=("x",))
    for action in (
        lambda: jvp(program, feeds, {"x": np.ones(2)}),
        lambda: vjp(program, feeds, {"output": np.ones(2)}, inputs=["x"]),
        lambda: dot_test(program, feeds, {"x": np.ones(2)}, {"output": np.ones(2)}),
    ):
        with pytest.raises(ValueError, match="external Boolean mask"):
            action()


def test_boolean_selectors_have_primal_but_no_derivative_byte_budget() -> None:
    feeds = {"mask": np.array([True, False]), "x": np.ones(2), "y": np.ones(2)}
    program = capture(lambda mask, x, y: ns.where(mask, x, y), feeds)
    primal_bytes = sum(n.spec.size * n.spec.itemsize for n in program.live_nodes)
    real_bytes = sum(
        n.spec.size * n.spec.itemsize
        for n in program.live_nodes
        if n.spec.dtype != "bool"
    )
    budget = primal_bytes + real_bytes
    for action in (
        lambda limit: jvp(program, feeds, {"x": np.ones(2)}, max_bytes=limit),
        lambda limit: vjp(program, feeds, {"output": np.ones(2)}, max_bytes=limit),
        lambda limit: dot_test(
            program, feeds, {"x": np.ones(2)}, {"output": np.ones(2)}, max_bytes=limit
        ),
    ):
        action(budget)
        with pytest.raises(ValueError, match="autodiff.*budget"):
            action(budget - 1)

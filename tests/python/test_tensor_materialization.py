"""Consumer-aware materialization: independent arithmetic and refusal gates."""

import json
import typing
from fractions import Fraction

import numpy as np
import pytest
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    MaterializationDecision,
    MaterializationPlan,
    Node,
    Program,
    TensorSpec,
    einsum,
    execute,
    exp,
    input_tensor,
    jvp,
    linearize,
    materialization_footprint,
    optimize,
    plan_materializations,
    transpose,
    transpose_program,
    vjp,
)


def _input(
    name: str, spaces: tuple[IndexSpace, ...], *, dtype: str = "float64"
) -> Node:
    return input_tensor(
        name,
        TensorSpec(
            tuple(Index(f"x{i}", space) for i, space in enumerate(spaces)),
            dtype=dtype,
            role="parameter",
            differentiable=True,
        ),
    )


def _ladder(
    *, occupied: int = 2, virtual: int = 5, dtype: str = "float64"
) -> tuple[Program, Node]:
    # One explicit auxiliary slice of cc/df_spectator_pairs.py's
    # ac,ijcd,bd->ijab ladder, expressed with an avoidable vvvv producer.
    o = IndexSpace("o", "occupied", occupied)
    v = IndexSpace("v", "virtual", virtual)
    left = _input("left", (v, v), dtype=dtype)
    right = _input("right", (v, v), dtype=dtype)
    tau = _input("tau", (o, o, v, v), dtype=dtype)
    dense = einsum("ac,bd->abcd", left, right)
    result = einsum("abcd,ijcd->ijab", dense, tau)
    return Program({"ladder": result}), dense


def _feeds(program: Program, seed: int = 1580) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        n.attrs["name"]: rng.normal(size=n.spec.shape).astype(n.spec.dtype)
        for n in program.live_nodes
        if n.op == "input"
    }


def _selected(plan: MaterializationPlan) -> list[MaterializationDecision]:
    return [d for d in plan.decisions if d.status == "selected"]


def test_df_ladder_removes_visible_dense_source_and_bounds_complete_logical_work() -> (
    None
):
    program, dense = _ladder()
    before_json = program.dumps()
    plan = plan_materializations(program, allow_reassociation=True)
    assert _selected(plan)
    assert dense not in plan.program.live_nodes
    assert plan.before.max_work_degree == 6
    assert plan.after.max_work_degree == 5
    assert plan.after.materialized_bytes < plan.before.materialized_bytes
    assert plan.after.peak_live_bytes < plan.before.peak_live_bytes
    assert plan.after.concrete_work <= plan.before.concrete_work
    assert plan.after.input_bytes == plan.before.input_bytes
    assert plan.after.output_bytes == plan.before.output_bytes == 2 * 2 * 5 * 5 * 8
    assert plan.program.outputs["ladder"].spec == program.outputs["ladder"].spec
    assert program.dumps() == before_json
    # The pass is explicit: default planning and ordinary optimizer retain vvvv.
    assert plan_materializations(program).program.logical_hash == program.logical_hash
    assert dense in optimize(program).live_nodes


@pytest.mark.parametrize("occupied,virtual", [(1, 3), (2, 5), (3, 6)])
def test_df_ladder_matches_independent_dense_numpy_oracle(
    occupied: int, virtual: int
) -> None:
    program, _ = _ladder(occupied=occupied, virtual=virtual)
    feeds = _feeds(program)
    plan = plan_materializations(program, allow_reassociation=True)
    assert _selected(plan)
    dense = np.einsum("ac,bd->abcd", feeds["left"], feeds["right"], optimize=False)
    expected = np.einsum("abcd,ijcd->ijab", dense, feeds["tau"], optimize=False)
    np.testing.assert_allclose(
        execute(plan.program, feeds).outputs["ladder"], expected, rtol=2e-13, atol=2e-13
    )


def test_ad_and_generated_derivatives_match_independent_ladder_finite_differences() -> (
    None
):
    program, _ = _ladder()
    feeds = _feeds(program)
    chosen = plan_materializations(program, allow_reassociation=True).program
    rng = np.random.default_rng(20)
    directions = {name: rng.normal(size=x.shape) for name, x in feeds.items()}
    bar = rng.normal(size=(2, 2, 5, 5))

    def independent(values: dict[str, np.ndarray]) -> np.ndarray:
        return np.einsum(
            "ac,ijcd,bd->ijab",
            values["left"],
            values["tau"],
            values["right"],
            optimize=False,
        )

    expected_jvp = (
        np.einsum("ac,ijcd,bd->ijab", directions["left"], feeds["tau"], feeds["right"])
        + np.einsum(
            "ac,ijcd,bd->ijab", feeds["left"], directions["tau"], feeds["right"]
        )
        + np.einsum(
            "ac,ijcd,bd->ijab", feeds["left"], feeds["tau"], directions["right"]
        )
    )
    reference_vjp = {
        "left": np.einsum("ijab,ijcd,bd->ac", bar, feeds["tau"], feeds["right"]),
        "right": np.einsum("ijab,ac,ijcd->bd", bar, feeds["left"], feeds["tau"]),
        "tau": np.einsum("ijab,ac,bd->ijcd", bar, feeds["left"], feeds["right"]),
    }
    for candidate in (program, chosen):
        tangent = jvp(candidate, feeds, directions).output_tangents["ladder"]
        np.testing.assert_allclose(tangent, expected_jvp, rtol=3e-13, atol=3e-13)
        reverse = vjp(candidate, feeds, {"ladder": bar}).input_cotangents
        for name in feeds:
            np.testing.assert_allclose(
                reverse[name], reference_vjp[name], rtol=3e-13, atol=3e-13
            )
        generated_jvp = linearize(candidate, list(feeds)).program
        tangent_feeds = {**feeds, **{f"d_{k}": x for k, x in directions.items()}}
        np.testing.assert_allclose(
            execute(generated_jvp, tangent_feeds).outputs["d_ladder"],
            expected_jvp,
            rtol=3e-13,
            atol=3e-13,
        )
        generated_vjp = transpose_program(
            candidate, ["ladder"], inputs=list(feeds)
        ).program
        reverse_values = execute(generated_vjp, {**feeds, "bar_ladder": bar}).outputs
        for name in feeds:
            np.testing.assert_allclose(
                reverse_values[f"bar_{name}"],
                reference_vjp[name],
                rtol=3e-13,
                atol=3e-13,
            )
    for step in (1e-5, 3e-6):
        plus = {k: x + step * directions[k] for k, x in feeds.items()}
        minus = {k: x - step * directions[k] for k, x in feeds.items()}
        np.testing.assert_allclose(
            (independent(plus) - independent(minus)) / (2 * step),
            expected_jvp,
            rtol=3e-8,
            atol=3e-8,
        )


def test_private_dummy_indices_coefficients_and_output_order_are_preserved() -> None:
    n = IndexSpace("n", "ao", 4)
    a, b, c, d = [_input(name, (n, n)) for name in ("a", "b", "c", "d")]
    producer = einsum("ip,pj,kl->ijkl", a, b, c, coefficient=Fraction(-3, 2))
    program = Program(
        {"out": einsum("ijkl,jl->ki", producer, d, coefficient=Fraction(5, 7))}
    )
    feeds = _feeds(program)
    plan = plan_materializations(program, allow_reassociation=True)
    assert _selected(plan)
    dense = -1.5 * np.einsum("ip,pj,kl->ijkl", feeds["a"], feeds["b"], feeds["c"])
    expected = (5 / 7) * np.einsum("ijkl,jl->ki", dense, feeds["d"])
    np.testing.assert_allclose(
        execute(plan.program, feeds).outputs["out"], expected, rtol=3e-13, atol=3e-13
    )


def test_plan_and_selected_program_roundtrip_deterministically() -> None:
    program, _ = _ladder()
    first = plan_materializations(program, allow_reassociation=True)
    replay = Program.loads(program.dumps())
    second = plan_materializations(replay, allow_reassociation=True)
    assert first.to_payload() == second.to_payload()
    payload = json.loads(json.dumps(first.to_payload()))
    assert payload["source_hash"] == program.logical_hash
    assert payload["selected_hash"] != program.logical_hash
    assert payload["runtime_endpoint_selection_proven"] is False
    selected_replay = Program.loads(first.program.dumps())
    assert selected_replay.logical_hash == first.program.logical_hash
    feeds = _feeds(program)
    np.testing.assert_array_equal(
        execute(first.program, feeds).outputs["ladder"],
        execute(selected_replay, feeds).outputs["ladder"],
    )


@pytest.mark.parametrize("kind", ["output", "multi", "repeated", "view", "nonlinear"])
def test_shared_alias_and_observable_boundaries_retain_materialization(
    kind: str,
) -> None:
    program, dense = _ladder()
    root = program.outputs["ladder"]
    outputs = {"ladder": root}
    if kind == "output":
        outputs["dense"] = dense
    elif kind == "multi":
        outputs["second"] = einsum("abcd->ab", dense)
    elif kind == "repeated":
        outputs = {"norm": einsum("abcd,abcd->", dense, dense)}
    elif kind == "view":
        outputs = {"view": transpose(dense, (1, 0, 2, 3))}
    else:
        outputs = {"sum": exp(dense)}
    original = Program(outputs)
    plan = plan_materializations(original, allow_reassociation=True)
    assert not _selected(plan)
    assert plan.program.logical_hash == original.logical_hash
    assert dense in plan.program.live_nodes
    assert any(
        (
            "retained-output"
            if kind == "output"
            else "consumer-boundary"
            if kind in ("view", "nonlinear")
            else "shared-value"
        )
        in d.reason
        for d in plan.decisions
    )


@pytest.mark.parametrize(
    "options,reason",
    [
        ({}, "reduction-order"),
        ({"allow_reassociation": True, "available_rules": ()}, "unsupported-provider"),
        ({"allow_reassociation": True, "max_materialized_bytes": 0}, "resource-budget"),
        (
            {
                "allow_reassociation": True,
                "max_operands": 3,
                "max_intermediate_axes": {"virtual": 0, "occupied": 0},
            },
            "work-regression",
        ),
    ],
)
def test_permission_provider_and_resource_refusals_preserve_exact_source(
    options: dict[str, typing.Any], reason: str
) -> None:
    program, _ = _ladder()
    plan = plan_materializations(program, **options)
    assert not _selected(plan)
    assert plan.program.logical_hash == program.logical_hash
    assert any(reason in d.reason for d in plan.decisions)


def test_precision_execution_and_float32_are_not_silently_reassociated() -> None:
    program, _ = _ladder()
    explicit = Program(
        program.outputs, provenance={"precision_execution": {"schema": "probe"}}
    )
    fp32, _ = _ladder(dtype="float32")
    for source, reason in (
        (explicit, "precision-execution"),
        (fp32, "precision-domain"),
    ):
        plan = plan_materializations(source, allow_reassociation=True)
        assert not _selected(plan)
        assert plan.program.logical_hash == source.logical_hash
        assert any(reason in d.reason for d in plan.decisions)


def test_exact_budget_threshold_and_truncated_inventory() -> None:
    program, _ = _ladder()
    unconstrained = plan_materializations(program, allow_reassociation=True)
    size = unconstrained.after.materialized_bytes
    exact = plan_materializations(
        program, allow_reassociation=True, max_materialized_bytes=size
    )
    below = plan_materializations(
        program, allow_reassociation=True, max_materialized_bytes=size - 1
    )
    assert _selected(exact) and not _selected(below)
    limited = plan_materializations(program, max_candidates=1)
    assert len(limited.decisions) == 1 and limited.truncated > 0
    assert limited.program.logical_hash == program.logical_hash


def test_opaque_rank_four_sources_request_proof_without_inventing_factorization() -> (
    None
):
    program, _ = _ladder()
    spec = program.outputs["ladder"].spec
    source = _input("weights", tuple(i.space for i in spec.indices))
    original = Program({"out": einsum("ijab->ab", source)})
    plan = plan_materializations(original, allow_reassociation=True)
    assert plan.program.logical_hash == original.logical_hash
    assert any(
        d.status == "requires-provider" and d.rule == "factorized-source"
        for d in plan.decisions
    )
    assert plan.before.output_bytes == 5 * 5 * 8


def test_logical_lifetimes_charge_inputs_outputs_and_dead_definition_boundaries() -> (
    None
):
    n = IndexSpace("n", "ao", 2)
    x, y = _input("x", (n, n)), _input("y", (n, n))
    first = einsum("ij,jk->ik", x, y)
    second = einsum("ij,jk->ik", first, y)
    dead = einsum("ij,kl->ijkl", x, y)
    program = Program({"out": second}, definitions=(dead,))
    footprint = materialization_footprint(program)
    assert footprint.input_bytes == 64
    assert footprint.materialized_bytes == 64
    assert footprint.output_bytes == 32
    # At second's production both first and second are live, plus caller inputs.
    assert footprint.peak_live_bytes == 128
    assert footprint.max_materialized_degree == 2
    passthrough = materialization_footprint(Program({"out": x}))
    assert passthrough.input_bytes == passthrough.output_bytes == 32
    assert passthrough.peak_live_bytes == 32
    assert passthrough.materialized_bytes == 0


@pytest.mark.parametrize(
    "options",
    [
        {"allow_reassociation": 1},
        {"max_operands": True},
        {"max_operands": 9},
        {"max_candidates": 0},
        {"available_rules": ("vendor-gemm",)},
        {"available_rules": ("single-consumer-einsum",) * 2},
        {"max_materialized_bytes": -1},
        {"max_intermediate_axes": {"unknown": 2}},
    ],
)
def test_invalid_plan_requests_fail_closed(options: dict[str, typing.Any]) -> None:
    program, _ = _ladder()
    with pytest.raises((TypeError, ValueError)):
        plan_materializations(program, **options)


def test_high_live_storage_regression_is_rejected_even_with_lower_work() -> None:
    program, dense = _ladder(occupied=8, virtual=2)
    plan = plan_materializations(program, allow_reassociation=True)
    assert not _selected(plan)
    assert dense in plan.program.live_nodes
    assert any("storage-regression" in d.reason for d in plan.decisions)


def test_diagonal_producer_and_finite_operand_bound_are_explicit_barriers() -> None:
    n = IndexSpace("n", "ao", 3)
    a, b = _input("a", (n, n)), _input("b", (n, n))
    c = _input("c", (n,))
    producer = einsum("ii,jk,l->ijkl", a, b, c)
    original = Program({"out": einsum("ijkl,jk->il", producer, b)})
    blocked = plan_materializations(original, allow_reassociation=True)
    assert blocked.program.logical_hash == original.logical_hash
    assert any("diagonal-domain" in d.reason for d in blocked.decisions)
    producer = einsum("im,mj,kl->ijkl", a, b, a)
    original = Program({"out": einsum("ijkl,jl->ik", producer, b)})
    blocked = plan_materializations(original, allow_reassociation=True, max_operands=3)
    assert blocked.program.logical_hash == original.logical_hash
    assert any("search-bound" in d.reason for d in blocked.decisions)


def test_rebuilt_downstream_consumers_are_reconsidered_without_stale_proofs() -> None:
    program, _ = _ladder()
    spaces = tuple(i.space for i in program.outputs["ladder"].spec.indices[-2:])
    weight = _input("weight", spaces)
    source = Program({"out": einsum("ijab,ab->ij", program.outputs["ladder"], weight)})
    feeds = _feeds(source)
    plan = plan_materializations(source, allow_reassociation=True)
    assert plan.after.max_materialized_degree < 4
    assert plan.after.max_work_degree < plan.before.max_work_degree
    assert plan.after.output_bytes == 2 * 2 * 8
    expected = np.einsum(
        "ac,bd,ijcd,ab->ij",
        feeds["left"],
        feeds["right"],
        feeds["tau"],
        feeds["weight"],
        optimize=False,
    )
    np.testing.assert_allclose(
        execute(plan.program, feeds).outputs["out"], expected, rtol=3e-13, atol=3e-13
    )


def _large_legal_contraction() -> Program:
    # Shape-only: source storage is about 68 GiB, but an unneeded search subset
    # combining z and x would require exactly 2**63 bytes. Never allocate arrays.
    n = IndexSpace("n", "ao", 128)
    g = IndexSpace("g", "auxiliary", 65536)
    a, b = _input("a", (n, n)), _input("b", (n, n))
    z = _input("z", (n, n, n, n))
    x, y = _input("x", (g, g)), _input("y", (g, g))
    dense = einsum("ac,bd->abcd", a, b)
    return Program({"out": einsum("abcd,abcd,gh,gh->", dense, z, x, y)})


def test_oversized_search_subset_does_not_discard_legal_alternatives() -> None:
    source = _large_legal_contraction()
    source_json = source.dumps()
    plan = plan_materializations(source, allow_reassociation=True)
    assert _selected(plan)
    assert plan.after.materialized_bytes < plan.before.materialized_bytes
    assert plan.after.peak_live_bytes <= plan.before.peak_live_bytes
    assert plan.after.concrete_work <= plan.before.concrete_work
    assert plan.after.input_bytes == plan.before.input_bytes
    assert plan.after.output_bytes == plan.before.output_bytes == 8
    assert source.dumps() == source_json


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_materialized_bytes": 0}, "resource-budget"),
        (
            {"max_intermediate_axes": {"ao": 0, "auxiliary": 0}},
            "work-regression",
        ),
    ],
)
def test_oversized_search_preserves_original_when_no_plan_is_admissible(
    options: dict[str, typing.Any], reason: str
) -> None:
    source = _large_legal_contraction()
    plan = plan_materializations(source, allow_reassociation=True, **options)
    assert not _selected(plan)
    assert plan.program.logical_hash == source.logical_hash
    assert plan.after == plan.before
    assert any(reason in decision.reason for decision in plan.decisions)

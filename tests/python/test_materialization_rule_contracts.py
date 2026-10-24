"""Shared requests and visible-sum rules against MP2/CC source contracts."""

import json
import typing

import numpy as np
import pytest
from generativeqc_compiler.common.materialization import (
    MaterializationRuleRequest,
    materialization_diagnostic,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    execute,
    exp,
    input_tensor,
    jvp,
    linearize,
    plan_materializations,
    transpose_program,
    vjp,
)


def _input(name: str, n: int, rank: int, dtype: str = "float64") -> Node:
    space = IndexSpace("mo", "ao", n)
    return input_tensor(
        name,
        TensorSpec(
            tuple(Index(f"p{i}", space) for i in range(rank)),
            role="parameter",
            differentiable=True,
            dtype=dtype,
        ),
    )


def _mp2_source(n: int = 4, dtype: str = "float64") -> tuple[Program, Node, Node]:
    # Fock-like part of mp2_gradient.hpp::FactorizedTwoElectronWeights.
    # The correlation iajb term is a distinct opaque/block source, not guessed.
    f, occupied, density = (
        _input(name, n, 2, dtype) for name in ("f", "occupied", "density")
    )
    direct = einsum("pq,rs->pqrs", f, occupied, coefficient=2)
    exchange = einsum("ps,qr->pqrs", f, occupied)
    weights = add(direct, exchange, coefficients=(1, -1))
    return Program({"action": einsum("pqrs,rs->pq", weights, density)}), weights, direct


def _feeds(n: int = 4, seed: int = 1580) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "f": rng.normal(size=(n, n)),
        "occupied": np.diag([1.0] * (n // 2) + [0.0] * (n - n // 2)),
        "density": rng.normal(size=(n, n)),
    }


def _oracle(values: dict[str, np.ndarray]) -> np.ndarray:
    f, occ, density = (values[name] for name in ("f", "occupied", "density"))
    n = len(f)
    out = np.zeros_like(f)
    # Independent literal source formula, without compiler or einsum lowering.
    for p, q, r, s in np.ndindex((n, n, n, n)):
        out[p, q] += (2 * f[p, q] * occ[r, s] - f[p, s] * occ[q, r]) * density[r, s]
    return out


def _sum_decision(plan: typing.Any) -> typing.Any:
    return next(d for d in plan.decisions if d.rule == "distribute-add-einsum")


@pytest.mark.parametrize("n", [2, 4, 6])
def test_mp2_fock_like_source_uses_visible_factors_without_dense_weight(n: int) -> None:
    program, weights, _ = _mp2_source(n)
    original = program.dumps()
    plan = plan_materializations(program, allow_reassociation=True)
    assert _sum_decision(plan).status == "selected"
    assert weights not in plan.program.live_nodes
    assert plan.after.max_materialized_degree == 2
    assert plan.after.max_work_degree < plan.before.max_work_degree
    assert plan.after.materialized_bytes < plan.before.materialized_bytes
    assert plan.after.peak_live_bytes <= plan.before.peak_live_bytes
    assert plan.after.concrete_work <= plan.before.concrete_work
    assert plan.after.input_bytes == plan.before.input_bytes
    assert plan.after.output_bytes == plan.before.output_bytes == n * n * 8
    feeds = _feeds(n)
    np.testing.assert_allclose(
        execute(plan.program, feeds).outputs["action"],
        _oracle(feeds),
        rtol=2e-13,
        atol=2e-13,
    )
    assert program.dumps() == original


def test_sum_rule_interpreted_and_generated_derivatives_match_independent_source() -> (
    None
):
    program, _, _ = _mp2_source()
    chosen = plan_materializations(program, allow_reassociation=True).program
    feeds = _feeds()
    rng = np.random.default_rng(24)
    direction = rng.normal(size=(4, 4))
    bar = rng.normal(size=(4, 4))
    occupied, density = feeds["occupied"], feeds["density"]
    expected_jvp = (
        2 * direction * np.sum(occupied * density) - direction @ density.T @ occupied.T
    )
    expected_vjp = 2 * bar * np.sum(occupied * density) - bar @ occupied @ density
    for source in (program, chosen):
        np.testing.assert_allclose(
            jvp(source, feeds, {"f": direction}).output_tangents["action"],
            expected_jvp,
            rtol=3e-13,
            atol=3e-13,
        )
        np.testing.assert_allclose(
            vjp(source, feeds, {"action": bar}).input_cotangents["f"],
            expected_vjp,
            rtol=3e-13,
            atol=3e-13,
        )
        forward = linearize(source, ["f"]).program
        reverse = transpose_program(source, ["action"], inputs=["f"]).program
        np.testing.assert_allclose(
            execute(forward, {**feeds, "d_f": direction}).outputs["d_action"],
            expected_jvp,
            rtol=3e-13,
            atol=3e-13,
        )
        np.testing.assert_allclose(
            execute(reverse, {**feeds, "bar_action": bar}).outputs["bar_f"],
            expected_vjp,
            rtol=3e-13,
            atol=3e-13,
        )
    for step in (1e-4, 1e-5):
        plus, minus = (
            {**feeds, "f": feeds["f"] + step * direction},
            {**feeds, "f": feeds["f"] - step * direction},
        )
        np.testing.assert_allclose(
            (_oracle(plus) - _oracle(minus)) / (2 * step),
            expected_jvp,
            rtol=3e-9,
            atol=3e-9,
        )


def test_actual_cc_ladder_source_and_t1_response_remain_numerically_correct() -> None:
    from generativeqc_compiler.cc.doubles import build_ccsd_program

    from tools.generativeqc_cc.oracle import dense_feeds, random_case

    complete = build_ccsd_program(2, 8)
    program = Program({"ladder": complete.outputs["D05_vv_ladder"]})
    feeds = dense_feeds(*random_case(2, 8, 1580))
    plan = plan_materializations(program, allow_reassociation=True)
    assert any(
        d.rule == "distribute-add-einsum" and d.status == "selected"
        for d in plan.decisions
    )

    def reference(t1: np.ndarray, tangent: np.ndarray | None = None) -> np.ndarray:
        w = (
            np.einsum("acbd->abcd", feeds["vvvv"])
            - np.einsum("kdac,kb->abcd", feeds["ovvv"], t1)
            - np.einsum("kcbd,ka->abcd", feeds["ovvv"], t1)
        )
        tau = feeds["t2"] + np.einsum("ia,jb->ijab", t1, t1)
        if tangent is None:
            return np.einsum("abcd,ijcd->ijab", w, tau)
        dw = -np.einsum("kdac,kb->abcd", feeds["ovvv"], tangent) - np.einsum(
            "kcbd,ka->abcd", feeds["ovvv"], tangent
        )
        dtau = np.einsum("ia,jb->ijab", tangent, t1) + np.einsum(
            "ia,jb->ijab", t1, tangent
        )
        return np.einsum("abcd,ijcd->ijab", dw, tau) + np.einsum(
            "abcd,ijcd->ijab", w, dtau
        )

    expected = reference(feeds["t1"])
    rng = np.random.default_rng(17)
    direction = rng.normal(size=feeds["t1"].shape)
    bar = rng.normal(size=expected.shape)
    expected_tangent = reference(feeds["t1"], direction)
    expected_reverse = np.zeros_like(direction)
    for index in np.ndindex(direction.shape):
        basis = np.zeros_like(direction)
        basis[index] = 1
        expected_reverse[index] = np.sum(bar * reference(feeds["t1"], basis))
    for source in (program, plan.program):
        np.testing.assert_allclose(
            execute(source, feeds).outputs["ladder"], expected, rtol=3e-12, atol=3e-12
        )
        np.testing.assert_allclose(
            jvp(source, feeds, {"t1": direction}).output_tangents["ladder"],
            expected_tangent,
            rtol=3e-12,
            atol=3e-12,
        )
        np.testing.assert_allclose(
            vjp(source, feeds, {"ladder": bar}, inputs=["t1"]).input_cotangents["t1"],
            expected_reverse,
            rtol=3e-12,
            atol=3e-12,
        )
        generated = linearize(source, ["t1"]).program
        np.testing.assert_allclose(
            execute(generated, {**feeds, "d_t1": direction}).outputs["d_ladder"],
            expected_tangent,
            rtol=3e-12,
            atol=3e-12,
        )
        reverse = transpose_program(source, ["ladder"], inputs=["t1"]).program
        np.testing.assert_allclose(
            execute(reverse, {**feeds, "bar_ladder": bar}).outputs["bar_t1"],
            expected_reverse,
            rtol=3e-12,
            atol=3e-12,
        )
    for step in (1e-5, 3e-6):
        np.testing.assert_allclose(
            (
                reference(feeds["t1"] + step * direction)
                - reference(feeds["t1"] - step * direction)
            )
            / (2 * step),
            expected_tangent,
            rtol=3e-8,
            atol=3e-8,
        )


@pytest.mark.parametrize(
    "kind", ["retained", "shared", "repeat", "opaque", "nonlinear"]
)
def test_sum_rule_does_not_guess_private_or_factorized_source_semantics(
    kind: str,
) -> None:
    program, weights, term = _mp2_source()
    outputs = dict(program.outputs)
    if kind == "retained":
        outputs["weights"] = weights
    elif kind == "shared":
        outputs["term"] = term
    elif kind == "repeat":
        weights = add(term, term)
        outputs = {"action": einsum("pqrs,rs->pq", weights, _input("density", 4, 2))}
    elif kind == "opaque":
        weights = add(term, _input("opaque", 4, 4))
        outputs = {"action": einsum("pqrs,rs->pq", weights, _input("density", 4, 2))}
    else:
        outputs = {"nonlinear": exp(weights)}
    source = Program(outputs)
    plan = plan_materializations(source, allow_reassociation=True)
    assert plan.program.logical_hash == source.logical_hash
    assert not any(d.status == "selected" for d in plan.decisions)
    assert all(d.request is not None for d in plan.decisions)


@pytest.mark.parametrize(
    "options,reason",
    [
        ({}, "reduction-order"),
        (
            {
                "allow_reassociation": True,
                "available_rules": ("single-consumer-einsum",),
            },
            "unsupported-provider",
        ),
        (
            {"allow_reassociation": True, "max_distribution_terms": 1},
            "distribution-bound",
        ),
        ({"allow_reassociation": True, "max_materialized_bytes": 0}, "resource-budget"),
    ],
)
def test_sum_rule_permissions_and_budgets_retain_original(
    options: dict[str, typing.Any], reason: str
) -> None:
    program, _, _ = _mp2_source()
    plan = plan_materializations(program, **options)
    assert plan.program.logical_hash == program.logical_hash
    assert any(reason in d.reason for d in plan.decisions)


def test_sum_rule_precision_and_exact_resource_thresholds() -> None:
    program, _, _ = _mp2_source()
    strict = Program(
        program.outputs, provenance={"precision_execution": {"schema": "explicit"}}
    )
    fp32, _, _ = _mp2_source(dtype="float32")
    for source, reason in ((strict, "precision-execution"), (fp32, "precision-domain")):
        plan = plan_materializations(source, allow_reassociation=True)
        assert plan.program.logical_hash == source.logical_hash
        assert any(reason in d.reason for d in plan.decisions)
    accepted = plan_materializations(program, allow_reassociation=True)
    size = accepted.after.materialized_bytes
    exact = plan_materializations(
        program, allow_reassociation=True, max_materialized_bytes=size
    )
    below = plan_materializations(
        program, allow_reassociation=True, max_materialized_bytes=size - 1
    )
    assert _sum_decision(exact).status == "selected"
    assert below.program.logical_hash == program.logical_hash


@pytest.mark.parametrize("limit", [0, True, 9])
def test_invalid_distribution_limits_fail_before_planning(limit: typing.Any) -> None:
    program, _, _ = _mp2_source()
    with pytest.raises(ValueError, match="max_distribution_terms"):
        plan_materializations(program, max_distribution_terms=limit)


def test_request_snapshots_full_source_consumer_precision_and_resource_identity() -> (
    None
):
    program, _, _ = _mp2_source()
    first = plan_materializations(program, allow_reassociation=True)
    replay = plan_materializations(
        Program.loads(program.dumps()), allow_reassociation=True
    )
    assert first.to_payload() == replay.to_payload()
    request = _sum_decision(first).request
    assert request is not None
    payload = request.to_payload()
    assert payload["source_identity"] == program.logical_hash
    assert payload["producer"]["spec"]["dtype"] == "float64"
    assert payload["consumers"][0]["operand_positions"] == [0]
    assert payload["automatic_rewrite"] is False
    assert payload["required_contracts"]
    detached = json.loads(json.dumps(payload))
    detached["producer"]["spec"]["dtype"] = "float32"
    assert request.to_payload() == payload
    assert (
        _sum_decision(
            plan_materializations(
                program, allow_reassociation=True, max_materialized_bytes=0
            )
        ).request.request_hash
        != request.request_hash
    )
    assert (
        _sum_decision(
            plan_materializations(
                program, allow_reassociation=True, max_distribution_terms=1
            )
        ).request.request_hash
        != request.request_hash
    )
    changed = Program(
        {
            "changed": einsum(
                "pqrs,rs->qp",
                program.outputs["action"].inputs[0],
                _input("density", 4, 2),
            )
        }
    )
    assert (
        _sum_decision(
            plan_materializations(changed, allow_reassociation=True)
        ).request.request_hash
        != request.request_hash
    )


def test_shared_diagnostic_uses_same_request_without_granting_admission() -> None:
    program, _, _ = _mp2_source()
    request = _sum_decision(plan_materializations(program)).request
    assert request is not None
    payload = request.to_payload()
    diagnostic = materialization_diagnostic(
        origin=payload["origin"],
        subject=payload["subject"],
        dense_elements=256,
        dense_growth_degree=4,
        certificate_scope="logical shape; no source or alias proof",
        rule_request=request,
    )
    assert diagnostic["rule_request"] == payload
    assert diagnostic["consumer_abi_verified"] is False
    assert diagnostic["automatic_rewrite"] is False
    assert diagnostic["support"]["strict_reduction_proven"] is False
    with pytest.raises(ValueError, match="does not match"):
        materialization_diagnostic(
            origin="native-source",
            subject=payload["subject"],
            dense_elements=256,
            certificate_scope="probe",
            rule_request=request,
        )


def test_request_is_data_only_and_does_not_reinterpret_opaque_inputs() -> None:
    program, _, _ = _mp2_source()
    weights = _input("opaque_weights", 4, 4)
    source = Program({"out": einsum("pqrs,rs->pq", weights, _input("density", 4, 2))})
    plan = plan_materializations(source, allow_reassociation=True)
    assert plan.program.logical_hash == source.logical_hash
    request = next(d.request for d in plan.decisions if d.status == "requires-provider")
    assert request is not None
    assert request.to_payload()["external_rule_families"] == [
        "factorized-source",
        "block-source",
    ]
    with pytest.raises(TypeError):
        MaterializationRuleRequest(
            origin="probe",
            source_identity=program.logical_hash,
            subject={"name": "x"},
            producer={"callback": lambda: None},
            consumers=(),
            retained_output=False,
            allow_reassociation=True,
        )


def test_actual_cc_ladder_retains_losing_lower_memory_candidate() -> None:
    from generativeqc_compiler.cc.doubles import build_ccsd_program

    from tools.generativeqc_cc.oracle import dense_feeds, random_case

    complete = build_ccsd_program(2, 4)
    source = Program({"ladder": complete.outputs["D05_vv_ladder"]})
    plan = plan_materializations(source, allow_reassociation=True)
    assert plan.program.logical_hash == source.logical_hash
    assert any(
        "work-regression" in d.reason
        and d.candidate is not None
        and d.candidate.materialized_bytes < d.before.materialized_bytes
        for d in plan.decisions
    )
    feeds = dense_feeds(*random_case(2, 4, 41))
    np.testing.assert_array_equal(
        execute(plan.program, feeds).outputs["ladder"],
        execute(source, feeds).outputs["ladder"],
    )


def test_common_request_snapshots_caller_data_and_retained_output_barrier() -> None:
    producer = {"spec": {"shape": [3, 3, 3, 3], "dtype": "float64"}}
    consumers = [{"op": "einsum", "operand_positions": [0]}]
    request = MaterializationRuleRequest(
        origin="native-audit",
        source_identity="verified-source-version",
        subject={"name": "cotangent"},
        producer=producer,
        consumers=consumers,
        retained_output=True,
        allow_reassociation=False,
    )
    old = request.to_payload()
    producer["spec"]["shape"][0] = 7
    consumers[0]["operand_positions"][0] = 2
    assert request.to_payload() == old
    assert old["external_rule_families"] == []
    assert old["automatic_rewrite"] is False
    with pytest.raises(ValueError, match="does not match"):
        materialization_diagnostic(
            origin="native-audit",
            subject={"name": "cotangent"},
            dense_elements=81,
            certificate_scope="probe",
            retained_output=False,
            rule_request=request,
        )

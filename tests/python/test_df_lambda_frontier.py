"""Prove compact immutable Lambda storage without changing the adjoint graph."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_lambda_reduction import (
    build_df_lambda_reduction_programs,
)
from generativeqc_compiler.tensor import Node, Program, execute
from generativeqc_compiler.tensor.ir import input_tensor
from generativeqc_compiler.tensor.iteration_reuse import (
    analyze_iteration_reuse,
    invariant_frontier,
)
from test_df_cc_lambda import case
from test_df_lambda_reduction import prepare

from tools.generate_df_lambda import (
    core_reuse_frontier,
    core_reuse_identity,
    core_reuse_plan,
    cuda_source,
    header,
    matrix_programs,
)
from tools.generate_rccsd_native import _arena_plan, _cuda_program


@pytest.mark.parametrize("occupied,virtuals", [(1, 3), (2, 3), (3, 2), (3, 4)])
def test_frontier_keeps_the_original_core_under_unrelated_signed_seeds(
    occupied: int, virtuals: int
) -> None:
    """Detached persistent values reproduce the full independent original graph."""
    _, _, feeds = case(occupied, virtuals, q=5)
    pipeline = build_df_lambda_reduction_programs(occupied, virtuals)
    _, cuts = prepare(pipeline, feeds)
    program = matrix_program(pipeline.core)
    plan = analyze_iteration_reuse(
        program,
        invariant_inputs=tuple(
            node.attrs["name"]
            for node in program.live_nodes
            if node.op == "input" and not node.attrs["name"].startswith("bar_")
        ),
    )
    frontier = invariant_frontier(program, plan)
    assert len(frontier) < len(plan.invariant_nodes)
    saved = {f"saved_{index}": node for index, node in enumerate(frontier)}
    cached = execute(Program(saved), {**feeds, **cuts}).outputs
    replacements = {
        node: input_tensor(name, replace(node.spec, role="input"))
        for name, node in saved.items()
    }
    for node in program.dependency_order:
        if node not in replacements:
            replacements[node] = (
                node
                if node.op == "input"
                else Node(
                    node.op,
                    tuple(replacements[source] for source in node.inputs),
                    node.spec,
                    node.attributes,
                )
            )
    dynamic = Program(
        {name: replacements[node] for name, node in program.outputs.items()}
    )
    assert sum(node.op != "input" for node in dynamic.live_nodes) == len(
        plan.dynamic_nodes
    )
    rng = np.random.default_rng(61210)
    for scale in (0.0, 1.0, 1e-5, -3.0):
        frame = {
            **feeds,
            **cuts,
            "bar_singles_residual": rng.normal(size=(occupied, virtuals)) * scale,
            "bar_doubles_residual": rng.normal(
                size=(occupied, occupied, virtuals, virtuals)
            )
            * scale,
        }
        expected = execute(program, frame).outputs
        actual = execute(dynamic, {**frame, **cached}).outputs
        for name in expected:
            np.testing.assert_allclose(
                actual[name], expected[name], atol=2e-14, rtol=2e-14
            )
    with pytest.raises(ValueError, match="complete reuse plan"):
        invariant_frontier(program, replace(plan, dynamic_nodes=()))


def test_frontier_native_slots_are_disjoint_and_scratch_is_already_bounded() -> None:
    """The compiler charges only the boundary, never a second dynamic arena."""
    program = matrix_programs()["staged_core"]
    plan = core_reuse_plan()
    frontier = core_reuse_frontier()
    assert frontier == invariant_frontier(program, plan)
    assert len(frontier) == 40 and len(plan.invariant_nodes) == 103
    arena = _arena_plan(program, retained_nodes=frontier)
    assert not Counter(arena.sizes[len(frontier) :]) - Counter(
        _arena_plan(program).sizes
    )

    def elements(expression: str) -> int:
        if expression.isdigit():
            return int(expression)
        assert expression.startswith("checked_product({") and expression.endswith("})")
        return math.prod(
            {"o": 9, "v": 221}[dimension]
            for dimension in expression[len("checked_product({") : -2].split(",")
        )

    retained_bytes = 8 * sum(elements(size) for size in arena.sizes[: len(frontier)])
    assert retained_bytes == 396517056
    assert retained_bytes < 8 * sum(
        elements(size)
        for size in _arena_plan(program, retained_nodes=plan.invariant_nodes).sizes
    )
    generated = cuda_source()
    for phase, nodes in (
        ("prepare", plan.invariant_nodes),
        ("dynamic", plan.dynamic_nodes),
    ):
        body = generated.split(
            f"run_staged_core_reuse_{phase}(StagedCudaState& s){{", 1
        )[1].split("\n}", 1)[0]
        assert body.count("=retain(") == len(frontier)
        assert body.count("=allocate(") == len(arena.sizes) - len(frontier)
        assert body.count("<<<") + body.count(".execute(") == len(nodes)
    assert core_reuse_identity() != plan.identity
    assert core_reuse_identity() in header() and plan.identity in header()
    assert "staged_core_reuse_scratch_arena_elements" in header()


def test_unproved_or_duplicate_storage_nodes_are_rejected() -> None:
    """A storage hint cannot promote dynamic values or bypass a dependency proof."""
    program = matrix_programs()["staged_core"]
    plan = core_reuse_plan()
    common = {"reuse_plan": plan, "reuse_phase": "prepare"}
    with pytest.raises(ValueError, match="proven invariant"):
        _cuda_program(
            program,
            "probe",
            "void",
            reuse_storage_nodes=(plan.dynamic_nodes[0],),
            **common,
        )
    with pytest.raises(ValueError, match="unique proven invariant"):
        _cuda_program(
            program,
            "probe",
            "void",
            reuse_storage_nodes=(plan.invariant_nodes[0],) * 2,
            **common,
        )
    with pytest.raises(ValueError, match="preserve the invariant frontier"):
        _cuda_program(
            program,
            "probe",
            "void",
            reuse_storage_nodes=core_reuse_frontier()[1:],
            **common,
        )
    with pytest.raises(ValueError, match="proven reuse plan"):
        _cuda_program(program, "probe", "void", retained_arena_field="retained")

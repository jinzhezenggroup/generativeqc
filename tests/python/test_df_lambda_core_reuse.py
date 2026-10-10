"""Compiler-proven immutable primal staging and its bounded native fallback."""

from __future__ import annotations

import typing
from dataclasses import replace

import numpy as np
import pytest
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_lambda_reduction import (
    build_df_lambda_reduction_programs,
)
from generativeqc_compiler.tensor import Node, Program, execute
from generativeqc_compiler.tensor.ir import input_tensor
from generativeqc_compiler.tensor.iteration_reuse import analyze_iteration_reuse
from test_df_cc_lambda import case, run
from test_df_cc_lambda import probe as _native_probe
from test_df_lambda_reduction import prepare

probe = _native_probe


@pytest.mark.parametrize("o,v", [(1, 3), (2, 3), (3, 2)])
def test_fixed_core_values_are_independent_of_all_adjoint_seeds(o: int, v: int) -> None:
    """Split the existing IR, rather than using another set of Lambda equations."""
    _, _, feeds = case(o, v, q=5)
    pipeline = build_df_lambda_reduction_programs(o, v)
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
    assert plan.invariant_nodes and plan.dynamic_nodes
    dependencies = dict(plan.input_dependencies)
    assert all(
        not any(name.startswith("bar_") for name in dependencies[node])
        for node in plan.invariant_nodes
    )
    saved = {f"saved_{index}": node for index, node in enumerate(plan.invariant_nodes)}
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
    rng = np.random.default_rng(2136)
    for scale in (0.0, 1.0, 1e-5, -3.0):
        frame = {
            **feeds,
            **cuts,
            "bar_singles_residual": rng.normal(size=(o, v)) * scale,
            "bar_doubles_residual": rng.normal(size=(o, o, v, v)) * scale,
        }
        expected = execute(program, frame).outputs
        actual = execute(dynamic, {**frame, **cached}).outputs
        for name in expected:
            # Named NumPy feeds are contiguous copies rather than interpreter
            # views; protect the existing scientific gate, not BLAS bit patterns.
            np.testing.assert_allclose(
                actual[name], expected[name], atol=2e-14, rtol=2e-14
            )


def test_generated_phases_use_distinct_retained_storage_and_descriptors() -> None:
    """Generated phases share a proven arena layout, not their descriptor slots."""
    from tools.generate_df_lambda import (
        core_reuse_plan,
        cuda_header,
        cuda_source,
        header,
    )

    plan = core_reuse_plan()
    generated = cuda_source()
    assert "staged_core_reuse_arena_elements" in header()
    assert plan.identity in header()
    assert "core_reuse_contraction_host_bytes" in cuda_header()
    for phase, nodes in (
        ("prepare", plan.invariant_nodes),
        ("dynamic", plan.dynamic_nodes),
    ):
        body = generated.split(
            f"run_staged_core_reuse_{phase}(StagedCudaState& s){{", 1
        )[1].split("\n}", 1)[0]
        assert "auto* arena=s.response_arena;" in body
        assert "auto* retained_arena=s.core_reuse_arena;" in body
        assert body.count("<<<") + body.count(".execute(") == len(nodes)
        assert f"s.core_reuse_{phase}_contractions.execute(" in body


@pytest.mark.parametrize("o,v", [(1, 3), (2, 3), (3, 2)])
@pytest.mark.parametrize("source", [False, True])
def test_native_core_reuse_matches_uncached_fp64_and_refuses_budget(
    probe: typing.Any, o: int, v: int, source: bool
) -> None:
    """Keep exact audits and all parameter/factor outputs on a common primal."""
    _, _, arrays = case(o, v, q=5)
    status, control, _, old_counts, error = run(
        probe, arrays, source=source, core_reuse=False
    )
    assert status == 0, error
    status, actual, values, counts, error = run(probe, arrays, source=source)
    assert status == 0, error
    assert counts[20] == 1 and counts[21] > 0 and counts[22] == 1
    assert counts[23] == counts[1] + 1  # Final factor VJP also requests core seeds.
    assert counts[1] == old_counts[1]
    assert counts[16] < old_counts[16] and counts[18] < old_counts[18]
    assert max(values[1:]) < 1e-9
    for value, expected in zip(actual, control, strict=True):
        np.testing.assert_array_equal(value, expected)
    status, fallback, _, bounded_counts, error = run(
        probe, arrays, source=source, budget=int(old_counts[2])
    )
    assert status == 0, error
    assert bounded_counts[20] == 0 and bounded_counts[13] == 1
    for value, expected in zip(fallback, control, strict=True):
        np.testing.assert_array_equal(value, expected)


def test_owner_local_retention_does_not_cross_equal_shape_problems(
    probe: typing.Any,
) -> None:
    """A new Hamiltonian with the same dimensions gets a new preparation epoch."""
    _, _, arrays = case(2, 3, q=5)
    status, _, _, _, error = run(probe, arrays)
    assert status == 0, error
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["fov"] *= 0.9
    status, control, _, _, error = run(probe, changed, core_reuse=False)
    assert status == 0, error
    status, actual, _, counts, error = run(probe, changed)
    assert status == 0, error
    assert counts[22] == 1
    for value, expected in zip(actual, control, strict=True):
        np.testing.assert_array_equal(value, expected)

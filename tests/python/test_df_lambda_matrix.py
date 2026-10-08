"""Batched adjoints against independent expanded equations and scalar Q rows."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_lambda_reduction import (
    build_df_lambda_reduction_programs,
)
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.cc.lambda_equations import AMPLITUDES, RESIDUALS
from generativeqc_compiler.tensor import execute, transpose_program
from test_df_cc_native_solver import _case
from test_df_lambda_reduction import prepare


@pytest.mark.parametrize("o,v,q,batch", [(1, 3, 5, 2), (2, 3, 5, 3), (3, 2, 3, 8)])
def test_batched_matrix_adjoint_and_factor_rows(
    o: int, v: int, q: int, batch: int
) -> None:
    _, _, feeds = _case(o, v, q)
    rng = np.random.default_rng(1820)
    feeds["t1"] = rng.normal(scale=0.06, size=(o, v))
    two = rng.normal(scale=0.04, size=(o, o, v, v))
    feeds["t2"] = (two + two.transpose(1, 0, 3, 2)) / 2
    seeds = {
        "bar_singles_residual": rng.normal(size=(o, v)),
        "bar_doubles_residual": rng.normal(size=(o, o, v, v)),
    }
    pipeline = build_df_lambda_reduction_programs(o, v)
    prepared, cuts = prepare(pipeline, feeds)
    core = execute(matrix_program(pipeline.core), {**feeds, **cuts, **seeds}).outputs
    totals = {
        "bar_t1": core["bar_t1"].copy(),
        "bar_t2": core["bar_t2"].copy(),
        "bar_df_tau": np.zeros_like(prepared["df_tau"]),
    }
    for start in range(0, q, batch):
        stop = min(q, start + batch)
        frame = {
            **feeds,
            **prepared,
            **core,
            "bov": feeds["bov"][start:stop],
            "bvv": feeds["bvv"][start:stop],
        }
        for name, original in (
            ("auxiliary", pipeline.auxiliary),
            ("factors", pipeline.factors),
            ("primal", pipeline.primal.auxiliary),
        ):
            packed = matrix_program(original, batch_size=stop - start)
            actual = execute(packed, frame).outputs
            expected = [
                execute(
                    original, {**frame, "bov": feeds["bov"][k], "bvv": feeds["bvv"][k]}
                ).outputs
                for k in range(start, stop)
            ]
            for field, values in actual.items():
                varying = packed.outputs[field].spec.indices[0].space.kind == "batch"
                reference = (
                    np.stack([row[field] for row in expected])
                    if varying
                    else expected[0][field]
                )
                np.testing.assert_allclose(values, reference, atol=2e-11, rtol=0)
                if name == "auxiliary":
                    totals[field] += (
                        values.sum(axis=0) if varying else (stop - start) * values
                    )
            assert all(
                sum(i.space.kind == "virtual" for i in n.spec.indices) <= 2
                for n in packed.live_nodes
            )
    reversed_prepare = execute(
        matrix_program(pipeline.prepare), {**feeds, "bar_df_tau": totals["bar_df_tau"]}
    ).outputs
    expanded = transpose_program(
        build_ccsd_program(o, v, form="expanded", diagnostics=False),
        RESIDUALS,
        inputs=AMPLITUDES,
    ).program
    expected = execute(expanded, {**feeds, **seeds}).outputs
    for name in expected:
        np.testing.assert_allclose(
            totals[name] + reversed_prepare[name], expected[name], atol=2e-11, rtol=0
        )


def test_runtime_batch_queries_and_gemm_coverage() -> None:
    from tools.generate_df_lambda import header, matrix_programs
    from tools.generate_rccsd_native import (
        _packed_batched_matrix_gemm,
        _packed_matrix_gemm,
    )

    programs = matrix_programs()
    assert (
        sum(
            _packed_matrix_gemm(n) is not None
            for n in programs["staged_core"].live_nodes
        )
        > 100
    )
    assert (
        sum(
            _packed_batched_matrix_gemm(n) is not None
            for n in programs["staged_auxiliary"].live_nodes
        )
        > 0
    )
    generated = header()
    assert (
        "staged_auxiliary_matrix_arena_elements(std::size_t o,std::size_t v,std::size_t q)"
        in generated
    )
    assert "staged_matrix_operator_hash" in generated


def test_lambda_matrix_actions_avoid_redundant_operand_transposes() -> None:
    """Protect reduced seed traffic without changing the independent actions."""
    from tools.generate_df_lambda import matrix_programs
    from tools.generate_rccsd_native import (
        _packed_batched_matrix_gemm,
        _packed_matrix_gemm,
    )

    programs = matrix_programs()
    # The old NN-only layouts used 148/37/64 materialized permutations here.
    for name, ceiling in (
        ("staged_core", 120),
        ("staged_auxiliary", 30),
        ("staged_factors", 53),
    ):
        program = programs[name]
        assert sum(node.op == "transpose" for node in program.live_nodes) <= ceiling
        recipes = [
            _packed_matrix_gemm(node) or _packed_batched_matrix_gemm(node)
            for node in program.live_nodes
        ]
        assert any(recipe is not None and "T" in recipe[:2] for recipe in recipes)

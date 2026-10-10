"""Prove the factored ladder's operator, bounded work, and unchanged other cuts."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.cc.df_hoist import build_df_auxiliary_reduction_programs
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_spectator_pairs import (
    LADDER_OUTPUT,
    _prove_ladder_pair_symmetry,
    factor_ladder_dressing,
    fold_occupied_ladder_pairs,
)
from generativeqc_compiler.tensor import Program, add, einsum, execute
from test_df_cc_native_solver import _case
from test_df_cc_spectator_pairs import _summands


@pytest.mark.parametrize("occupied,virtuals", [(1, 3), (2, 3), (3, 2), (4, 5)])
def test_factored_ladder_matches_original_for_arbitrary_tau(
    occupied: int, virtuals: int
) -> None:
    original = build_df_auxiliary_reduction_programs(occupied, virtuals).auxiliary
    factored = factor_ladder_dressing(original)
    assert factored is not original
    assert factored.provenance["df_ladder_dressing_factorization"]
    assert all(
        factored.outputs[name] is node
        for name, node in original.outputs.items()
        if name != LADDER_OUTPUT
    )
    _, _, feeds = _case(occupied, virtuals, 2)
    random = np.random.default_rng(2247 + occupied)
    inputs = {
        **feeds,
        "bov": feeds["bov"][0],
        "bvv": feeds["bvv"][0],
        "df_tau": random.normal(scale=0.2, size=feeds["t2"].shape),
    }
    expected = execute(original, inputs).outputs
    for program in (factored, pack_df_contractions(factored)):
        actual = execute(program, inputs).outputs
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, atol=2e-12, rtol=0)


def test_dummy_index_space_order_does_not_reject_an_equivalent_factorization() -> None:
    """An occupied dummy introduced inside D must not acquire a virtual ordinal."""
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    inputs = {
        node.attrs["name"]: node for node in original.live_nodes if node.op == "input"
    }
    dressing = einsum("kc,ka->ac", inputs["bov"], inputs["t1"])
    dressed = add(inputs["bvv"], dressing, coefficients=(1, -1))
    root = add(
        einsum("ac,ijcd,bd->ijab", dressed, inputs["df_tau"], inputs["bvv"]),
        einsum(
            "ac,ijcd,kd,kb->ijab",
            inputs["bvv"],
            inputs["df_tau"],
            inputs["bov"],
            inputs["t1"],
            coefficient=-1,
        ),
    )
    _prove_ladder_pair_symmetry(root, reference=original.outputs[LADDER_OUTPUT])


def test_range_guard_is_needed_even_when_the_original_ladder_is_finite() -> None:
    """A premature D product can overflow although the old tau-first tree is zero."""
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    factored = factor_ladder_dressing(original)
    inputs = {
        "df_tau": np.zeros((2, 2, 3, 3)),
        "t1": np.zeros((2, 3)),
        "bov": np.zeros((2, 3)),
        "bvv": np.zeros((3, 3)),
    }
    inputs["t1"][0, 0] = 1e140
    inputs["bov"][0, 1] = 1e180
    expected = execute(
        Program({LADDER_OUTPUT: original.outputs[LADDER_OUTPUT]}), inputs
    ).outputs[LADDER_OUTPUT]
    assert np.array_equal(expected, np.zeros_like(expected))
    with pytest.raises(ValueError, match="non-finite"):
        execute(Program({LADDER_OUTPUT: factored.outputs[LADDER_OUTPUT]}), inputs)


def test_a_different_symmetric_operator_is_not_an_equivalence_proof() -> None:
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    root = original.outputs[LADDER_OUTPUT]
    changed = add(root, coefficients=(2,))
    _prove_ladder_pair_symmetry(changed)
    with pytest.raises(ValueError, match="differs from its original polynomial"):
        _prove_ladder_pair_symmetry(changed, reference=root)
    unsupported = Program({**original.outputs, LADDER_OUTPUT: changed})
    assert factor_ladder_dressing(unsupported) is unsupported


def test_factored_native_tree_reduces_work_instead_of_introducing_v4_loops() -> None:
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    factored = factor_ladder_dressing(original)
    before = pack_df_contractions(fold_occupied_ladder_pairs(original))
    folded = fold_occupied_ladder_pairs(factored)
    for candidate in (
        pack_df_contractions(folded),
        matrix_program(folded, batch_size=3),
    ):
        assert _summands(candidate, 9, 221) == 1054175083
        assert _summands(before, 9, 221) - _summands(candidate, 9, 221) == 58902246
        assert all(
            len(node.inputs) <= 2
            for node in candidate.live_nodes
            if node.op == "einsum"
        )
        assert all(
            sum(index.space.kind == "virtual" for index in node.spec.indices) <= 2
            for node in candidate.live_nodes
        )

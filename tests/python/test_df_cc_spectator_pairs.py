"""Free-pair folding must preserve the original ladder and other primary cuts."""

from __future__ import annotations

from math import prod

import numpy as np
import pytest
from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.cc.df_hoist import build_df_auxiliary_reduction_programs
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_spectator_pairs import (
    LADDER_OUTPUT,
    PAIRED_TAU_INPUT,
    build_ladder_pair_majorant,
    fold_occupied_ladder_pairs,
)
from generativeqc_compiler.tensor import Program, execute
from generativeqc_compiler.tensor.ir import add, einsum, input_tensor, transpose
from test_df_cc_native_solver import _case

from tools.generativeqc_cc.df_factorized import virtual_corrections
from tools.generativeqc_cc.oracle import DeterminantOracle


def _expand_ladder(packed: np.ndarray, nocc: int) -> np.ndarray:
    nvir = packed.shape[-1]
    full = np.empty((nocc, nocc, nvir, nvir))
    position = 0
    for first in range(nocc):
        for second in range(first, nocc):
            full[first, second] = packed[position]
            if first != second:
                full[second, first] = packed[position].T
            position += 1
    return full


@pytest.mark.parametrize("nocc,nvir,naux", [(1, 3, 2), (2, 3, 3), (3, 2, 4)])
def test_free_occupied_pair_fold_preserves_all_auxiliary_outputs(
    nocc: int, nvir: int, naux: int
) -> None:
    _, _, feeds = _case(nocc, nvir, naux)
    generator = np.random.default_rng(2184 + nocc)
    doubles = generator.normal(scale=0.08, size=feeds["t2"].shape)
    feeds["t2"] = (doubles + doubles.transpose(1, 0, 3, 2)) / 2
    pipeline = build_df_auxiliary_reduction_programs(nocc, nvir)
    prepared = execute(pipeline.prepare, feeds).outputs
    tau = prepared["df_tau"]
    pairs = np.array(
        [tau[first, second] for first in range(nocc) for second in range(first, nocc)]
    )
    original = pipeline.auxiliary
    folded = fold_occupied_ladder_pairs(original)
    assert all(
        node is folded.outputs[name]
        for name, node in original.outputs.items()
        if name != LADDER_OUTPUT
    )
    for auxiliary in range(naux):
        inputs = {
            **feeds,
            **prepared,
            PAIRED_TAU_INPUT: pairs,
            "bov": feeds["bov"][auxiliary],
            "bvv": feeds["bvv"][auxiliary],
        }
        expected = execute(original, inputs).outputs
        actual = execute(pack_df_contractions(folded), inputs).outputs
        for name, reference in expected.items():
            value = (
                _expand_ladder(actual[name], nocc)
                if name == LADDER_OUTPUT
                else actual[name]
            )
            np.testing.assert_allclose(value, reference, atol=2e-12, rtol=0)
    batched = matrix_program(folded, batch_size=naux)
    actual_batch = execute(
        batched, {**feeds, **prepared, PAIRED_TAU_INPUT: pairs}
    ).outputs
    for auxiliary in range(naux):
        inputs = {
            **feeds,
            **prepared,
            "bov": feeds["bov"][auxiliary],
            "bvv": feeds["bvv"][auxiliary],
        }
        expected = execute(original, inputs).outputs
        for name, reference in expected.items():
            value = actual_batch[name][auxiliary]
            if name == LADDER_OUTPUT:
                value = _expand_ladder(value, nocc)
            np.testing.assert_allclose(value, reference, atol=2e-12, rtol=0)


def _summands(program: Program, nocc: int, nvir: int, naux: int = 1) -> int:
    dimensions = {
        "occupied": nocc,
        "virtual": nvir,
        "pair": nocc * (nocc + 1) // 2,
        "batch": naux,
    }
    return sum(
        prod(
            {
                label: dimensions[index.space.kind]
                for child, labels in zip(node.inputs, node.attrs["labels"], strict=True)
                for label, index in zip(labels, child.spec.indices, strict=True)
            }.values()
        )
        for node in program.live_nodes
        if node.op == "einsum"
    )


def test_fold_reduces_physical_ladder_work_without_three_virtual_axes() -> None:
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    folded = pack_df_contractions(fold_occupied_ladder_pairs(original))
    assert all(
        sum(index.space.kind == "virtual" for index in node.spec.indices) <= 2
        for node in folded.live_nodes
    )
    assert _summands(folded, 9, 221) < 0.65 * _summands(original, 9, 221)
    assert _summands(folded, 21, 243) < 0.6 * _summands(original, 21, 243)


@pytest.mark.parametrize("nocc,nvir", [(2, 3), (3, 2)])
def test_derived_majorant_bounds_arbitrary_tau_pair_projection(
    nocc: int, nvir: int
) -> None:
    """The norm proof covers asymmetric tau; native admission chooses its limit."""
    _, _, feeds = _case(nocc, nvir, 3)
    generator = np.random.default_rng(1917 + nocc)
    tau = generator.normal(scale=0.3, size=feeds["t2"].shape)
    projected = (tau + tau.transpose(1, 0, 3, 2)) / 2
    delta = np.max(np.abs(tau - projected))
    pairs = np.array(
        [
            projected[first, second]
            for first in range(nocc)
            for second in range(first, nocc)
        ]
    )
    original = build_df_auxiliary_reduction_programs(nocc, nvir).auxiliary
    folded = fold_occupied_ladder_pairs(original)
    majorant = build_ladder_pair_majorant(original)
    assert majorant.amplitude_norm[1] == (0,)
    amplitude_norm = np.max(np.sum(np.abs(feeds["t1"]), axis=0))
    for auxiliary in range(3):
        inputs = {
            **feeds,
            "df_tau": tau,
            PAIRED_TAU_INPUT: pairs,
            "bov": feeds["bov"][auxiliary],
            "bvv": feeds["bvv"][auxiliary],
        }
        norms = {
            name: np.array(np.max(np.sum(np.abs(inputs[source]), axis=axes)))
            for name, source, axes in majorant.factor_norms
        }
        coefficients = execute(majorant.coefficients, norms).outputs
        limit = delta * (
            coefficients["coefficient_0"]
            + coefficients["coefficient_1"] * amplitude_norm
        )
        expected = execute(original, inputs).outputs[LADDER_OUTPUT]
        actual = _expand_ladder(execute(folded, inputs).outputs[LADDER_OUTPUT], nocc)
        assert np.max(np.abs(actual - expected)) <= limit * (1 + 1e-12)


def _evaluate_folded(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute every auxiliary cut, reconstruct, and consume the complete core."""
    nocc, nvir = feeds["t1"].shape
    pipeline = build_df_auxiliary_reduction_programs(nocc, nvir)
    folded = fold_occupied_ladder_pairs(pipeline.auxiliary)
    prepared = execute(pipeline.prepare, feeds).outputs
    tau = prepared["df_tau"]
    projected = (tau + tau.transpose(1, 0, 3, 2)) / 2
    pairs = np.array(
        [
            projected[first, second]
            for first in range(nocc)
            for second in range(first, nocc)
        ]
    )
    reduced = {
        name: np.zeros(node.spec.shape)
        for name, node in pipeline.auxiliary.outputs.items()
    }
    for auxiliary in range(feeds["bov"].shape[0]):
        row = execute(
            pack_df_contractions(folded),
            {
                **feeds,
                **prepared,
                PAIRED_TAU_INPUT: pairs,
                "bov": feeds["bov"][auxiliary],
                "bvv": feeds["bvv"][auxiliary],
            },
        ).outputs
        for name, value in row.items():
            reduced[name] += (
                _expand_ladder(value, nocc) if name == LADDER_OUTPUT else value
            )
    return execute(pipeline.core, {**feeds, **reduced}).outputs


@pytest.mark.parametrize("nocc,nvir,naux", [(1, 3, 2), (2, 3, 4), (3, 2, 3)])
def test_folded_complete_core_matches_independent_determinants_and_numpy(
    nocc: int, nvir: int, naux: int
) -> None:
    fock, integrals, feeds = _case(nocc, nvir, naux)
    generator = np.random.default_rng(2191 + nocc)
    feeds["t1"] = generator.normal(scale=0.06, size=feeds["t1"].shape)
    doubles = generator.normal(scale=0.08, size=feeds["t2"].shape)
    feeds["t2"] = (doubles + doubles.transpose(1, 0, 3, 2)) / 2
    actual = _evaluate_folded(feeds)
    reference = DeterminantOracle(fock, integrals, nocc).evaluate_full(
        feeds["t1"], feeds["t2"]
    )
    for name, expected in zip(
        ("correlation_energy", "singles_residual", "doubles_residual"),
        reference,
        strict=True,
    ):
        np.testing.assert_allclose(actual[name], expected, atol=2e-12, rtol=0)
    zeroed = {
        name: value if name in ("t1", "t2", "bov", "bvv") else np.zeros_like(value)
        for name, value in feeds.items()
    }
    virtual = _evaluate_folded(zeroed)
    singles, doubles = virtual_corrections(
        feeds["bov"], feeds["bvv"], feeds["t1"], feeds["t2"]
    )
    np.testing.assert_allclose(virtual["singles_residual"], singles, atol=2e-12, rtol=0)
    np.testing.assert_allclose(virtual["doubles_residual"], doubles, atol=2e-12, rtol=0)


def test_fold_requires_output_symmetry_not_just_free_spectators() -> None:
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    inputs = {
        node.attrs["name"]: node for node in original.live_nodes if node.op == "input"
    }
    one_sided = einsum("ijac,cb->ijab", inputs["df_tau"], inputs["bvv"])
    with pytest.raises(ValueError, match="does not preserve simultaneous"):
        fold_occupied_ladder_pairs(Program({LADDER_OUTPUT: one_sided}))
    # The negative T1 dressings must remain signed in the exact proof. Losing
    # one dressing cannot be hidden by Counter's positive-only subtraction.
    root = original.outputs[LADDER_OUTPUT].inputs[0]
    incomplete = add(*root.inputs[:2])
    with pytest.raises(ValueError, match="does not preserve simultaneous"):
        fold_occupied_ladder_pairs(Program({LADDER_OUTPUT: incomplete}))


def test_fold_fails_closed_for_nonfree_spectators_and_unproven_sources() -> None:
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    inputs = {
        node.attrs["name"]: node for node in original.live_nodes if node.op == "input"
    }
    shared = einsum("ijab,ia->ijab", inputs["df_tau"], inputs["t1"])
    with pytest.raises(ValueError, match="free in one operand"):
        fold_occupied_ladder_pairs(Program({LADDER_OUTPUT: shared}))
    unknown = input_tensor("unproven_tau", inputs["df_tau"].spec)
    with pytest.raises(ValueError, match="tau-only spectator source"):
        fold_occupied_ladder_pairs(Program({LADDER_OUTPUT: unknown}))
    nonlinear = einsum("ijac,ijcb->ijab", inputs["df_tau"], inputs["df_tau"])
    with pytest.raises(ValueError, match="exactly one tau operand"):
        build_ladder_pair_majorant(Program({LADDER_OUTPUT: nonlinear}))


def test_fold_tracks_reversed_occupied_axes_through_transpose() -> None:
    _, _, feeds = _case(2, 3, 1)
    pipeline = build_df_auxiliary_reduction_programs(2, 3)
    prepared = execute(pipeline.prepare, feeds).outputs
    root = transpose(pipeline.auxiliary.outputs[LADDER_OUTPUT], (1, 0, 3, 2))
    original = Program({LADDER_OUTPUT: root})
    folded = fold_occupied_ladder_pairs(original)
    tau = prepared["df_tau"]
    pairs = np.array([tau[first, second] for first, second in ((0, 0), (0, 1), (1, 1))])
    inputs = {
        **feeds,
        **prepared,
        PAIRED_TAU_INPUT: pairs,
        "bov": feeds["bov"][0],
        "bvv": feeds["bvv"][0],
    }
    actual = _expand_ladder(execute(folded, inputs).outputs[LADDER_OUTPUT], 2)
    expected = execute(original, inputs).outputs[LADDER_OUTPUT]
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


def test_symbolic_reflection_proof_has_a_finite_expansion_budget() -> None:
    original = build_df_auxiliary_reduction_programs(2, 3).auxiliary
    oversized = add(*(original.outputs[LADDER_OUTPUT] for _ in range(65)))
    with pytest.raises(ValueError, match="term limit"):
        fold_occupied_ladder_pairs(Program({LADDER_OUTPUT: oversized}))

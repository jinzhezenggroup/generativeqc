"""Budget-visible DF matrix layouts against the independent full residual."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.cc.df_hoist import build_df_auxiliary_reduction_programs
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    einsum,
    execute,
    input_tensor,
    transpose,
)
from test_df_cc_auxiliary_reduction import _contraction_terms
from test_df_cc_native_solver import _case

from tools.generate_rccsd_native import (
    _packed_batched_matrix_gemm,
    _packed_matrix_gemm,
)


def _operand(name: str, labels: str, dimensions: dict[str, int]) -> Node:
    """Distinct typed axes prevent equal extents from implying symmetry."""
    return input_tensor(
        name,
        TensorSpec(
            tuple(
                Index(label, IndexSpace(label, "occupied", dimensions[label]))
                for label in labels
            ),
            role="input",
        ),
    )


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("left_transposed", [False, True])
@pytest.mark.parametrize("right_transposed", [False, True])
def test_direct_matrix_transposes_need_no_packing(
    batch: bool, left_transposed: bool, right_transposed: bool
) -> None:
    dimensions = {"q": 3, "i": 2, "j": 5, "k": 7}
    prefix = "q" if batch else ""
    left = prefix + ("ki" if left_transposed else "ik")
    right = prefix + ("jk" if right_transposed else "kj")
    expression = f"{left},{right}->{prefix}ij"
    left_operand = _operand("left", left, dimensions)
    right_operand = _operand("right", right, dimensions)
    source = Program(
        {
            "product": einsum(
                expression,
                left_operand,
                right_operand,
                coefficient=Fraction(-3, 4),
            )
        }
    )
    packed = pack_df_contractions(source, allow_batch=batch)
    assert not any(node.op == "transpose" for node in packed.live_nodes)
    recipe = (_packed_batched_matrix_gemm if batch else _packed_matrix_gemm)(
        packed.outputs["product"]
    )
    assert recipe is not None
    assert recipe[:2] == (
        "T" if left_transposed else "N",
        "T" if right_transposed else "N",
    )
    rng = np.random.default_rng(2101)
    feeds = {
        "left": rng.normal(size=left_operand.spec.shape),
        "right": rng.normal(size=right_operand.spec.shape),
    }
    np.testing.assert_allclose(
        execute(packed, feeds).outputs["product"],
        -0.75 * np.einsum(expression, feeds["left"], feeds["right"]),
        atol=2e-13,
        rtol=0,
    )


@pytest.mark.parametrize("nested", [False, True])
def test_ad_permutation_views_fold_into_grouped_matrix_labels(nested: bool) -> None:
    dimensions = {"a": 2, "b": 3, "c": 7, "k": 5}
    left = _operand("left", "kab", dimensions)
    right = _operand("right", "ck", dimensions)
    view = (
        transpose(transpose(left, (2, 0, 1)), (2, 0, 1))
        if nested
        else transpose(left, (1, 2, 0))
    )
    source = Program({"product": einsum("abk,kc->abc", view, transpose(right, (1, 0)))})
    packed = pack_df_contractions(source)
    assert not any(node.op == "transpose" for node in packed.live_nodes)
    assert _packed_matrix_gemm(packed.outputs["product"])[:2] == ("T", "T")
    rng = np.random.default_rng(2102)
    feeds = {
        "left": rng.normal(size=left.spec.shape),
        "right": rng.normal(size=right.spec.shape),
    }
    np.testing.assert_allclose(
        execute(packed, feeds).outputs["product"],
        np.einsum("kab,ck->abc", feeds["left"], feeds["right"]),
        atol=2e-13,
        rtol=0,
    )


@pytest.mark.parametrize(
    "expression,allow_batch",
    [("ik,jk->ij", False), ("ik,ki->i", False), ("qik,qkj->qij", False)],
)
def test_shared_views_survive_contraction_packing(
    expression: str, allow_batch: bool
) -> None:
    dimensions = {"q": 3, "i": 2, "j": 5, "k": 2}
    left_labels, right_labels = expression.split("->")[0].split(",")
    owner = _operand("left", left_labels[::-1], dimensions)
    view = transpose(owner, tuple(reversed(range(len(left_labels)))))
    right = _operand("right", right_labels, dimensions)
    source = Program({"product": einsum(expression, view, right), "view": view})
    packed = pack_df_contractions(source, allow_batch=allow_batch)
    rng = np.random.default_rng(2103)
    feeds = {
        "left": rng.normal(size=owner.spec.shape),
        "right": rng.normal(size=right.spec.shape),
    }
    actual, expected = execute(packed, feeds).outputs, execute(source, feeds).outputs
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-13, rtol=0)
    assert packed.outputs["view"].op == "transpose"


def test_interleaved_batches_and_reduction_axes_still_require_packing() -> None:
    dimensions = {"q": 3, "i": 2, "j": 5, "k": 2, "l": 2}
    left = _operand("left", "kiql", dimensions)
    right = _operand("right", "qlkj", dimensions)
    source = Program({"product": einsum("kiql,qlkj->qij", left, right)})
    packed = pack_df_contractions(source, allow_batch=True)
    assert sum(node.op == "transpose" for node in packed.live_nodes) == 2
    assert _packed_batched_matrix_gemm(packed.outputs["product"])[:2] == ("N", "N")
    rng = np.random.default_rng(2104)
    feeds = {
        "left": rng.normal(size=left.spec.shape),
        "right": rng.normal(size=right.spec.shape),
    }
    np.testing.assert_allclose(
        execute(packed, feeds).outputs["product"],
        np.einsum("kiql,qlkj->qij", feeds["left"], feeds["right"]),
        atol=2e-13,
        rtol=0,
    )


@pytest.mark.parametrize("o,v,q", [(1, 1, 2), (1, 3, 2), (2, 3, 4), (3, 2, 3)])
def test_packed_df_residual_matches_full_equations(o: int, v: int, q: int) -> None:
    _, _, feeds = _case(o, v, q)
    rng = np.random.default_rng(1817)
    feeds["t1"] = rng.normal(scale=0.07, size=(o, v))
    t2 = rng.normal(scale=0.05, size=(o, o, v, v))
    feeds["t2"] = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    pipeline = build_df_auxiliary_reduction_programs(o, v)
    packed = {
        name: pack_df_contractions(getattr(pipeline, name))
        for name in ("prepare", "auxiliary", "core")
    }
    prepared = execute(packed["prepare"], feeds).outputs
    cuts = {
        name: np.zeros(value.spec.shape)
        for name, value in packed["auxiliary"].outputs.items()
    }
    for ov, vv in zip(feeds["bov"], feeds["bvv"], strict=True):
        values = {**feeds, **prepared, "bov": ov, "bvv": vv}
        row = execute(packed["auxiliary"], values).outputs
        scalar = execute(pipeline.auxiliary, values).outputs
        for name in cuts:
            np.testing.assert_allclose(row[name], scalar[name], atol=2e-12, rtol=0)
            cuts[name] += row[name]
    actual = execute(packed["core"], {**feeds, **cuts}).outputs
    expected = execute(
        build_ccsd_program(o, v, form="expanded", diagnostics=False), feeds
    ).outputs
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-12, rtol=0)


def test_packing_preserves_runtime_work_and_bounded_tensor_ranks() -> None:
    pipeline = build_df_auxiliary_reduction_programs(2, 3)
    for name in ("prepare", "auxiliary", "core"):
        source = getattr(pipeline, name)
        packed = pack_df_contractions(source)
        for o, v in ((2, 3), (9, 221), (21, 243)):
            # Shared value numbering may reuse newly identical packed values;
            # it cannot increase the semantic contraction work.
            assert _contraction_terms(packed, o, v) <= _contraction_terms(source, o, v)
        assert all(
            sum(i.space.kind == "virtual" for i in n.spec.indices) <= 2
            for n in packed.live_nodes
        )
        if name != "prepare":
            assert (
                sum(_packed_matrix_gemm(n) is not None for n in packed.live_nodes) > 10
            )
        assert packed.logical_hash == pack_df_contractions(source).logical_hash

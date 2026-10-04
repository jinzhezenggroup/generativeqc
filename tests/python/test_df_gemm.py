"""Budget-visible DF matrix layouts against the independent full residual."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.cc.df_hoist import build_df_auxiliary_reduction_programs
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.tensor import execute
from test_df_cc_auxiliary_reduction import _contraction_terms
from test_df_cc_native_solver import _case

from tools.generate_rccsd_native import _packed_matrix_gemm


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

"""The packed audit must keep the original expanded, factor-dependent graph."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.cc.df_equations import build_df_virtual_response_programs
from generativeqc_compiler.cc.df_gemm import pack_df_contractions
from generativeqc_compiler.tensor import execute, prepare_for_backend
from test_df_cc_auxiliary_reduction import _contraction_terms
from test_df_cc_native_solver import _case

from tools.generate_df_ccsd_native import packed_replay_program, programs
from tools.generativeqc_cc.df_factorized import virtual_corrections


@pytest.mark.parametrize("nocc,nvir", [(1, 3), (2, 3), (3, 2)])
def test_original_replay_packing_matches_independent_virtual_equations(
    nocc: int, nvir: int
) -> None:
    _, _, arrays = _case(nocc, nvir, 3)
    original = prepare_for_backend(
        build_df_virtual_response_programs(nocc, nvir).primal,
        "cuda",
        preserve_reduction_order=True,
    )
    packed = pack_df_contractions(original)
    for index in range(3):
        feeds = {**arrays, "bov": arrays["bov"][index], "bvv": arrays["bvv"][index]}
        expected = execute(original, feeds).outputs
        actual = execute(packed, feeds).outputs
        reference = virtual_corrections(
            arrays["bov"][index : index + 1],
            arrays["bvv"][index : index + 1],
            arrays["t1"],
            arrays["t2"],
        )
        for name, independent in zip(
            ("df_virtual_singles", "df_virtual_doubles"), reference, strict=True
        ):
            np.testing.assert_allclose(actual[name], expected[name], atol=2e-12, rtol=0)
            np.testing.assert_allclose(actual[name], independent, atol=2e-12, rtol=0)


def test_replay_uses_original_factors_and_accounts_for_common_contraction() -> None:
    """Packing deduplicates one o*v*v contraction, without primal hoisting."""
    original = programs("cuda")["virtual"]
    packed = packed_replay_program()
    assert {node.attrs["name"] for node in packed.live_nodes if node.op == "input"} == {
        "bov",
        "bvv",
        "t1",
        "t2",
    }
    assert all(
        sum(index.space.kind == "virtual" for index in node.spec.indices) <= 2
        for node in packed.live_nodes
    )
    for nocc, nvir in ((2, 3), (9, 221), (21, 243)):
        assert (
            _contraction_terms(original, nocc, nvir)
            - _contraction_terms(packed, nocc, nvir)
            == nocc * nvir * nvir
        )

"""Ratchet retired compiler compatibility exports."""

from __future__ import annotations

from generativeqc_compiler import integral
from generativeqc_compiler.integral import cuda_lowering

RETIRED_EXPORTS = (
    "DpppFusedPlan",
    "build_dppp_fused_plan",
    "dppp_components",
    "emit_dppp_fused_cuda",
    "evaluate_dppp_fused_component",
    "emit_ppps_1110_resident_bra_cuda",
)


def test_retired_dppp_and_ppps_1110_exports_do_not_return() -> None:
    for module in (integral, cuda_lowering):
        for name in RETIRED_EXPORTS:
            assert not hasattr(module, name)
            assert name not in getattr(module, "__all__", ())

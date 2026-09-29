"""Retirement guards for Direct primitive-pair cache and force-result layouts."""

import json
from pathlib import Path

from generativeqc_compiler.integral.direct_pair_cache_cuda import (
    emit_direct_pair_cache_header,
)

ROOT = Path(__file__).resolve().parents[2]


def test_direct_pair_cache_math_is_compiler_owned() -> None:
    source = emit_direct_pair_cache_header()
    assert source.startswith("#pragma once\n")
    assert "__global__ void build_shell_primitive_pair_cache_kernel" in source
    assert "const double reduced_exponent = alpha * beta / exponent_sum;" in source
    assert "product_center(alpha, first, beta, second)" in source
    assert "alpha / exponent_sum" in source
    runtime = (ROOT / "src/scf/cuda/direct_pair_cache.cu").read_text(encoding="utf-8")
    assert '#include "generated_direct_pair_cache.cuh"' in runtime
    assert "reduced_exponent = alpha * beta / exponent_sum" not in runtime
    assert "launch_build_shell_primitive_pair_cache_kernel" in runtime


def test_direct_gradient_types_are_runtime_layout_only() -> None:
    assert not (ROOT / "src/scf/cuda/direct_native_gradient_types.cuh").exists()
    layout = ROOT / "src/scf/cuda/direct_gradient_types.cuh"
    assert layout.exists()
    source = layout.read_text(encoding="utf-8")
    assert "struct PsssWeightedGradient" in source
    assert "struct CartesianQuartetGradient" in source
    ownership = json.loads(
        (
            ROOT
            / "docs/cuda_ownership/files/src/scf/cuda/direct_gradient_types.cuh.json"
        ).read_text(encoding="utf-8")
    )
    assert ownership["role"] == "runtime"


def test_shared_native_recurrence_family_is_closed() -> None:
    retirement = json.loads(
        (ROOT / "docs/cuda_ownership/direct_hf_retirement.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        family["id"] != "shared-native-recurrence-support"
        for family in retirement["families"]
    )

"""Retirement guards for compiler-owned Direct order-two shell contraction."""

from pathlib import Path

from vibeqc_compiler.integral.direct_order2_shell_cuda import (
    emit_direct_order2_shell_header,
)

ROOT = Path(__file__).resolve().parents[2]


def test_direct_order2_shell_is_compiler_owned() -> None:
    source = emit_direct_order2_shell_header()
    assert source.startswith("#pragma once\n")
    assert (
        "Generated from the compiler-owned Direct order-two shell lowering." in source
    )
    assert "Order2IntegralVector" in source
    assert "contracted_eri_cartesian_source_order2_shell" in source
    assert "generated_direct_eri_order2.cuh" in source


def test_native_order2_shell_owner_is_retired() -> None:
    assert not (ROOT / "src/scf/cuda/direct_native_order2_shell.cuh").exists()
    fock = (ROOT / "src/scf/cuda/direct_fock_order2.cuh").read_text(encoding="utf-8")
    source_contraction = (
        ROOT / "python/vibeqc_compiler/integral/direct_source_contraction_cuda.py"
    ).read_text(encoding="utf-8")
    assert '#include "generated_direct_order2_shell.cuh"' in fock
    assert '#include "generated_direct_order2_shell.cuh"' in source_contraction
    assert "direct_native_order2_shell.cuh" not in fock
    assert "direct_native_order2_shell.cuh" not in source_contraction


def test_direct_order2_shell_generation_is_registered() -> None:
    generated = (ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    assert "VIBEQC_DIRECT_ORDER2_SHELL_HEADER" in generated
    assert "generate_direct_order2_shell.py" in generated
    assert cuda.count("VIBEQC_DIRECT_ORDER2_SHELL_HEADER") == 2

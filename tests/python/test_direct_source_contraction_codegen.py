"""Retirement guards for compiler-owned Direct source contraction."""

from pathlib import Path

from vibeqc_compiler.integral.direct_source_contraction_cuda import (
    emit_direct_source_contraction_header,
)

ROOT = Path(__file__).resolve().parents[2]


def test_direct_source_contraction_is_compiler_owned() -> None:
    source = emit_direct_source_contraction_header()
    assert source.startswith("#pragma once\n")
    assert (
        "Generated from the compiler-owned Direct source-contraction lowering."
        in source
    )
    assert "contracted_eri_cartesian_source_shell_class(" in source
    assert "dispatch_contracted_eri_cartesian_source_shell_class" in source
    assert "weight * primitive_eri_cartesian_shell_class" in source
    assert "direct_native_source_contraction.cuh" not in source


def test_native_source_contraction_owner_is_retired() -> None:
    native = ROOT / "src/scf/cuda/direct_native_source_contraction.cuh"
    assert not native.exists()
    for relative in (
        "src/scf/cuda/direct_fock_quartet.cuh",
        "src/scf/cuda/direct_force_quartet.cuh",
        "src/scf/cuda/direct_schwarz_kernels.cu",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '#include "generated_direct_source_contraction.cuh"' in source
        assert "direct_native_source_contraction.cuh" not in source


def test_source_contraction_generation_is_registered() -> None:
    generated = (ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    assert "VIBEQC_DIRECT_SOURCE_CONTRACTION_HEADER" in generated
    assert "generate_direct_source_contraction.py" in generated
    assert cuda.count("VIBEQC_DIRECT_SOURCE_CONTRACTION_HEADER") == 2

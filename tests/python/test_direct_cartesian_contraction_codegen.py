"""Retirement guards for compiler-owned Direct Cartesian/contraction support."""

from pathlib import Path

from vibeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)

ROOT = Path(__file__).resolve().parents[2]


def test_direct_cartesian_contraction_is_compiler_owned() -> None:
    headers = emit_direct_cartesian_contraction_headers()
    assert set(headers) == {
        "generated_direct_cartesian.cuh",
        "generated_direct_contraction.cuh",
    }
    cartesian = headers["generated_direct_cartesian.cuh"]
    contraction = headers["generated_direct_contraction.cuh"]
    assert "primitive_eri(" in cartesian
    assert "primitive_eri_cartesian(" in cartesian
    assert "primitive_eri_psss(" in cartesian
    assert '#include "generated_direct_cartesian.cuh"' in contraction
    assert "contracted_eri_cartesian(" in contraction
    assert "contracted_eri_order(" in contraction


def test_native_cartesian_contraction_owners_are_retired() -> None:
    for name in ("direct_native_cartesian.cuh", "direct_native_contraction.cuh"):
        assert not (ROOT / "src/scf/cuda" / name).exists()


def test_cartesian_and_contraction_consumers_use_generated_headers() -> None:
    cartesian_consumers = (
        "src/scf/cuda/weighted_eri_kernels.cu",
        "python/vibeqc_compiler/integral/direct_recurrence_cuda.py",
        "python/vibeqc_compiler/integral/direct_source_contraction_cuda.py",
    )
    for relative in cartesian_consumers:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "generated_direct_cartesian.cuh" in source
        assert "direct_native_cartesian.cuh" not in source

    contraction_consumers = (
        "src/scf/cuda/direct_jk_kernels.cu",
        "src/scf/cuda/direct_cached_tensor_kernels.cu",
        "src/scf/cuda/direct_reference_force.cu",
        "src/scf/cuda/direct_packed_fock_kernels.cu",
    )
    for relative in contraction_consumers:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '#include "generated_direct_contraction.cuh"' in source
        assert "direct_native_contraction.cuh" not in source


def test_direct_cartesian_contraction_generation_is_registered() -> None:
    generated = (ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    assert "VIBEQC_DIRECT_CARTESIAN_CONTRACTION_HEADERS" in generated
    assert "generate_direct_cartesian_contraction.py" in generated
    assert cuda.count("VIBEQC_DIRECT_CARTESIAN_CONTRACTION_HEADERS") == 2

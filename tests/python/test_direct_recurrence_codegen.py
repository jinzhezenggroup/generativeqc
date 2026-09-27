"""Retirement guards for compiler-owned Direct primitive recurrence support."""

from pathlib import Path

from vibeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)

ROOT = Path(__file__).resolve().parents[2]


def test_direct_recurrence_headers_are_compiler_owned() -> None:
    headers = emit_direct_recurrence_headers()
    assert set(headers) == {
        "generated_direct_eri_order2.cuh",
        "generated_direct_eri_order3.cuh",
        "generated_direct_eri_order4.cuh",
        "generated_direct_shell_class.cuh",
    }
    assert "primitive_eri_order2(" in headers["generated_direct_eri_order2.cuh"]
    assert "primitive_eri_order3(" in headers["generated_direct_eri_order3.cuh"]
    assert (
        "make_fourth_order_pair_expansion" in headers["generated_direct_eri_order4.cuh"]
    )
    shell = headers["generated_direct_shell_class.cuh"]
    assert "primitive_eri_cartesian_shell_class(" in shell
    assert '#include "generated_direct_eri_order2.cuh"' in shell
    assert '#include "generated_direct_eri_order3.cuh"' in shell
    assert '#include "generated_direct_eri_order4.cuh"' in shell


def test_native_recurrence_owners_are_retired() -> None:
    for name in (
        "direct_native_eri_order2.cuh",
        "direct_native_eri_order3.cuh",
        "direct_native_eri_order4.cuh",
        "direct_native_shell_class.cuh",
    ):
        assert not (ROOT / "src/scf/cuda" / name).exists()


def test_remaining_consumers_use_generated_recurrence() -> None:
    order2_shell = (ROOT / "src/scf/cuda/direct_native_order2_shell.cuh").read_text(
        encoding="utf-8"
    )
    pair3 = (ROOT / "src/scf/cuda/direct_native_pair_order3.cuh").read_text(
        encoding="utf-8"
    )
    pair_gradient = (
        ROOT / "python/vibeqc_compiler/integral/direct_pair_gradient_cuda.py"
    ).read_text(encoding="utf-8")
    source_contraction = (
        ROOT / "python/vibeqc_compiler/integral/direct_source_contraction_cuda.py"
    ).read_text(encoding="utf-8")
    assert '#include "generated_direct_eri_order2.cuh"' in order2_shell
    assert '#include "generated_direct_eri_order2.cuh"' in pair3
    assert '#include "generated_direct_eri_order4.cuh"' in pair_gradient
    assert '#include "generated_direct_shell_class.cuh"' in source_contraction


def test_direct_recurrence_generation_is_registered() -> None:
    generated = (ROOT / "cmake/VibeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    assert "VIBEQC_DIRECT_RECURRENCE_HEADERS" in generated
    assert "generate_direct_recurrence.py" in generated
    assert cuda.count("VIBEQC_DIRECT_RECURRENCE_HEADERS") == 2

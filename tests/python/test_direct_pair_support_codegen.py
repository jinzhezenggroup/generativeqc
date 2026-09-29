"""Retirement guards for compiler-owned Direct pair/Hermite support."""

from pathlib import Path

from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)

ROOT = Path(__file__).resolve().parents[2]


def test_direct_pair_support_is_compiler_owned() -> None:
    headers = emit_direct_pair_support_headers()
    assert set(headers) == {
        "generated_direct_pair_order2.cuh",
        "generated_direct_pair_order3.cuh",
        "generated_direct_shell_pair_hermite.cuh",
    }
    assert "LowOrderHermiteTerm" in headers["generated_direct_pair_order2.cuh"]
    assert (
        "make_low_order_pair_expansion" in headers["generated_direct_pair_order2.cuh"]
    )
    assert "ThirdOrderPairExpansion" in headers["generated_direct_pair_order3.cuh"]
    assert (
        "make_third_order_pair_expansion" in headers["generated_direct_pair_order3.cuh"]
    )
    assert (
        "ShellPairHermiteCoefficients"
        in headers["generated_direct_shell_pair_hermite.cuh"]
    )
    assert (
        "fill_shell_pair_hermite" in headers["generated_direct_shell_pair_hermite.cuh"]
    )


def test_native_pair_support_owners_are_retired() -> None:
    for name in (
        "direct_native_pair_order2.cuh",
        "direct_native_pair_order3.cuh",
        "direct_native_shell_pair_hermite.cuh",
    ):
        assert not (ROOT / "src/scf/cuda" / name).exists()


def test_generated_recurrence_consumes_generated_pair_support() -> None:
    joined = "\n".join(emit_direct_recurrence_headers().values())
    assert "direct_native_pair_order2.cuh" not in joined
    assert "direct_native_pair_order3.cuh" not in joined
    assert "direct_native_shell_pair_hermite.cuh" not in joined
    assert "generated_direct_pair_order2.cuh" in joined
    assert "generated_direct_pair_order3.cuh" in joined
    assert "generated_direct_shell_pair_hermite.cuh" in joined


def test_direct_pair_support_generation_is_registered() -> None:
    generated = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text(encoding="utf-8")
    assert "GENERATIVEQC_DIRECT_PAIR_SUPPORT_HEADERS" in generated
    assert "generate_direct_pair_support.py" in generated
    assert cuda.count("GENERATIVEQC_DIRECT_PAIR_SUPPORT_HEADERS") == 2

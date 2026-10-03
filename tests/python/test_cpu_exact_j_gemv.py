"""Regression guard for the CPU exact restricted J-only BLAS route."""

from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "scf" / "fock_build.cpp"
).read_text(encoding="utf-8")


def test_restricted_j_only_uses_shared_cpu_gemv() -> None:
    assert '#include "tensor/cpu_linalg.hpp"' in SOURCE
    assert (
        "strategy.spec.coulomb.present && !strategy.spec.exchange.present && "
        "!unrestricted"
    ) in SOURCE
    assert "tensor::cpu_gemv('N', count, count, eri.data(), density.data()" in SOURCE


def test_scalar_jk_fallback_remains_for_exchange_and_uhf() -> None:
    gemv = SOURCE.index("tensor::cpu_gemv")
    fallback = SOURCE.index("if (unrestricted)", gemv)
    assert "return result;" in SOURCE[gemv:fallback]
    body = SOURCE[fallback : SOURCE.index("FockMatrices assemble_fock", fallback)]
    # Restricted J-only alone takes GEMV; every exchange/UHF combination still
    # dispatches to the shared scalar owner after that fast path returns.
    expected = (
        "true, true, false",
        "true, false, true",
        "true, true, true",
        "false, false, true",
        "false, true, true",
    )
    assert body.count("contract_exact_source_major<") == len(expected)
    for flags in expected:
        assert (
            f"contract_exact_source_major<{flags}>(nbf, eri, density, beta, result)"
        ) in body

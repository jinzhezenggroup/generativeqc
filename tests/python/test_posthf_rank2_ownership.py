"""Keep production post-HF rank-2 basis transforms on the shared GEMM owner."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _section(path: str, begin: str, end: str) -> str:
    source = (ROOT / path).read_text()
    return source.split(begin, 1)[1].split(end, 1)[0]


def test_mp2_derivative_rank2_pullbacks_use_shared_owner() -> None:
    source = (ROOT / "src/posthf/mp2_derivative_common.cpp").read_text()
    assert "pullback_matrix(" not in source
    assert source.count("posthf::rank2_mo_to_ao(") == 2


def test_mp2_hcore_transform_uses_shared_owner() -> None:
    body = _section(
        "src/posthf/mp2_force.cpp",
        "std::vector<double> hcore_mo(",
        "EnergyAdjoint energy_adjoint(",
    )
    assert "posthf::rank2_ao_to_mo(" in body
    assert "for (std::size_t" not in body


def test_cc_raw_hcore_transform_uses_shared_owner() -> None:
    body = _section(
        "src/cc/rccsdt_force.cpp",
        "RawHamiltonian raw_hamiltonian(",
        "struct ResponseWeights",
    )
    assert "posthf::rank2_ao_to_mo(" in body
    assert "out.h.assign(" not in body


def test_rank2_owner_is_two_gemms_per_direction() -> None:
    source = (ROOT / "src/posthf/rank2_transform.cpp").read_text()
    ao_to_mo = source.split("std::vector<double> rank2_ao_to_mo(", 1)[1].split(
        "std::vector<double> rank2_mo_to_ao(", 1
    )[0]
    mo_to_ao = source.split("std::vector<double> rank2_mo_to_ao(", 1)[1]
    assert ao_to_mo.count("tensor::cpu_gemm(") == 2
    assert mo_to_ao.count("tensor::cpu_gemm(") == 2

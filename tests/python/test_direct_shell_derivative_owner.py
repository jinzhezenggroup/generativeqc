"""Guard the prepared Direct shell derivative owner used by DFT stationary forces."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_generated_exchange_owner_retains_bounded_force_state() -> None:
    header = _source("src/scf/cuda/direct_coulomb.hpp")
    source = _source("src/scf/cuda/direct_coulomb.cpp")
    consumer = _source("src/scf/cuda/direct_jk_kernels.cu")
    for token in (
        "force_capability",
        "bounded_pair_order",
        "shell_pair_block_bounds",
        "force_cursor",
        "execute_generated_full_range_energy_derivatives",
    ):
        assert token in header
        assert token in source
    assert "launch_bounded_shell_energy_derivative(" in source
    assert "direct_bounded_fallback.hpp" not in source
    assert "launch_bounded_direct_shell_quartet_kernel_scaled(" in consumer
    assert "DirectScreeningPurpose::Force" in consumer


def test_retained_direct_plan_prepares_shell_derivative_lease() -> None:
    source = _source("src/scf/cuda/direct_jk.cpp")
    assert "prepare_generated_exchange(" in source
    assert "derivative_order != 0" in source
    assert "execute_cuda_direct_shell_full_range_derivatives_device(" in source
    assert (
        "plan->derivative_order == 0 && plan->generated_exchange != nullptr" in source
    )


def test_prepared_rsh_prefers_shell_native_radial_derivatives() -> None:
    source = _source("src/scf/cuda_fock_execution.cpp")
    begin = source.index("execute_prepared_cuda_direct_rsh_energy_derivatives_device(")
    end = source.index(
        "execute_prepared_cuda_direct_shell_full_range_derivatives_device(", begin
    )
    body = source[begin:end]
    shell = body.index("execute_cuda_direct_shell_rsh_energy_derivatives_device(")
    fallback = body.index("execute_cuda_direct_rsh_energy_derivatives_device(")
    assert shell < fallback
    assert "status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED" in body
    assert "unit_long_range" not in body


def test_range_identity_reaches_generic_cartesian_shell_source() -> None:
    source = _source(
        "python/generativeqc_compiler/integral/direct_source_contraction_cuda.py"
    )
    quartet = _source("src/scf/cuda/direct_force_quartet.cuh")
    bounded = _source("src/scf/cuda/direct_bounded_fallback.cu")
    assert "CoulombRange range" in source
    assert "primitive_eri_cartesian<MaximumAngular>" in source
    assert "derivative_coordinate, range, omega" in source
    assert "radial_range, radial_omega" in quartet
    assert "radial_range != generativeqc::integrals::CoulombRange::Full" in bounded
    assert "radial_range, radial_omega" in bounded


def test_shell_rsh_runs_explicit_full_short_and_long_range_passes() -> None:
    source = _source("src/scf/cuda/direct_coulomb.cpp")
    assert "execute_generated_rsh_energy_derivatives(" in source
    assert "DirectCoulombRange::Full" in source
    assert "DirectCoulombRange::Short" in source
    assert "DirectCoulombRange::Long" in source
    assert (
        "{0.0, short_exchange_coefficient, DirectCoulombRange::Short, omega}" in source
    )
    assert "{0.0, long_exchange_coefficient, DirectCoulombRange::Long, omega}" in source
    assert "long_unit" not in source

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


def test_prepared_rsh_uses_shell_full_range_and_lr_only_reconstruction() -> None:
    source = _source("src/scf/cuda_fock_execution.cpp")
    begin = source.index("execute_prepared_cuda_direct_rsh_energy_derivatives_device(")
    end = source.index(
        "execute_prepared_cuda_direct_shell_full_range_derivatives_device(", begin
    )
    body = source[begin:end]
    assert "execute_cuda_direct_shell_full_range_derivatives_device(" in body
    assert "source, p.spin, 0.0, 0.0, 1.0, c.exchange.omega" in body
    assert "full_range[coordinates + coordinate]" in body
    assert "p.exchange.coefficient * long_unit" in body

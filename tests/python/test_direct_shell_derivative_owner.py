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
        "execute_generated_rsh_energy_derivatives",
    ):
        assert token in header
        assert token in source
    assert "launch_bounded_shell_energy_derivative(" in source
    assert "launch_bounded_shell_rsh_derivatives(" in source
    rsh_begin = source.index("cudaError_t execute_generated_rsh_energy_derivatives(")
    rsh_end = source.index("cudaError_t enqueue_generated_coulomb(", rsh_begin)
    rsh_body = source[rsh_begin:rsh_end]
    assert rsh_body.count("launch_bounded_shell_rsh_derivatives(") == 1
    assert "launch_bounded_shell_range_exchange_derivative(" not in rsh_body
    assert "for (unsigned source" not in rsh_body
    assert "direct_bounded_fallback.hpp" not in source
    assert "launch_bounded_direct_shell_quartet_kernel_scaled(" in consumer
    assert "DirectScreeningPurpose::Force" in consumer


def test_fused_rsh_scratch_budget_matches_owner_allocation() -> None:
    owner = _source("src/scf/cuda/direct_coulomb.cpp")
    capacity = _source("src/scf/cuda/direct_jk.cpp")
    assert "charge(product(atoms, 9), sizeof(double))" in owner
    assert "plan->force = doubles(product(atoms, 9))" in owner
    assert "add(atoms, 9 * sizeof(double))" in capacity


def test_retained_direct_plan_prepares_shell_derivative_lease() -> None:
    source = _source("src/scf/cuda/direct_jk.cpp")
    assert "prepare_generated_exchange(" in source
    assert "derivative_order != 0" in source
    assert "execute_cuda_direct_shell_full_range_derivatives_device(" in source
    assert (
        "plan->derivative_order == 0 && plan->generated_exchange != nullptr" in source
    )


def test_prepared_rsh_uses_shell_sr_lr_scheduler() -> None:
    source = _source("src/scf/cuda_fock_execution.cpp")
    begin = source.index("execute_prepared_cuda_direct_rsh_energy_derivatives_device(")
    end = source.index(
        "execute_prepared_cuda_direct_shell_full_range_derivatives_device(", begin
    )
    body = source[begin:end]
    assert "execute_cuda_direct_shell_rsh_energy_derivatives_device(" in body
    assert "p.exchange.coefficient + c.exchange.coefficient" in body
    assert "c.exchange.omega" in body
    assert "unit_long_range" not in body
    assert "full_range[coordinates + coordinate]" not in body


def test_generic_stationary_cuda_adopts_prepared_shell_jk_before_ao4_fallback() -> None:
    driver = _source("python/generativeqc/_stationary_cuda.py")
    assert "native_shell = state._source.cuda_shell_full_range_derivatives()" in driver
    assert "stationary_full_range_integral_route=(" in driver
    assert '"prepared-direct-shell-v1"' in driver

    begin = driver.index("task_sources = [")
    end = driver.index("for source, rank, operator in task_sources:", begin)
    selection = driver[begin:end]
    assert "if shell_full_range is None:" in selection
    assert 'task_sources.append(("coulomb", 4, "four_center_eri"))' in selection
    assert 'task_sources.append(("exact_exchange", 4, "four_center_eri"))' in selection
    assert '"mode": "native-shell"' in selection
    assert '"logical_tasks": 0' in selection


def test_snapshot_shell_jk_fallback_is_capability_only() -> None:
    snapshot = _source("python/generativeqc/_ks_snapshot.py")
    begin = snapshot.index("def cuda_shell_full_range_derivatives(")
    end = snapshot.index("def ecp_derivatives(", begin)
    body = snapshot[begin:end]
    assert "status == _native.STATUS_NOT_IMPLEMENTED" in body
    assert "_native.check(self._library, status" in body
    assert "return None" in body


def test_native_generic_shell_jk_borrows_resident_final_density() -> None:
    source = _source("src/methods/dft_method.cpp")
    begin = source.index("generativeqc_status cuda_shell_full_range_gradient(")
    end = source.index("Result execute(bool compute_forces)", begin)
    body = source[begin:end]
    assert "resident_final_density(expected, resident_density" in body
    assert "execute_prepared_cuda_direct_shell_full_range_derivatives_device(" in body
    assert "density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE" in body
    assert "range_strategy_" in body

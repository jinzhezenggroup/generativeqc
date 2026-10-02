"""Source contracts for the method-neutral fixed-density CUDA profile (#1480)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_native_profile_is_token_checked_and_does_not_enter_solver() -> None:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    begin = source.index("generativeqc_status CudaKsPlan::profile_fixed_density_components(")
    end = source.index("generativeqc_status CudaKsPlan::read_final_state(", begin)
    body = source[begin:end]
    assert "const auto current = impl_->token()" in body
    assert "execute_cuda_direct_jk_device" in body
    assert "execute_cuda_density_fitting_rhf_jk_device" in body
    assert "enqueue_prepared_cuda_exchange_correction" in body
    assert "enqueue_replay_body" in body
    assert "begin(" not in body
    assert "enqueue_iteration" not in body
    assert "finish_iteration" not in body
    assert "impl_->token() != current" in body


def test_snapshot_and_benchmark_publish_fixed_density_boundary() -> None:
    snapshot = (ROOT / "python/generativeqc/_ks_snapshot.py").read_text()
    matrix = (ROOT / "benchmarks/dft_force_matrix.py").read_text()
    assert "generativeqc_ks_snapshot_cuda_fixed_density_profile_v1" in snapshot
    assert '"measurement_boundary": "fixed_density_scf_components"' in snapshot
    helper = matrix[
        matrix.index("def _fixed_density_scf_component_profile(") :
        matrix.index("def _scf_trace_profile(")
    ]
    assert "NativeKsSnapshot(batch, 0)" in helper
    assert "batch.execute" not in helper
    assert "_fixed_density_scf_component_profile(batch)" in matrix

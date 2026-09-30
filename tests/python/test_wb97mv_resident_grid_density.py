"""Structural guards for resident WB97M-V grid-density handoff."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER = (ROOT / "python/generativeqc/_stationary_wb97mv_cuda.py").read_text()
SNAPSHOT = (ROOT / "python/generativeqc/_ks_snapshot.py").read_text()
GRID = (ROOT / "python/generativeqc_compiler/dft/cuda.py").read_text()
NATIVE_GRID = (ROOT / "src/dft/cuda_grid.cu").read_text()
CUDA_KS = (ROOT / "src/dft/cuda_ks.cpp").read_text()


def test_wb97mv_grid_density_never_roundtrips_through_host() -> None:
    assert "source.cuda_resident_density()" in DRIVER
    assert "self.grid.set_density_device(" in DRIVER
    assert "self.grid.set_density(" not in DRIVER
    assert '"grid_density_h2d_bytes": 0' in DRIVER
    assert '"grid_density_source": "exact-final-scf-device-binding"' in DRIVER


def test_snapshot_resident_density_is_token_checked() -> None:
    begin = SNAPSHOT.index("    def cuda_resident_density(")
    end = SNAPSHOT.index("    def cuda_full_range_derivatives(", begin)
    body = SNAPSHOT[begin:end]
    assert body.count("self.check_current()") == 2
    assert "generativeqc_ks_snapshot_cuda_resident_density_v1" in body
    assert "source_stream" in body


def test_grid_resident_density_uses_device_ordering_only() -> None:
    begin = NATIVE_GRID.index("int grid_cuda_density_device_v1(")
    end = NATIVE_GRID.index("int grid_cuda_source_v1(", begin)
    body = NATIVE_GRID[begin:end]
    assert "cudaMemcpyHostToDevice" not in body
    assert "cudaMemcpyDeviceToDevice" in body
    assert "split_restricted_density<<<" in body
    assert "cudaEventRecord(destination_ready, ctx.stream)" in body
    assert "cudaStreamWaitEvent(producer, destination_ready, 0)" in body
    assert "cudaEventRecord(source_copied, producer)" in body
    assert "cudaStreamWaitEvent(ctx.stream, source_copied, 0)" in body


def test_grid_python_binding_reports_zero_host_upload() -> None:
    begin = GRID.index("    def set_density_device(")
    end = GRID.index("    def set_source(", begin)
    body = GRID[begin:end]
    assert '"grid_cuda_density_device_v1"' in body
    assert '"source_upload_bytes": 0' in body
    assert '"resident_density"' in body


def test_native_density_binding_carries_producer_stream() -> None:
    begin = CUDA_KS.index("generativeqc_status CudaKsPlan::resident_final_density(")
    end = CUDA_KS.index(
        "generativeqc_status CudaKsPlan::resident_final_stationary_weights(", begin
    )
    body = CUDA_KS[begin:end]
    assert "reinterpret_cast<void*>(impl_->stream)" in body

"""Structural guards for resident WB97M-V molecular-grid point reuse."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOIN = (ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py").read_text()
GRID = (ROOT / "python/generativeqc_compiler/dft/cuda.py").read_text()
NATIVE_GRID = (ROOT / "src/dft/cuda_grid.cu").read_text()
SNAPSHOT = (ROOT / "python/generativeqc/_ks_snapshot.py").read_text()


def test_wb97mv_join_uses_resident_point_slices() -> None:
    assert "cuda_resident_grid()" in JOIN
    assert "feature_task_device_points(" in JOIN
    assert "points[begin:end]" not in JOIN
    assert '"grid_point_source": "exact-native-resident-grid"' in JOIN
    assert '"grid_point_h2d_bytes": 0' in JOIN


def test_snapshot_resident_grid_is_token_checked() -> None:
    begin = SNAPSHOT.index("    def cuda_resident_grid(")
    end = SNAPSHOT.index("    def cuda_resident_density(", begin)
    body = SNAPSHOT[begin:end]
    assert body.count("self.check_current()") == 2
    assert "generativeqc_ks_snapshot_cuda_resident_grid_v1" in body


def test_cuda_grid_device_point_route_skips_host_copy() -> None:
    begin = NATIVE_GRID.index("static int grid_cuda_run_selected_impl(")
    end = NATIVE_GRID.index("int grid_cuda_run_selected_v1(", begin)
    body = NATIVE_GRID[begin:end]
    assert "if (!points_on_device && npoint)" in body
    assert "cudaMemcpyAsync(p.points, points" in body
    assert "task_points = points" in body
    assert "p.current_points = task_points" in body
    assert "active, task_points, npoint" in body
    device_begin = NATIVE_GRID.index("int grid_cuda_run_selected_device_deferred_v1(")
    device_end = NATIVE_GRID.index("int grid_cuda_run_v1(", device_begin)
    device = NATIVE_GRID[device_begin:device_end]
    assert "feature_output, jet_output, 1, 1, error, size" in device


def test_borrowed_grid_view_publishes_exact_task_pointer() -> None:
    begin = NATIVE_GRID.index("int grid_cuda_view_v1(")
    end = NATIVE_GRID.index("int grid_cuda_basis_v1(", begin)
    assert "p.current_points" in NATIVE_GRID[begin:end]


def test_python_device_point_lease_never_accepts_host_coordinates() -> None:
    begin = GRID.index("    def feature_task_device_points(")
    end = GRID.index("    def feature_task(", begin)
    body = GRID[begin:end]
    assert '"grid_cuda_run_selected_device_deferred_v1"' in body
    assert "pointer(points)" not in body
    assert "ct.c_void_p(device_points)" in body

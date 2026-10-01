"""Regression guards for stationary CUDA reuse of the native resident grid."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()


def _geometry_block() -> str:
    function = SCHEDULER.index("def _complete_rks_cuda_gradient_diagnostic(")
    begin = SCHEDULER.index("        grid = state.grid", function)
    end = SCHEDULER.index(
        '        with timeline.phase("xc_geometry_drain"):', begin
    )
    return SCHEDULER[begin:end]


def test_generic_stationary_cuda_borrows_native_grid() -> None:
    """PBE/PBE0 force geometry must not restage the current native grid."""
    body = _geometry_block()
    assert 'getattr(state._source, "cuda_resident_grid", None)' in body
    assert "resident_grid.device != device" in body
    assert "resident_grid.point_count != grid_points" in body
    assert "grid_points != na * points_per_atom" in body
    assert "ao.feature_task_device_points(" in body
    assert "sources.geometry_molecular_resident_weights(" in body
    assert "resident_grid.points + 3 * begin * 8" in body
    assert "resident_grid.weights + begin * 8" in body
    assert "resident_grid.atomic_weights + begin * 8" in body


def test_generic_stationary_cuda_reports_zero_production_grid_uploads() -> None:
    """Keep the #1569 production transfer contract explicit and reviewable."""
    body = _geometry_block()
    assert '"grid_owner_source": "implicit-atom-major-index"' in body
    assert '"grid_owner_h2d_bytes": 0' in body
    assert '"grid_point_source": "exact-native-resident-grid"' in body
    assert '"grid_point_h2d_bytes": 0' in body
    assert '"grid_weight_source": "exact-native-resident-grid"' in body
    assert '"grid_weight_h2d_bytes": 0' in body
    assert '"grid_atomic_measure_source": "exact-native-resident-grid"' in body
    assert '"grid_atomic_measure_h2d_bytes": 0' in body


def test_generic_stationary_cuda_keeps_explicit_host_fallback() -> None:
    """Diagnostics without a native lease retain the checked legacy route."""
    body = _geometry_block()
    assert "with ao.feature_task(" in body
    assert "sources.geometry(" in body
    assert '"grid_point_source": "host-grid-points"' in body
    assert '"grid_weight_source": "host-grid-weights"' in body

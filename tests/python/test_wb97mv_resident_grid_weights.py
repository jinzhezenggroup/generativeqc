"""Structural guards for resident stationary molecular-grid weights."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOIN = (ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py").read_text()
SCHEDULER = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
NATIVE = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()


def _native_block(marker: str, next_marker: str) -> str:
    begin = NATIVE.index(marker)
    end = NATIVE.index(next_marker, begin)
    return NATIVE[begin:end]


def test_wb97mv_join_uses_resident_weight_slices() -> None:
    assert "device_weights = resident_grid.weights + begin * 8" in JOIN
    assert "geometry_molecular_resident_weights(" in JOIN
    assert "geometry_external_device_molecular_resident_weights(" in JOIN
    assert '"grid_weight_source": "exact-native-resident-grid"' in JOIN
    assert '"grid_weight_h2d_bytes": 0' in JOIN


def test_semilocal_resident_weight_enqueue_uploads_only_raw_measure() -> None:
    body = _native_block(
        "int stationary_geometry_molecular_resident_weights_enqueue(",
        "int stationary_geometry_molecular_enqueue(",
    )
    assert "upload(*p, p->weights" not in body
    assert "upload(*p, p->raw, raw" in body
    assert "device_weights, p->raw" in body


def test_nonlocal_resident_weight_enqueue_uploads_only_raw_measure() -> None:
    body = _native_block(
        "int stationary_geometry_external_device_molecular_resident_weights_enqueue(",
        "int stationary_geometry_molecular_resident_weights_enqueue(",
    )
    assert "upload(*p, p->weights" not in body
    assert "upload(*p, p->raw, raw" in body
    assert "device_weights, p->raw, external_device" in body


def test_python_scheduler_passes_device_weights_without_host_pointer_conversion() -> None:
    begin = SCHEDULER.index("    def geometry_molecular_resident_weights(")
    end = SCHEDULER.index(
        "    def geometry_external_device_molecular_resident_weights(", begin
    )
    body = SCHEDULER[begin:end]
    assert '"stationary_geometry_molecular_resident_weights_enqueue"' in body
    assert '_ptr(host_weights)' not in body
    assert "device_weights," in body

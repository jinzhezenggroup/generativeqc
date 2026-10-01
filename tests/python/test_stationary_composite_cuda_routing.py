"""Structural guards for compiler-driven composite stationary CUDA routing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = (ROOT / "python/generativeqc/batch.py").read_text()
DRIVER = (ROOT / "python/generativeqc/_stationary_composite_cuda.py").read_text()
COMPILER = (
    ROOT / "python/generativeqc_compiler/method/stationary_cuda.py"
).read_text()


def test_public_cuda_force_routing_has_no_method_name_special_case() -> None:
    begin = BATCH.index("    def _public_dft_cuda_force(")
    end = BATCH.index("\n    def ", begin + 10)
    body = BATCH[begin:end]
    assert "requires_composite_stationary_cuda(state)" in body
    assert "PreparedCompositeStationaryCudaGradient" in body
    assert "wb97m" not in body.lower()


def test_composite_route_is_selected_from_compiler_source_inventory() -> None:
    assert "stationary_external_provider_sources(plan)" in DRIVER
    assert "_COMPOSITE_EXTERNAL_SOURCES" in DRIVER
    assert "source.method_ir" in DRIVER
    assert "_method_name" not in DRIVER
    assert "resolve_method(" not in DRIVER


def test_composite_route_keeps_shared_point_model_for_ordinary_dft() -> None:
    assert "source.nonlocal_density_policy == MOLECULAR_VV10_DENSITY_POLICY" in DRIVER
    assert "else SCF_POINT_MODEL" in DRIVER


def test_composite_driver_inherits_live_functional_code() -> None:
    assert "functional = int(source.functional_code)" in DRIVER
    assert "functional=functional" in DRIVER
    assert "metadata[6] != 4" not in DRIVER
    assert "functional=4" not in DRIVER


def test_compiler_owns_external_provider_inventory() -> None:
    assert "def stationary_external_provider_sources(" in COMPILER
    assert "stationary_runtime_sources(plan)" in COMPILER
    assert '"ecp_local", "ecp_nonlocal"' in COMPILER

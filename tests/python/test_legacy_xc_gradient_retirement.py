"""Prevent the retired standalone XC-gradient CUDA owner from returning."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_xc_gradient_cuda_owner_is_not_a_production_build_input() -> None:
    generated = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    identity = (ROOT / "cmake/GenerativeQCSourceIdentity.json").read_text(
        encoding="utf-8"
    )
    geometry = (ROOT / "python/generativeqc_compiler/xc/geometry_cuda.py").read_text(
        encoding="utf-8"
    )

    assert "GENERATIVEQC_XC_GRADIENT_SOURCE" not in generated
    assert "generate_xc_gradient_cuda.py" not in generated
    assert "generate_xc_gradient_cuda.py" not in identity
    assert "emit_native_geometry_cuda" not in geometry
    assert "geometry_kernel<<<1, workers" not in geometry
    assert not (ROOT / "tools/generate_xc_gradient_cuda.py").exists()
    assert not (ROOT / "docs/cuda_ownership/generated/xc_gradient.json").exists()

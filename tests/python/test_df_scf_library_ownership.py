"""Regression gates for DF SCF dense-library ownership."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIIS_SOURCE = ROOT / "src/scf/cuda/df_scf_diis.cpp"
LIBRARY_SOURCE = ROOT / "src/scf/cuda/df_scf_library.cpp"


def test_df_diis_uses_shared_dense_library_provider() -> None:
    """Keep vendor BLAS submission out of scientific DIIS method code."""
    diis = DIIS_SOURCE.read_text(encoding="utf-8")
    library = LIBRARY_SOURCE.read_text(encoding="utf-8")

    assert "cublas" not in diis.lower()
    assert "scf_gemm_strided(" in diis
    assert "cublasDgemmStridedBatched" in library

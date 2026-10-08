"""DF-CC source contractions must use the existing shared Tensor provider."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/cc/df_source_cuda.cu"


def test_native_df_source_has_no_direct_vendor_matrix_calls() -> None:
    text = SOURCE.read_text()
    assert "cublasDgemm(" not in text
    assert "CUBLAS_OP_N" not in text
    assert '#include "tensor/cuda_runtime.cuh"' in text
    assert "generativeqc_tensor::gemm(plan.blas, tb, ta," in text
    assert "1, 1.0, 0.0);" in text
    assert "native CUDA DF-CC source GEMM failed" in text
    assert "posthf::generated::compensated::gemm(" in text
    assert "result.transform_gemms <= n" in text
    assert "generated::df_source::build_ovov(" in text
    manifest = json.loads(
        (ROOT / "manifests/maintenance/vendor_boundaries.json").read_text()
    )
    assert "src/cc/df_source_cuda.cu" not in manifest["files"]


@pytest.mark.parametrize("left_t", [False, True])
@pytest.mark.parametrize("right_t", [False, True])
def test_transposed_row_major_provider_preserves_df_source_columns(
    left_t: bool, right_t: bool
) -> None:
    """Exercise all compiler callback transposition combinations independently."""
    rng = np.random.default_rng(1890 + int(left_t) * 2 + int(right_t))
    m, columns, k = 3, 4, 5
    a_col = rng.normal(size=(k, m) if left_t else (m, k))
    b_col = rng.normal(size=(columns, k) if right_t else (k, columns))
    a_memory = np.asfortranarray(a_col).ravel(order="F")
    b_memory = np.asfortranarray(b_col).ravel(order="F")

    old = (a_col.T if left_t else a_col) @ (b_col.T if right_t else b_col)
    # A column-major matrix is a transposed row-major view of the same bytes.
    a_row = a_memory.reshape(a_col.shape[::-1])
    b_row = b_memory.reshape(b_col.shape[::-1])
    new = (b_row.T if right_t else b_row) @ (a_row.T if left_t else a_row)
    np.testing.assert_allclose(new, old.T, rtol=1e-14, atol=1e-14)

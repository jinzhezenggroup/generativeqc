"""Guard the CCSD(T) triples matrix submission at the shared Tensor boundary.

The old method-local cublasDgemm call has identical column-major dimensions,
leading dimensions, host alpha/beta values and transpose flags. CUDA numerical
and E+F qualification remain independent from this source-ownership check.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/cc/df_triples_cuda.cu"
VENDOR_INVENTORY = ROOT / "manifests/maintenance/vendor_boundaries.json"


def test_triples_reverse_and_fock_products_use_shared_column_gemm() -> None:
    source = SOURCE.read_text()
    wrapper = source.split("inline void execute_strided_fp64_product(", 1)[1].split(
        "// Pure view packing/scatter.", 1
    )[0]

    assert "cublasDgemm(" not in source
    assert "generativeqc_tensor::ColumnGemmView request{" in wrapper
    assert "generativeqc_tensor::submit_column_gemm(request)" in wrapper
    assert "generativeqc_tensor::blas_check(" in wrapper

    # These fields preserve the previous cublasDgemm submission exactly.
    for binding in (
        ".handle = context.handle",
        ".transpose_left = ta != 'N'",
        ".transpose_right = tb != 'N'",
        ".rows = static_cast<int>(m)",
        ".columns = static_cast<int>(n)",
        ".reduction = static_cast<int>(k)",
        ".alpha = &alpha",
        ".left = a",
        ".left_ld = static_cast<int>(lda)",
        ".right = b",
        ".right_ld = static_cast<int>(ldb)",
        ".beta = &beta",
        ".output = c",
        ".output_ld = static_cast<int>(ldc)",
    ):
        assert binding in wrapper

    # The existing output finite checks and scientific reverse/response paths stay.
    assert source.count("execute_strided_fp64_product(") == 5
    assert "audit_matrix<<<" in source


def test_triples_no_longer_owns_a_direct_vendor_submission() -> None:
    inventory = json.loads(VENDOR_INVENTORY.read_text())
    assert "src/cc/df_triples_cuda.cu" not in inventory["files"]
    # The actual library provider remains in the shared Tensor runtime.
    provider = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    assert "inline cublasStatus_t submit_column_gemm(" in provider
    assert "return cublasDgemm(" in provider

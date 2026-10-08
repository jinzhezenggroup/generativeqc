"""Keep RI-MP2 matrix operations independent from cuBLAS provider names."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/posthf/ri_mp2_cuda.cu"


def test_ri_mp2_uses_shared_tensor_blas_boundary() -> None:
    source = SOURCE.read_text()
    assert '#include "tensor/cuda_runtime.cuh"' in source
    assert "cublasDgemm(" not in source
    assert "cublasDgemmStridedBatched(" not in source
    assert "CUBLAS_OP_" not in source
    assert source.count("generativeqc_tensor::gemm(plan.blas,") == 1
    assert source.count('submit("RI-MP2 ') == 4
    assert "static_cast<std::int64_t>(b_stride), 0," in source
    assert "static_cast<std::int64_t>(a_stride), 0," in source
    assert "source_row_generations" in source
    assert "reduce_ri_mp2_block<<<" in source
    vendor = json.loads(
        (ROOT / "manifests/maintenance/vendor_boundaries.json").read_text()
    )
    assert "src/posthf/ri_mp2_cuda.cu" not in vendor["files"]


@pytest.mark.parametrize("na,n,block,occupied", [(3, 5, 2, 3), (5, 7, 4, 2)])
def test_ao_mo_projection_keeps_column_major_bytes(
    na: int, n: int, block: int, occupied: int
) -> None:
    rng = np.random.default_rng(1890 + na)
    raw_col = rng.normal(size=(na, n))
    virtual_col = rng.normal(size=(n, block))
    expected = raw_col @ virtual_col
    row_lhs = np.asfortranarray(virtual_col).ravel(order="F").reshape(block, n)
    row_rhs = np.asfortranarray(raw_col).ravel(order="F").reshape(n, na)
    np.testing.assert_allclose(row_lhs @ row_rhs, expected.T, atol=1e-14)
    tmp_col = rng.normal(size=(na * block, n))
    occupied_col = rng.normal(size=(n, occupied))
    expected_second = tmp_col @ occupied_col
    row_lhs = np.asfortranarray(occupied_col).ravel(order="F").reshape(occupied, n)
    row_rhs = np.asfortranarray(tmp_col).ravel(order="F").reshape(n, na * block)
    np.testing.assert_allclose(
        row_lhs @ row_rhs, expected_second.T, atol=1e-14
    )


@pytest.mark.parametrize("a_count,b_count,na,batch", [(2, 3, 5, 1), (4, 2, 7, 3)])
def test_direct_and_exchange_broadcast_batch_mappings(
    a_count: int, b_count: int, na: int, batch: int
) -> None:
    rng = np.random.default_rng(1900 + batch)
    first = rng.normal(size=(batch + 1, na, a_count))
    second = rng.normal(size=(batch + 1, na, b_count))
    fixed_a = first[0]
    fixed_b = second[0]
    for j in range(batch):
        # The old cuBLAS call has transpose-left and nontranspose-right,
        # broadcasting its first matrix (stride zero).
        original_direct = fixed_a.T @ second[j]
        original_exchange = fixed_b.T @ first[j]
        # Tensor's row-major view reverses the physical operands, preserving
        # each column-major result's exact packed byte layout.
        new_direct = second[j].T @ fixed_a
        new_exchange = first[j].T @ fixed_b
        np.testing.assert_allclose(new_direct, original_direct.T, atol=1e-14)
        np.testing.assert_allclose(new_exchange, original_exchange.T, atol=1e-14)

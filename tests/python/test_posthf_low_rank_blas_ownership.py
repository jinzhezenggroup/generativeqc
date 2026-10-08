"""Keep low-rank J/K math independent from vendor BLAS selection."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
LOW_RANK = ROOT / "src/posthf/cuda_low_rank.cu"
TENSOR = ROOT / "src/tensor/cuda_runtime.cuh"


def test_low_rank_uses_only_shared_tensor_provider() -> None:
    source = LOW_RANK.read_text()
    provider = TENSOR.read_text()
    manifest = json.loads(
        (ROOT / "manifests/maintenance/vendor_boundaries.json").read_text()
    )
    assert "src/posthf/cuda_low_rank.cu" not in manifest["files"]
    assert "cublasDgemm(" not in source
    assert "cublasDdot(" not in source
    assert "cublasSetPointerMode(" not in source
    assert source.count("dot_to_device(p.context,") == 1
    assert source.count("gemm(p.context, 'N', 'N', n, n, n,") == 2
    assert "dot_to_device(Context& context" in provider
    assert "cublasDdot(context.handle" in provider
    assert provider.index("const auto restore_status = cublasSetPointerMode(") < (
        provider.index("blas_check(dot_status);")
    )


@pytest.mark.parametrize("size", [2, 3, 5])
def test_old_column_major_and_shared_row_major_jk_are_same(
    size: int,
) -> None:
    rng = np.random.default_rng(size + 1890)
    # Do not rely on symmetric operands: prove the transpose/storage identity
    # that maps the original cuBLAS column-major calls into row-major GEMMs.
    physical = rng.normal(size=(size, size))
    density = rng.normal(size=(size, size))
    old_stage_col = physical.T @ density.T
    old_exchange_col = old_stage_col @ physical.T

    new_stage_row = density @ physical
    new_exchange_row = physical @ new_stage_row

    np.testing.assert_allclose(new_stage_row, old_stage_col.T, atol=1e-14)
    np.testing.assert_allclose(new_exchange_row, old_exchange_col.T, atol=1e-14)
    # The second GEMM uses beta=1: existing output is accumulated, not reset.
    initial = rng.normal(size=(size, size))
    np.testing.assert_allclose(
        initial + new_exchange_row,
        (initial.T + old_exchange_col).T,
        atol=1e-14,
    )


_CPP_PREFIX = r"""
#include <cassert>
#include <stdexcept>
using cublasStatus_t = int;
constexpr int CUBLAS_POINTER_MODE_HOST = 0;
constexpr int CUBLAS_POINTER_MODE_DEVICE = 1;
struct Context { int handle = 9; };
int mode = CUBLAS_POINTER_MODE_HOST;
int device_sets = 0;
int host_sets = 0;
int dot_failure = 0;
int cublasSetPointerMode(int handle, int next) {
  assert(handle == 9);
  mode = next;
  if (next == CUBLAS_POINTER_MODE_DEVICE) ++device_sets;
  else ++host_sets;
  return 0;
}
int cublasDdot(int handle, int n, const double* a, int ia,
               const double* b, int ib, double* out) {
  assert(handle == 9 && mode == CUBLAS_POINTER_MODE_DEVICE);
  if (dot_failure) return dot_failure;
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += a[i * ia] * b[i * ib];
  *out = sum;
  return 0;
}
void blas_check(int status) {
  if (status) throw std::runtime_error("injected BLAS error");
}
"""

_CPP_MAIN = r"""
int main() {
  Context context;
  double a[]{1, 2, 3}, b[]{4, -5, 6}, result = -99;
  dot_to_device(context, 3, a, b, &result);
  assert(result == 12 && mode == CUBLAS_POINTER_MODE_HOST);
  assert(device_sets == 1 && host_sets == 1);
  dot_failure = 5;
  bool rejected = false;
  try { dot_to_device(context, 3, a, b, &result); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected && result == 12);
  assert(mode == CUBLAS_POINTER_MODE_HOST);
  assert(device_sets == 2 && host_sets == 2);
}
"""


def test_device_dot_restores_host_pointer_mode_on_error(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler not available")
    source = TENSOR.read_text()
    start = source.index("inline void dot_to_device(")
    finish = source.index("\n}\n", start) + 2
    # Compile the *actual production helper* against narrow host stubs.
    probe = tmp_path / "dot_pointer_mode.cpp"
    executable = tmp_path / "dot_pointer_mode"
    probe.write_text(_CPP_PREFIX + source[start:finish] + _CPP_MAIN)
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(probe), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=15)

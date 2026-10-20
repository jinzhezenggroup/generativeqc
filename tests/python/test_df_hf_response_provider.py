"""Execute emitted charge helpers through the actual borrowed Tensor provider.

Only the vendor API is a recording numerical host double. This is host evidence,
not CUDA, resource-allocation or complete-endpoint qualification.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc_compiler.method.df_hf_response_cuda import emit_df_hf_response_cuda

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def charge_provider(tmp_path_factory: Any, required_native_cxx: Any) -> Any:
    directory = tmp_path_factory.mktemp("df-hf-response-provider")
    generated = emit_df_hf_response_cuda()
    provider = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    # Add only a route witness to the real forwarder. Its operands, status and
    # vendor submission remain the production implementation.
    submit = cpp_function_definition(provider, "submit_column_gemm")
    submit = submit.replace("{", "{ ++provider_calls;", 1)
    source = r"""
#include <cstddef>
using cublasHandle_t = void*;
using cublasStatus_t = int;
constexpr int CUBLAS_OP_N=0, CUBLAS_OP_T=1;
static int provider_calls=0, failure=0;
static long long recorded[9]{};
static void* expected_handle=nullptr;
static const double* expected_left=nullptr;
static const double* expected_right=nullptr;
static double* expected_output=nullptr;
int cublasDgemm(void* handle, int ta, int tb, int m, int n, int k,
               const double* alpha, const double* a, int lda,
               const double* b, int ldb, const double* beta, double* c, int ldc) {
  recorded[0]=ta; recorded[1]=tb; recorded[2]=m; recorded[3]=n;
  recorded[4]=k; recorded[5]=lda; recorded[6]=ldb; recorded[7]=ldc;
  recorded[8]=(handle == expected_handle && *alpha == 1.0 && *beta == 0.0
               && a == expected_left && b == expected_right && c == expected_output);
  if(failure) return failure;
  if(m<1 || n<1 || k<1 || lda<(ta?k:m) || ldb<(tb?n:k) || ldc<m) return 7;
  for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
    long double sum=0;
    for(int q=0;q<k;++q)
      sum += static_cast<long double>(a[ta?q+i*lda:i+q*lda])
           * b[tb?j+q*ldb:q+j*ldb];
    c[i+j*ldc]=*alpha*sum+(*beta?*beta*c[i+j*ldc]:0);
  }
  return 0;
}
"""
    source += (
        "namespace generativeqc_tensor {\n"
        + cpp_record_definition(provider, "ColumnGemmView")
        + ";\n"
        + submit
        + "\n}\n"
    )
    for name in ("df_rhf_charge_contract", "df_rhf_retained_charge_contract"):
        source += cpp_function_definition(generated, name) + "\n"
    source += r"""
extern "C" int run(int retained, int matrix, int auxiliary, int begin, int count,
                    int terms, const double* left, const double* right,
                    double* output, int status, long long* snapshot) {
  // A borrowed owner with observable stream/modes/workspace/lifetime sentinels.
  long long owner[5]={31,47,59,71,83};
  expected_handle=owner; failure=status; provider_calls=0;
  expected_left=left; expected_right=right;
  expected_output=retained ? output : output+begin;
  int result=retained
    ? df_rhf_retained_charge_contract(owner,matrix,auxiliary,left,right,output)
    : df_rhf_charge_contract(owner,matrix,auxiliary,begin,count,terms,right,left,output);
  for(int i=0;i<9;++i) snapshot[i]=recorded[i];
  snapshot[9]=provider_calls;
  for(int i=0;i<5;++i) snapshot[10+i]=owner[i];
  return result;
}
"""
    probe = directory / "probe.cpp"
    probe.write_text(source)
    library = required_native_cxx.build_shared(
        [probe], directory / "probe.so", compile_args=("-std=c++17", "-O0")
    )
    dll = ctypes.CDLL(str(library))
    run = dll.run
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    snapshot = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    run.argtypes = [ctypes.c_int] * 6 + [array] * 3 + [ctypes.c_int, snapshot]
    run.restype = ctypes.c_int
    return run


def _check_route(snapshot: np.ndarray, expected: list[int]) -> None:
    np.testing.assert_array_equal(snapshot[:8], expected)
    # One shared-provider route, borrowed handle, host alpha=1/beta=0 and no
    # mutation of the owner's stream, modes, workspace or lifetime sentinels.
    np.testing.assert_array_equal(snapshot[8:], [1, 1, 31, 47, 59, 71, 83])


@pytest.mark.parametrize(
    "matrix,auxiliary,begin,count,terms",
    [
        (1, 1, 0, 1, 1),
        (9, 11, 2, 5, 3),
        (4, 9, 7, 2, 4),
        (7, 8, 0, 8, 2),
    ],
)
def test_panel_charge_preserves_offsets_padding_and_terms(
    charge_provider: Any,
    matrix: int,
    auxiliary: int,
    begin: int,
    count: int,
    terms: int,
) -> None:
    rng = np.random.default_rng(1890 + matrix + count)
    fitted = np.ascontiguousarray(rng.normal(size=(count, matrix)))
    densities = np.ascontiguousarray(rng.normal(size=(terms, matrix)))
    before = fitted.copy(), densities.copy()
    output = np.full((terms, auxiliary), np.nan)
    snapshot = np.zeros(15, dtype=np.int64)
    assert (
        charge_provider(
            0,
            matrix,
            auxiliary,
            begin,
            count,
            terms,
            fitted,
            densities,
            output,
            0,
            snapshot,
        )
        == 0
    )
    expected = densities.astype(np.longdouble) @ fitted.astype(np.longdouble).T
    np.testing.assert_allclose(
        output[:, begin : begin + count], expected, atol=3e-14, rtol=3e-14
    )
    assert np.isnan(output[:, :begin]).all()
    assert np.isnan(output[:, begin + count :]).all()
    np.testing.assert_array_equal(fitted, before[0])
    np.testing.assert_array_equal(densities, before[1])
    _check_route(snapshot, [1, 0, count, terms, matrix, matrix, matrix, auxiliary])


@pytest.mark.parametrize("pairs,auxiliary", [(1, 1), (9, 7), (6, 11), (5, 5)])
def test_retained_nonsymmetric_factor_and_metric_root_route(
    charge_provider: Any, pairs: int, auxiliary: int
) -> None:
    rng = np.random.default_rng(pairs + 1890)
    # Also test square nonsymmetric factors: a mistaken transpose must not hide
    # behind the symmetry of physical metric roots in the existing FD suite.
    factor = np.ascontiguousarray(rng.normal(size=(pairs, auxiliary)))
    density = np.ascontiguousarray(rng.normal(size=pairs))
    output = np.full(auxiliary + 3, np.nan)
    snapshot = np.zeros(15, dtype=np.int64)
    assert (
        charge_provider(
            1, pairs, auxiliary, 0, 0, 1, factor, density, output, 0, snapshot
        )
        == 0
    )
    expected = density.astype(np.longdouble) @ factor.astype(np.longdouble)
    np.testing.assert_allclose(output[:auxiliary], expected, atol=3e-14, rtol=3e-14)
    assert np.isnan(output[auxiliary:]).all()
    _check_route(snapshot, [0, 0, auxiliary, 1, pairs, auxiliary, pairs, auxiliary])


@pytest.mark.parametrize("retained", [0, 1])
@pytest.mark.parametrize("failure", [1, 3, 7, 13, 14])
def test_raw_provider_failure_survives_without_output_or_owner_mutation(
    charge_provider: Any, retained: int, failure: int
) -> None:
    left, right = np.ones(12), np.ones(12)
    output = np.full(12, 43.0)
    snapshot = np.zeros(15, dtype=np.int64)
    assert (
        charge_provider(retained, 3, 4, 1, 2, 2, left, right, output, failure, snapshot)
        == failure
    )
    np.testing.assert_array_equal(output, np.full(12, 43.0))
    expected = [0, 0, 4, 1, 3, 4, 3, 4] if retained else [1, 0, 2, 2, 3, 3, 3, 4]
    _check_route(snapshot, expected)

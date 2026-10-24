"""Whole emitted occupied API with real adapters and a numerical vendor double.

This is independent host evidence, not CUDA or complete E/F qualification.
"""

from __future__ import annotations

import ctypes as ct
from typing import Any

import numpy as np
import pytest
from _df_occupied_provider_support import column_adapters
from generativeqc_compiler.method.df_occupied_response_cuda import (
    emit_occupied_response_helpers,
)


@pytest.fixture(scope="module")
def occupied_provider(tmp_path_factory: Any, required_native_cxx: Any) -> Any:
    folder = tmp_path_factory.mktemp("occupied-provider")
    source = r"""
#include <cstddef>
#include <cstdint>
#include <limits>
using cublasHandle_t = void*;
using cublasStatus_t = int;
constexpr int CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_FILL_MODE_LOWER=0;
constexpr int CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_INVALID_VALUE=7;
static bool provider_entry=false;
static int calls=0, fail_on=0, failure=0, bypass=0;
static std::int64_t owner[5]={31,47,59,71,83};
static std::int64_t records[2][17]{};
int cublasDgemmStridedBatched(void* handle,int ta,int tb,int m,int n,int k,
    const double* alpha,const double* a,int lda,long long sa,const double* b,
    int ldb,long long sb,const double* beta,double* c,int ldc,long long sc,int batches) {
  if (!provider_entry) ++bypass;
  provider_entry=false;
  auto* row=records[calls++ % 2];
  const std::int64_t data[]={ta,tb,m,n,k,lda,ldb,ldc,sa,sb,sc,batches,
    reinterpret_cast<std::intptr_t>(a),reinterpret_cast<std::intptr_t>(b),
    reinterpret_cast<std::intptr_t>(c),handle==owner,
    alpha && beta && *alpha==1 && *beta==0};
  for(int i=0;i<17;++i) row[i]=data[i];
  if(calls==fail_on) return failure;
  if(m<1 || n<1 || k<1 || lda<(ta?k:m) || ldb<(tb?n:k) || ldc<m) return 7;
  for(int p=0;p<batches;++p) for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
    long double sum=0;
    for(int q=0;q<k;++q)
      sum+=static_cast<long double>(a[p*sa+(ta?q+i*lda:i+q*lda)])
           *b[p*sb+(tb?j+q*ldb:q+j*ldb)];
    c[p*sc+i+j*ldc]=*alpha*sum+(*beta?*beta*c[p*sc+i+j*ldc]:0);
  }
  return 0;
}
int cublasDgemm(void* handle,int ta,int tb,int m,int n,int k,const double* alpha,
    const double* a,int lda,const double* b,int ldb,const double* beta,double* c,int ldc) {
  return cublasDgemmStridedBatched(handle,ta,tb,m,n,k,alpha,a,lda,0,b,ldb,0,beta,c,ldc,0,1);
}
// The complete emitted API includes the retained Gram/SYRK branch. It is not
// invoked by this probe; its existing independent tests remain authoritative.
int cublasDsyrk(void*,int,int,int,int,const double*,const double*,int,const double*,double*,int) {
  return 13;
}
"""
    source += column_adapters(witness=True) + emit_occupied_response_helpers()
    source += r"""
extern "C" int run(int op,int n,int r,int a,int begin,int count,
    const double* x,const double* y,double* temp,double* out,int site,int status,
    std::int64_t* snapshot) {
  calls=0; fail_on=site; failure=status; bypass=0; provider_entry=false;
  for(auto& row: records) for(auto& value: row) value=0;
  int result=7;
  if(op==0) result=df_occupied_project_panel(owner,n,r,a,begin,count,x,y,temp,out);
  if(op==1) result=df_occupied_finish_projection(owner,n,r,a,x,y,out);
  if(op==2) result=df_occupied_to_metric_eigenbasis(owner,a,r,x,y,out);
  if(op==3) result=df_occupied_apply_metric_root(owner,a,r,x,y,out);
  if(op==4) result=df_occupied_from_metric_eigenbasis(owner,a,r,x,y,out);
  snapshot[0]=calls; snapshot[1]=bypass;
  for(int i=0;i<5;++i) snapshot[2+i]=owner[i];
  for(int j=0;j<2;++j) for(int i=0;i<17;++i) snapshot[7+j*17+i]=records[j][i];
  return result;
}
"""
    probe = folder / "probe.cpp"
    probe.write_text(source)
    output = required_native_cxx.build_shared(
        [probe], folder / "probe.so", compile_args=("-std=c++20", "-O0")
    )
    library = ct.CDLL(str(output))
    run = library.run
    ptr = ct.POINTER(ct.c_double)
    run.argtypes = (
        [ct.c_int] * 6 + [ptr] * 4 + [ct.c_int] * 2 + [ct.POINTER(ct.c_int64)]
    )
    run.restype = ct.c_int

    def invoke(
        op: int,
        n: int,
        r: int,
        a: int,
        x: Any,
        y: Any,
        temp: Any,
        out: Any,
        begin: int = 0,
        count: int = 1,
        site: int = 0,
        status: int = 13,
    ) -> tuple[int, np.ndarray]:
        def pointer(value: Any) -> Any:
            return None if value is None else value.ctypes.data_as(ptr)

        snapshot = np.zeros(41, dtype=np.int64)
        result = run(
            op,
            n,
            r,
            a,
            begin,
            count,
            *map(pointer, (x, y, temp, out)),
            site,
            status,
            snapshot.ctypes.data_as(ct.POINTER(ct.c_int64)),
        )
        np.testing.assert_array_equal(snapshot[2:7], [31, 47, 59, 71, 83])
        return result, snapshot

    return invoke


def _submission(
    row: np.ndarray,
    shape: list[int],
    x: np.ndarray,
    y: np.ndarray,
    out: np.ndarray,
    offset: int = 0,
) -> None:
    np.testing.assert_array_equal(row[:12], shape)
    np.testing.assert_array_equal(
        row[12:15], [x.ctypes.data, y.ctypes.data, out.ctypes.data + 8 * offset]
    )
    np.testing.assert_array_equal(row[15:], [1, 1])


@pytest.mark.parametrize(
    "n,r,a,begin,count",
    [(1, 1, 1, 0, 1), (7, 3, 11, 2, 5), (9, 2, 7, 6, 1), (5, 5, 4, 0, 4)],
)
def test_advanced_panel_broadcast_and_disjoint_output(
    occupied_provider: Any, n: int, r: int, a: int, begin: int, count: int
) -> None:
    rng = np.random.default_rng(1890 + n + r + a)
    c = np.asfortranarray(rng.normal(size=(n, r)))
    # Deliberately nonsymmetric B: transpose mistakes cannot hide in symmetry.
    panels = np.ascontiguousarray(rng.normal(size=(a, n, n)))
    advanced = panels[begin : begin + count]
    temp = np.full(count * n * r + 3, np.nan)
    out = np.full(a * r * r + 3, np.nan)
    before = c.copy(), panels.copy()
    result, snapshot = occupied_provider(
        0, n, r, a, c, advanced, temp, out, begin, count
    )
    assert result == 0
    expected = np.einsum(
        "mi,qnm,nj->qij",
        c.astype(np.longdouble),
        advanced.astype(np.longdouble),
        c.astype(np.longdouble),
    )
    actual = (
        out[begin * r * r : (begin + count) * r * r]
        .reshape(count, r, r)
        .transpose(0, 2, 1)
    )
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
    assert (
        np.isnan(out[: begin * r * r]).all()
        and np.isnan(out[(begin + count) * r * r :]).all()
    )
    assert np.isnan(temp[count * n * r :]).all()
    np.testing.assert_array_equal(c, before[0])
    np.testing.assert_array_equal(panels, before[1])
    _submission(
        snapshot[7:24],
        [0, 0, n, r, n, n, n, n, n * n, 0, n * r, count],
        advanced,
        c,
        temp,
    )
    _submission(
        snapshot[24:],
        [1, 0, r, r, n, n, n, r, 0, n * r, r * r, count],
        c,
        temp,
        out,
        begin * r * r,
    )


@pytest.mark.parametrize("op", [1, 2, 3, 4])
@pytest.mark.parametrize("n,r,a", [(1, 1, 1), (7, 3, 11), (5, 2, 4)])
def test_ordinary_rectangular_layout_and_independent_oracle(
    occupied_provider: Any, op: int, n: int, r: int, a: int
) -> None:
    rng = np.random.default_rng(1890 + op + n + r + a)
    if op == 1:
        x = np.asfortranarray(rng.normal(size=(n, r)))
        y = np.asfortranarray(rng.normal(size=(a * r, n)))
        expected = y.astype(np.longdouble) @ x.astype(np.longdouble)
        shape = [0, 0, a * r, r, n, a * r, n, a * r, 0, 0, 0, 1]
        left, right = y, x
        argument = r
    else:
        rr = r * r
        x = np.asfortranarray(rng.normal(size=(a, a)))
        y = np.asfortranarray(rng.normal(size=(rr, a) if op != 4 else (a, rr)))
        e, s = x.astype(np.longdouble), y.astype(np.longdouble)
        expected = e.T @ s.T if op == 2 else (s @ e if op == 3 else s.T @ e.T)
        shape = (
            [1, 1, a, rr, a, a, rr, a, 0, 0, 0, 1]
            if op == 2
            else [0, 0, rr, a, a, rr, a, rr, 0, 0, 0, 1]
            if op == 3
            else [1, 1, rr, a, a, a, a, rr, 0, 0, 0, 1]
        )
        left, right = (x, y) if op == 2 else (y, x)
        argument = rr
    before = x.copy(), y.copy()
    out = np.full(expected.size + 5, np.nan)
    result, snapshot = occupied_provider(op, n, argument, a, x, y, None, out)
    assert result == 0
    np.testing.assert_allclose(
        out[: expected.size].reshape(expected.shape, order="F"),
        expected,
        atol=3e-13,
        rtol=3e-13,
    )
    assert np.isnan(out[expected.size :]).all()
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(y, before[1])
    _submission(snapshot[7:24], shape, left, right, out)


@pytest.mark.parametrize("op,site", [(0, 1), (0, 2), (1, 1), (2, 1), (3, 1), (4, 1)])
@pytest.mark.parametrize("status", [1, 3, 7, 13, 14])
def test_raw_error_stops_publication_and_preserves_owner(
    occupied_provider: Any, op: int, site: int, status: int
) -> None:
    x, y = np.ones(128), np.ones(128)
    temp, out = np.full(128, 43.0), np.full(128, 59.0)
    result, snapshot = occupied_provider(
        op, 5, 2, 7, x, y, temp, out, 2, 3, site, status
    )
    assert result == status
    np.testing.assert_array_equal(out, np.full(128, 59.0))
    if site == 1:
        np.testing.assert_array_equal(temp, np.full(128, 43.0))
    else:
        assert (temp[:30] == 5).all() and (temp[30:] == 43).all()
    assert snapshot[0] == site


@pytest.mark.parametrize("r,begin,count", [(0, 0, 3), (0, 7, 0), (2, 7, 0)])
def test_empty_projection_needs_no_buffers_or_submission(
    occupied_provider: Any, r: int, begin: int, count: int
) -> None:
    result, snapshot = occupied_provider(
        0, 5, r, 7, None, None, None, None, begin, count
    )
    assert result == 0 and snapshot[0] == 0


@pytest.mark.parametrize(
    "op,n,r,a,begin,count",
    [
        (0, 0, 0, 7, 0, 1),
        (0, 5, -1, 7, 0, 1),
        (0, 5, 6, 7, 0, 1),
        (0, 5, 2, 7, -1, 1),
        (0, 5, 2, 7, 8, 0),
        (0, 5, 2, 7, 6, 2),
        (0, 46341, 1, 7, 0, 1),
        (1, 5, 0, 7, 0, 1),
        (1, 5, 6, 7, 0, 1),
        (1, 5, 2, 2147483647, 0, 1),
        (2, 5, 0, 7, 0, 1),
        (3, 5, 2, 0, 0, 1),
        (4, 5, 2, 0, 0, 1),
    ],
)
def test_invalid_admission_precedes_array_access(
    occupied_provider: Any, op: int, n: int, r: int, a: int, begin: int, count: int
) -> None:
    result, snapshot = occupied_provider(
        op, n, r, a, None, None, None, None, begin, count
    )
    assert result == 7 and snapshot[0] == 0


@pytest.mark.parametrize("op", range(5))
def test_borrowed_shared_submission_boundary(occupied_provider: Any, op: int) -> None:
    x, y, temp, out = (np.ones(128) for _ in range(4))
    result, snapshot = occupied_provider(op, 5, 2, 7, x, y, temp, out, 2, 3)
    assert result == 0
    assert snapshot[0] > 0 and snapshot[1] == 0


def test_metric_root_rejects_identical_storage(occupied_provider: Any) -> None:
    x, out = np.ones(16), np.full(16, 59.0)
    result, snapshot = occupied_provider(3, 2, 4, 4, x, out, None, out)
    assert result == 7 and snapshot[0] == 0
    np.testing.assert_array_equal(out, np.full(16, 59.0))


@pytest.mark.parametrize(
    "op,missing",
    [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 1),
        (4, 3),
    ],
)
def test_nonempty_contract_requires_each_borrowed_buffer(
    occupied_provider: Any, op: int, missing: int
) -> None:
    arrays = [np.full(128, 59.0) for _ in range(4)]
    arrays[missing] = None
    result, snapshot = occupied_provider(op, 5, 2, 7, *arrays, 2, 3)
    assert result == 7 and snapshot[0] == 0
    for array in arrays:
        if array is not None:
            np.testing.assert_array_equal(array, np.full(128, 59.0))

"""Host-check emitted retained-charge layout and independent Coulomb adjoints.

The BLAS provider is emulated; this does not qualify CUDA execution or timing.
"""

from __future__ import annotations

import ctypes
import typing

import numpy as np
import pytest
from generativeqc_compiler.method.df_hf_response_cuda import emit_df_hf_response_cuda


@pytest.fixture(scope="module")
def retained_contract(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: typing.Any
) -> typing.Any:
    source = emit_df_hf_response_cuda()
    start = source.index("inline cublasStatus_t df_rhf_retained_charge_contract(")
    end = source.index("\n}", start) + 2
    directory = tmp_path_factory.mktemp("retained-coulomb")
    probe = directory / "probe.cpp"
    probe.write_text(
        r"""
#include <cstddef>
using cublasHandle_t = void*;
using cublasStatus_t = int;
constexpr int CUBLAS_OP_N=0, CUBLAS_OP_T=1;
int cublasDgemm(void*, int ta, int tb, int m, int n, int k,
               const double* alpha, const double* a, int lda,
               const double* b, int ldb, const double* beta, double* c, int ldc) {
  if (m<1 || n<1 || k<1 || lda<(ta ? k:m) || ldb<(tb ? n:k) || ldc<m)
    return 7;
  for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
    double sum=0;
    for(int q=0;q<k;++q)
      sum += a[ta ? q+i*lda : i+q*lda] * b[tb ? j+q*ldb : q+j*ldb];
    c[i+j*ldc] = *alpha*sum + (*beta ? *beta*c[i+j*ldc] : 0);
  }
  return 0;
}
"""
        + source[start:end]
        + r"""
extern "C" int contract(int pairs, int auxiliary, const double* factor,
                        const double* density, double* output) {
  return df_rhf_retained_charge_contract(nullptr,pairs,auxiliary,factor,density,output);
}
"""
    )
    library = native_cxx.build_shared(
        [probe], directory / "probe.so", compile_args=("-std=c++17", "-O0")
    )
    dll = ctypes.CDLL(str(library))
    function = dll.contract
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int, ctypes.c_int, array, array, array]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("n,a", [(1, 1), (3, 7), (7, 3)])
def test_retained_charge_matches_independent_metric_solve(
    retained_contract: typing.Any, packed: bool, n: int, a: int
) -> None:
    """Exercise actual emitted operands/strides with signed multi-density terms."""
    rng = np.random.default_rng(2170 + n + a)
    rotation = np.linalg.qr(rng.normal(size=(a, a)))[0]
    values = np.geomspace(0.2, 3.0, a)
    metric = (rotation * values) @ rotation.T
    root = np.ascontiguousarray((rotation / np.sqrt(values)) @ rotation.T)
    raw = rng.normal(size=(n, n, a))
    raw = (raw + raw.transpose(1, 0, 2)) / 2
    factor = raw @ root
    rows, columns = np.tril_indices(n)
    retained = np.ascontiguousarray(factor[rows, columns] if packed else factor)
    weights = np.zeros_like(raw)
    metric_weights = np.zeros_like(metric)
    expected_weights = np.zeros_like(raw)
    expected_metric = np.zeros_like(metric)
    terms = []
    for coefficient in (1.0, -0.7, 0.0):
        # Deliberately non-symmetric inputs test both physical off-diagonal entries.
        density = rng.normal(size=(n, n))
        terms.append((coefficient, density))
        folded = density[rows, columns] + np.where(
            rows == columns, 0, density[columns, rows]
        )
        vector = np.ascontiguousarray(folded if packed else density)
        charge = np.full(a, np.nan)
        potential = np.full(a, np.nan)
        pairs = len(rows) if packed else n * n
        assert retained_contract(pairs, a, retained, vector, charge) == 0
        assert retained_contract(a, a, root, charge, potential) == 0
        raw_charge = np.einsum("ij,ijp->p", density, raw)
        expected = np.linalg.solve(metric, raw_charge)
        np.testing.assert_allclose(potential, expected, atol=2e-12, rtol=2e-12)
        weights += coefficient * density[:, :, None] * potential
        metric_weights -= 0.5 * coefficient * np.outer(potential, potential)
        expected_weights += coefficient * density[:, :, None] * expected
        expected_metric -= 0.5 * coefficient * np.outer(expected, expected)
    np.testing.assert_allclose(weights, expected_weights, atol=2e-11, rtol=2e-12)
    np.testing.assert_allclose(metric_weights, expected_metric, atol=2e-10, rtol=2e-12)

    def energy(raw_values: np.ndarray, metric_values: np.ndarray) -> float:
        total = 0.0
        for coefficient, density in terms:
            charge = np.einsum("ij,ijp->p", density, raw_values)
            total += 0.5 * coefficient * charge @ np.linalg.solve(metric_values, charge)
        return float(total)

    direction = rng.normal(size=raw.shape)
    direction = (direction + direction.transpose(1, 0, 2)) / 2
    metric_direction = rng.normal(size=metric.shape)
    metric_direction = (metric_direction + metric_direction.T) / 2
    analytic = np.vdot(weights, direction) + np.vdot(metric_weights, metric_direction)
    step = 1e-5
    finite = (
        energy(raw + step * direction, metric + step * metric_direction)
        - energy(raw - step * direction, metric - step * metric_direction)
    ) / (2 * step)
    np.testing.assert_allclose(analytic, finite, atol=2e-6, rtol=2e-8)


def test_retained_charge_propagates_provider_failure(
    retained_contract: typing.Any,
) -> None:
    value = np.ones(1)
    assert retained_contract(0, 1, value, value, value) == 7

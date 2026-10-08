#pragma once

#include <cublas_v2.h>

#include <cstddef>

namespace generativeqc::tensor::cuda {

/** Already-admitted square FP64 column-major primitives. Borrow the current
 * handle/stream/modes/workspace. No allocation, admission, copies or fences. */
inline cublasStatus_t square_gemm(cublasHandle_t blas, bool transpose_left, int n, int solves,
                                  const double* left, const double* right, double* output) {
  const double one = 1.0, zero = 0.0;
  const auto stride = static_cast<long long>(n) * n;
  return cublasDgemmStridedBatched(blas, transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n,
                                   n, n, &one, left, n, stride, right, n, stride, &zero, output, n,
                                   stride, solves);
}
inline cublasStatus_t square_lower_solve(cublasHandle_t blas, bool right, bool transpose, int n,
                                         int solves, double** factors, double** matrices) {
  const double one = 1.0;
  return cublasDtrsmBatched(
      blas, right ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
      transpose ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &one,
      reinterpret_cast<const double* const*>(factors), n, matrices, n, solves);
}
}  // namespace generativeqc::tensor::cuda

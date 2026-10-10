#pragma once

#include <cublas_v2.h>

#include <cstddef>

namespace generativeqc::tensor::cuda {

/** Already-admitted FP64 product of column-major, square-padded panels.
 * Both leading dimensions and the output extent equal dimension; rank selects
 * the leading reduction coordinates without repacking the resident inputs.
 * Borrow the current handle/stream/modes/workspace. The caller validates all
 * storage and dimensions and handles an empty rank without submitting GEMM. */
inline cublasStatus_t square_panel_product(cublasHandle_t handle, int dimension, int rank,
                                           const double* left, bool transpose_left,
                                           const double* right, bool transpose_right,
                                           double* output) {
  const double one = 1.0, zero = 0.0;
  return cublasDgemm(handle, transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
                     transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N, dimension, dimension, rank, &one,
                     left, dimension, right, dimension, &zero, output, dimension);
}

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

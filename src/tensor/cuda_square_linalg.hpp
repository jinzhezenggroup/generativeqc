#pragma once

#include <cublas_v2.h>

#include <cstddef>

namespace generativeqc::tensor::cuda {

/** Already-admitted FP64 product of column-major, square-padded panels.
 * Both leading dimensions and the output extent equal dimension; rank selects
 * the leading reduction coordinates without repacking the resident inputs.
 * Borrow the current handle/stream/modes/workspace. The caller validates all
 * storage and dimensions and handles an empty rank without submitting GEMM.
 * Alpha and beta are host scalars; the default is an overwrite. */
inline cublasStatus_t square_panel_product(cublasHandle_t handle, int dimension, int rank,
                                           const double* left, bool transpose_left,
                                           const double* right, bool transpose_right,
                                           double* output, double alpha = 1.0, double beta = 0.0) {
  return cublasDgemm(handle, transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
                     transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N, dimension, dimension, rank,
                     &alpha, left, dimension, right, dimension, &beta, output, dimension);
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

/** In-place FP64 solve of L L^T x = b for one column. The private handle
 * normally has no workspace. The caller retains the factor and aligned
 * scratch through stream completion; report any failure restoring the handle
 * separately, since the caller cannot safely continue with that handle. */
inline cublasStatus_t square_lower_vector_solve(cublasHandle_t blas, int order,
                                                const double* factor, double* rhs, void* scratch,
                                                std::size_t scratch_bytes,
                                                cublasStatus_t& restored) noexcept {
  constexpr double one = 1;
  auto status = cublasSetWorkspace(blas, scratch, scratch_bytes);
  const auto solve = [&](cublasOperation_t transpose) {
    return cublasDtrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, transpose,
                       CUBLAS_DIAG_NON_UNIT, order, 1, &one, factor, order, rhs, order);
  };
  if (status == CUBLAS_STATUS_SUCCESS) status = solve(CUBLAS_OP_N);
  if (status == CUBLAS_STATUS_SUCCESS) status = solve(CUBLAS_OP_T);
  restored = cublasSetWorkspace(blas, nullptr, 0);
  return status;
}
}  // namespace generativeqc::tensor::cuda

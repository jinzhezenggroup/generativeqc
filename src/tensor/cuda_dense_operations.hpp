#pragma once

#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::tensor {

/** Prepared FP64 dense operations on an enclosing owner's admitted stream.
 * Preparation preserves host scalar mode and the provider's default math mode.
 * No execution method allocates, synchronizes, changes modes or discovers a
 * provider. The owner must settle work on the correct device before close();
 * explicit cleanup preserves its existing failure and destruction boundaries.
 */
class CudaDenseOperations {
 public:
  CudaDenseOperations() = default;
  CudaDenseOperations(const CudaDenseOperations&) = delete;
  CudaDenseOperations& operator=(const CudaDenseOperations&) = delete;

  /** A partial create/bind remains available for the owner's error cleanup. */
  cublasStatus_t create();
  cublasStatus_t bind(cudaStream_t stream);
  cublasStatus_t close() noexcept;
  explicit operator bool() const noexcept { return handle_ != nullptr; }

  cublasStatus_t copy(int count, const double* source, int source_stride, double* destination,
                      int destination_stride) const;
  cublasStatus_t scale(int count, const double* coefficient, double* values, int stride) const;
  cublasStatus_t add(int count, const double* coefficient, const double* source, int source_stride,
                     double* destination, int destination_stride) const;
  cublasStatus_t dot(int count, const double* left, int left_stride, const double* right,
                     int right_stride, double* result) const;
  cublasStatus_t norm(int count, const double* values, int stride, double* result) const;
  cublasStatus_t product(bool transpose_left, bool transpose_right, int rows, int columns,
                         int reduction, const double* alpha, const double* left, int left_ld,
                         const double* right, int right_ld, const double* beta, double* output,
                         int output_ld) const;

  /** Publish A + A^T into disjoint column-major square storage.
   * The optional successful-submission counter includes partial work on failure.
   * Providers without GEAM retain bounded copy/strided-vector execution; errors
   * never trigger a replay or change the already-bound handle/stream.
   */
  cublasStatus_t transpose_sum(int order, const double* input, double* output,
                               std::uint64_t* successful_submissions = nullptr) const;

 private:
  cublasHandle_t handle_{};
};

}  // namespace generativeqc::tensor

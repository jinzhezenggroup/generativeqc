#pragma once

#include <cublas_v2.h>

namespace generativeqc::tensor {

/** Prepared FP64 destination += scale * source on one caller-owned stream.
 * The enclosing owner admits provider storage, validates disjoint vector views,
 * and drains work before destruction. Preparation happens outside capture;
 * enqueue allocates nothing and preserves the prepared stream/pointer modes.
 */
class CudaVectorAccumulation {
 public:
  CudaVectorAccumulation() = default;
  CudaVectorAccumulation(const CudaVectorAccumulation&) = delete;
  CudaVectorAccumulation& operator=(const CudaVectorAccumulation&) = delete;
  ~CudaVectorAccumulation() {
    if (handle_) (void)cublasDestroy(handle_);
  }

  /** Partial preparation retains its handle solely for bounded cleanup. */
  cublasStatus_t prepare(cudaStream_t stream) {
    if (handle_) return CUBLAS_STATUS_INVALID_VALUE;
    auto status = cublasCreate(&handle_);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasSetStream(handle_, stream);
    return status;
  }

  /** Borrow nonoverlapping contiguous FP64 views until stream completion. */
  cublasStatus_t enqueue(int count, double scale, const double* source, double* destination) {
    return cublasDaxpy(handle_, count, &scale, source, 1, destination, 1);
  }

 private:
  cublasHandle_t handle_{};
};

}  // namespace generativeqc::tensor

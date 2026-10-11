#pragma once

#include "tensor/cuda_contraction.cuh"

namespace generativeqc::tensor {

/** Strict-FP64 resident vector primitives for the shared response controller.
 * The existing context owns handle/stream modes, explicit workspace, execution
 * versions and bounded provider growth. Callers own vector storage and account
 * for kProviderAllowance separately from their numeric arena.
 */
class CudaKrylovVectors {
 public:
  static constexpr std::size_t kProviderAllowance = CudaContractionContext::kProviderAllowance;
  bool prepare(cudaStream_t stream) { return context_.prepare(stream, 256ULL << 10); }
  void reset() noexcept { context_.reset(); }
  bool prepared() const noexcept { return context_.prepared(); }
  std::size_t retained_bytes() const noexcept { return context_.retained_bytes(); }
  std::size_t workspace_bytes() const noexcept { return context_.workspace_bytes(); }
  int provider_version() const noexcept { return context_.provider_version(); }
  int runtime_version() const noexcept { return context_.runtime_version(); }

  void scale(int dimension, double alpha, double* values) {
    generativeqc_tensor::blas_check(cublasDscal(context_.handle(), dimension, &alpha, values, 1));
  }
  void axpy(int dimension, double alpha, const double* source, double* destination) {
    generativeqc_tensor::blas_check(
        cublasDaxpy(context_.handle(), dimension, &alpha, source, 1, destination, 1));
  }
  double dot(int dimension, const double* left, const double* right) {
    double value = 0.0;
    generativeqc_tensor::blas_check(
        cublasDdot(context_.handle(), dimension, left, 1, right, 1, &value));
    return value;
  }
  double norm(int dimension, const double* values) {
    double value = 0.0;
    generativeqc_tensor::blas_check(cublasDnrm2(context_.handle(), dimension, values, 1, &value));
    return value;
  }

 private:
  CudaContractionContext context_;
};

}  // namespace generativeqc::tensor

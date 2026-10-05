#pragma once

#include <chrono>
#include <climits>
#include <cstdio>

#include "runtime/resource_cuda.cuh"
#include "tensor/cuda_contraction.cuh"
#include "tensor/cuda_panel_product.hpp"

namespace generativeqc::tensor {
#if defined(GENERATIVEQC_TEST_HOOKS)
inline thread_local bool panel_product_library_for_test = false;
inline thread_local bool panel_product_unavailable_for_test = false;
#endif

namespace panel_product_detail {
__global__ void publish(double* values, std::size_t count, int* error) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x) {
    if (!isfinite(values[i])) {
      atomicCAS(error, 0, 1);
      values[i] = 0.0;
    }
  }
}
}  // namespace panel_product_detail

/** Optional strict-FP64 panel provider. Unknown endpoint cost retains the
 * generated incumbent; only explicit qualification can select the new offer.
 * A single prepared handle and matrix cache serve every full/tail panel.
 */
class CudaPanelProduct final : public PreparedPanelProduct {
 public:
  template <std::size_t N>
  CudaPanelProduct(const runtime::NativeLoweringRequest& request,
                   std::array<runtime::NativeLoweringCandidate, N> offers, std::string_view target,
                   std::string_view compilation, std::size_t columns, std::size_t rows,
                   std::size_t batches, cudaStream_t stream, std::size_t budget,
                   std::size_t incumbent, std::size_t alternative, bool dense_physical)
      : columns_(columns), rows_(rows), batches_(batches) {
    static_assert(sizeof(CudaPanelProduct) + 3 * sizeof(offers) + 4096 <= host_reservation);
    if (!columns || !rows || !batches || incumbent >= N || alternative >= N)
      throw std::invalid_argument("invalid panel product domain");
    const auto matrix_bytes = contraction_product(
        contraction_product(contraction_product(columns, columns), batches), sizeof(double));
    (void)contraction_product(contraction_product(contraction_product(columns, rows), batches),
                              sizeof(double));
    const auto start = std::chrono::steady_clock::now();
    cudaStreamCaptureStatus capture{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("panel product preparation cannot capture");
    auto& offer = offers[alternative];
    if (offer.algorithm != "matrix-panel-gemm" || offer.precision >= request.precisions.size() ||
        request.inputs != 2 || request.input_dtypes[0] != runtime::PrecisionDtype::Fp64 ||
        request.input_dtypes[1] != runtime::PrecisionDtype::Fp64 ||
        !runtime::strict_requested_precision(request, request.precisions[offer.precision]) ||
        !request.precisions[offer.precision].arithmetic.is_strict_fp64() ||
        request.precisions[offer.precision].publication_dtype != runtime::PrecisionDtype::Fp64 ||
        !request.precisions[offer.precision].casts.empty() ||
        !request.precisions[offer.precision].refinement.empty() ||
        !request.precisions[offer.precision].audit.empty())
      throw std::invalid_argument("panel product requires an admitted strict-FP64 candidate");
    for (auto& candidate : offers) {
      // Validate the lookup before the shared selector can inspect the offer.
      if (candidate.precision >= request.precisions.size())
        throw std::invalid_argument("panel candidate precision is outside the request");
      candidate.host_bytes = host_reservation;
      if (!request.precisions[candidate.precision].arithmetic.is_strict_fp64())
        candidate.rejection = "panel provider does not implement mixed arithmetic";
    }
    offer.provider_bytes = CudaContractionContext::kProviderAllowance;
    offer.cache_bytes = matrix_bytes;
    bool qualified = false, available = true;
#if defined(GENERATIVEQC_TEST_HOOKS)
    qualified = panel_product_library_for_test;
    available = !panel_product_unavailable_for_test;
#endif
    if (!dense_physical)
      offer.rejection = "indexed and response domains retain generated execution";
    else if (!qualified)
      offer.rejection = "matrix-panel endpoint profile is not qualified";
    else if (columns > INT_MAX || rows > INT_MAX)
      offer.rejection = "matrix-panel dimensions exceed the provider integer domain";
    else if (budget < matrix_bytes || budget - matrix_bytes < offer.provider_bytes)
      offer.rejection = "matrix-panel cache and provider allowance are not admitted";
    else if (!available)
      offer.rejection = "matrix-panel provider is unavailable";
    if (offer.rejection.empty() && !context_.prepare(stream))
      offer.rejection = "matrix-panel provider allocation unavailable";
    if (offer.rejection.empty()) {
      bool host_oom = false;
      const auto status = runtime::resource_cuda_malloc(reinterpret_cast<void**>(&matrix_),
                                                        matrix_bytes, &host_oom);
      // Only device exhaustion admits the generated retry. Failure to retain
      // host registry metadata cannot be reinterpreted as a smaller GPU budget.
      if (status == cudaErrorMemoryAllocation && !host_oom) {
        (void)cudaGetLastError();
        context_.reset();
        offer.rejection = "matrix-panel cache allocation unavailable";
      } else {
        generativeqc_tensor::cuda_check(status);
      }
    }
    try {
      if (!matrix_) context_.prepare_generated(stream);
      if (matrix_) {
        std::snprintf(version_.data(), version_.size(), "%d", context_.provider_version());
        offer.provider_version = version_.data();
      }
      const auto decision = runtime::select_native_lowering(request, offers, target, compilation, 1,
                                                            matrix_ ? alternative : incumbent);
      enabled_ = decision.selected == alternative;
      diagnostic_ = {
          offers[decision.selected],
          offer.rejection,
          host_reservation,
          enabled_ ? matrix_bytes : 0,
          enabled_ ? CudaContractionContext::kProviderAllowance : 0,
          context_.retained_bytes(),
          context_.provider_version(),
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
    } catch (...) {
      release_matrix();
      throw;
    }
  }
  ~CudaPanelProduct() override { release_matrix(); }
  bool enabled() const noexcept override { return enabled_; }
  double* materialized_matrices() const noexcept override { return matrix_; }
  const PanelProductDiagnostic& diagnostic() const noexcept override { return diagnostic_; }
  void execute(cudaStream_t stream, std::size_t rows, const double* panel, double* output,
               int* error) const override {
    if (!enabled_ || stream != context_.stream() || !rows || rows > rows_ || !panel || !output ||
        !error)
      throw std::invalid_argument("panel product invocation exceeds its prepared domain");
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != context_.device()) throw std::invalid_argument("panel product device changed");
    const auto input_bytes = rows * columns_ * sizeof(double);
    const auto output_bytes = batches_ * input_bytes;
    if (runtime::ranges_overlap(output, output_bytes, panel, input_bytes) ||
        runtime::ranges_overlap(output, output_bytes, matrix_, diagnostic_.matrix_bytes) ||
        runtime::ranges_overlap(panel, input_bytes, matrix_, diagnostic_.matrix_bytes))
      throw std::invalid_argument("panel product buffers alias");
    const double alpha = 1.0, beta = 0.0;
    for (std::size_t batch = 0; batch != batches_; ++batch)
      generativeqc_tensor::blas_check(
          cublasDgemm(context_.handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(columns_),
                      static_cast<int>(rows), static_cast<int>(columns_), &alpha,
                      matrix_ + batch * columns_ * columns_, static_cast<int>(columns_), panel,
                      static_cast<int>(columns_), &beta, output + batch * rows * columns_,
                      static_cast<int>(columns_)));
    const auto count = batches_ * rows * columns_;
    panel_product_detail::
        publish<<<std::min<std::size_t>(65535, (count + 127) / 128), 128, 0, stream>>>(
            output, count, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
  }

 private:
  void release_matrix() noexcept {
    if (!matrix_) return;
    int previous = context_.device();
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(context_.device());
    (void)cudaStreamSynchronize(context_.stream());
    (void)runtime::resource_cuda_free(matrix_);
    matrix_ = nullptr;
    (void)cudaSetDevice(previous);
  }
  CudaContractionContext context_;
  std::size_t columns_{}, rows_{}, batches_{};
  double* matrix_{};
  bool enabled_{};
  std::array<char, 32> version_{};
  PanelProductDiagnostic diagnostic_{};
};
}  // namespace generativeqc::tensor

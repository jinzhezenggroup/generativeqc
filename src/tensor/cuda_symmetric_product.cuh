#pragma once

#include <chrono>
#include <climits>

#include "tensor/cuda_contraction.cuh"
#include "tensor/cuda_symmetric_product.hpp"

namespace generativeqc::tensor {

#if defined(GENERATIVEQC_TEST_HOOKS)
// Explicit qualification only: production has no retained endpoint profile yet.
inline thread_local bool symmetric_product_library_for_test = false;
inline thread_local bool symmetric_product_unavailable_for_test = false;
#endif

namespace symmetric_product_detail {
__global__ void publish(double* c, std::size_t n, std::size_t batches, int* error) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < batches * n * n;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const auto row = i / n % n, col = i % n;
    if (row > col) continue;
    auto value = c[i];
    if (!isfinite(value)) {
      atomicCAS(error, 0, 3);
      value = 0.0;
    }
    c[i] = value;
    c[(i / (n * n) * n + col) * n + row] = value;
  }
}
}  // namespace symmetric_product_detail

/** Bounded rank-2k executor with an immutable shared-provider selection.
 * Local maps cannot be represented as a dense destination and always execute
 * the retained generated callback. Resource qualification never implies a
 * scientific precision change or an uncharged temporary/scatter allocation.
 */
class CudaSymmetricProduct final : public PreparedSymmetricProduct {
 public:
  template <std::size_t N>
  CudaSymmetricProduct(const runtime::NativeLoweringRequest& request,
                       const std::array<runtime::NativeLoweringCandidate, N>& candidates,
                       std::string_view target, std::string_view compilation, std::size_t columns,
                       std::size_t reduction, std::size_t batches, cudaStream_t stream,
                       std::size_t provider_budget)
      : columns_(columns), reduction_(reduction), batches_(batches) {
    static_assert(N == 2);
    static_assert(sizeof(CudaSymmetricProduct) + 3 * sizeof(candidates) + 8192 <= host_reservation);
    if (request.dtype != runtime::PrecisionDtype::Fp64 ||
        request.accumulation_dtype != runtime::PrecisionDtype::Fp64 || request.inputs != 3 ||
        request.input_dtypes[0] != runtime::PrecisionDtype::Fp64 ||
        request.input_dtypes[1] != runtime::PrecisionDtype::Fp64 ||
        request.input_dtypes[2] != runtime::PrecisionDtype::Fp64 ||
        request.precisions.size() != 1 ||
        !runtime::strict_requested_precision(request, request.precisions[0]) ||
        request.precisions[0].publication_dtype != runtime::PrecisionDtype::Fp64 ||
        !request.precisions[0].casts.empty() || !request.precisions[0].refinement.empty() ||
        !request.precisions[0].audit.empty() || candidates[0].provider != "generated.cuda" ||
        candidates[1].provider != "cublas" ||
        candidates[0].algorithm != "symmetric-cross-generated" ||
        candidates[1].algorithm != "symmetric-cross-rank2k")
      throw std::invalid_argument("symmetric product requires the canonical strict-FP64 portfolio");
    if (!columns || !reduction || !batches) throw std::invalid_argument("empty symmetric domain");
    (void)contraction_product(contraction_product(columns, columns), batches);
    (void)contraction_product(contraction_product(columns, reduction), batches);
    const auto start = std::chrono::steady_clock::now();
    auto offers = candidates;
    bool qualified = false, available = true;
#if defined(GENERATIVEQC_TEST_HOOKS)
    qualified = symmetric_product_library_for_test;
    available = !symmetric_product_unavailable_for_test;
#endif
    // Static legality precedes resource acquisition. Unknown endpoint costs do
    // not promote an optional implementation through an optimistic heuristic.
    auto& alternative = offers[1];
    alternative.provider_bytes = CudaContractionContext::kProviderAllowance;
    if (!qualified)
      alternative.rejection = "rank-2k endpoint profile is not qualified";
    else if (columns > INT_MAX || reduction > INT_MAX)
      alternative.rejection = "rank-2k dimensions exceed the provider integer domain";
    else if (provider_budget < CudaContractionContext::kProviderAllowance)
      alternative.rejection = "rank-2k provider allowance is not admitted";
    else if (!available)
      alternative.rejection = "rank-2k provider is unavailable";
    if (alternative.rejection.empty() && !context_.prepare(stream))
      alternative.rejection = "rank-2k provider allocation unavailable";
    const std::size_t incumbent = alternative.rejection.empty() ? 1 : 0;
    if (!incumbent) context_.prepare_generated(stream);
    const auto selection =
        runtime::select_native_lowering(request, offers, target, compilation, 1, incumbent);
    library_ = selection.selected == 1;
    diagnostic_ = {offers[selection.selected],
                   offers[0],
                   alternative.rejection,
                   host_reservation,
                   library_ ? CudaContractionContext::kProviderAllowance : 0,
                   context_.retained_bytes(),
                   context_.provider_version(),
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
  }

  void execute(cudaStream_t stream, const SymmetricProductInvocation& call) const override {
    if (stream != context_.stream())
      throw std::invalid_argument("symmetric product stream changed");
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != context_.device())
      throw std::invalid_argument("symmetric product device changed");
    if (call.columns > columns_ || !call.reduction || call.reduction > reduction_ ||
        !call.batches || call.batches > batches_ || !call.generated || !call.materialize ||
        !call.epilogue || !call.output || !call.error || (call.columns && (!call.a || !call.b)))
      throw std::invalid_argument("symmetric product invocation exceeds its prepared domain");
    if (!library_ || call.indexed || !call.columns) {
      call.generated(call.state);
      return;
    }
    call.materialize(call.state);
    const double alpha = 1.0, beta = call.accumulate ? 1.0 : 0.0;
    const auto n = call.columns, k = call.reduction;
    for (std::size_t spin = 0; spin < call.batches; ++spin)
      // Column-major lower is row-major upper, the compiler's authoritative
      // triangle. Packed reduction axes need neither replication nor a copy.
      generativeqc_tensor::blas_check(cublasDsyr2k(
          context_.handle(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, static_cast<int>(n),
          static_cast<int>(k), &alpha, call.a, static_cast<int>(n), call.b + spin * k * n,
          static_cast<int>(n), &beta, call.output + spin * n * n, static_cast<int>(n)));
    symmetric_product_detail::
        publish<<<generativeqc_tensor::blocks(call.batches * n * n, 128), 128, 0, stream>>>(
            call.output, n, call.batches, call.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    call.epilogue(call.state);
  }
  const SymmetricProductDiagnostic& diagnostic() const noexcept override { return diagnostic_; }

 private:
  std::size_t columns_, reduction_, batches_;
  CudaContractionContext context_;
  SymmetricProductDiagnostic diagnostic_;
  bool library_{};
};
}  // namespace generativeqc::tensor

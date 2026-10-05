#pragma once

#include <chrono>
#include <climits>

#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_cuda.cuh"
#include "tensor/cuda_contraction.cuh"
#include "tensor/cuda_symmetric_product.hpp"

namespace generativeqc::tensor {

#if defined(GENERATIVEQC_TEST_HOOKS)
// Explicit qualification only: production has no retained endpoint profile yet.
inline thread_local bool symmetric_product_library_for_test = false;
inline thread_local bool symmetric_product_indexed_library_for_test = false;
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

/** Publish one compact authoritative triangle into an immutable indexed view.
 * Unique sorted IDs give each thread exclusive ownership of both matrix legs;
 * successive tiles are ordered on the prepared stream, so no atomics or dense
 * zero/scatter buffer is needed. Entries outside the view remain untouched. */
__global__ void scatter(const double* compact, double* output, const std::size_t* ids,
                        std::size_t n, std::size_t full, std::size_t batches, bool accumulate,
                        int* error) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < batches * n * n;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const auto row = i / n % n, col = i % n, spin = i / (n * n);
    if (row > col) continue;
    const auto mu = ids[row], nu = ids[col];
    if (mu >= full || nu >= full) {
      atomicCAS(error, 0, 3);
      continue;
    }
    auto value = compact[i];
    if (!isfinite(value)) {
      atomicCAS(error, 0, 3);
      value = 0.0;
    }
    const auto target = (spin * full + mu) * full + nu;
    if (accumulate) value += output[target];
    if (!isfinite(value)) {
      atomicCAS(error, 0, 3);
      value = 0.0;
    }
    output[target] = value;
    output[(spin * full + nu) * full + mu] = value;
  }
}
}  // namespace symmetric_product_detail

/** Bounded rank-2k executor with an immutable shared-provider selection.
 * An independently qualified indexed recipe uses one charged compact-output
 * cache and scatters into the caller's mapped destination. No selection or
 * allocation occurs per tile. Unadmitted domains retain the generated callback.
 */
class CudaSymmetricProduct final : public PreparedSymmetricProduct {
 public:
  template <std::size_t N>
  CudaSymmetricProduct(const runtime::NativeLoweringRequest& request,
                       const std::array<runtime::NativeLoweringCandidate, N>& candidates,
                       std::string_view target, std::string_view compilation, std::size_t columns,
                       std::size_t reduction, std::size_t batches, cudaStream_t stream,
                       std::size_t provider_budget, bool indexed = false, bool nonempty = true)
      : columns_(columns), reduction_(reduction), batches_(batches), indexed_(indexed) {
    static_assert(N == 3);
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
        candidates[1].provider != "cublas" || candidates[2].provider != "cublas" ||
        candidates[0].algorithm != "symmetric-cross-generated" ||
        candidates[1].algorithm != "symmetric-cross-rank2k" ||
        candidates[2].algorithm != "indexed-symmetric-cross-rank2k")
      throw std::invalid_argument("symmetric product requires the canonical strict-FP64 portfolio");
    if (!columns || !reduction || !batches) throw std::invalid_argument("empty symmetric domain");
    const auto matrix_bytes = contraction_product(
        contraction_product(contraction_product(columns, columns), batches), sizeof(double));
    (void)contraction_product(contraction_product(contraction_product(columns, reduction), batches),
                              sizeof(double));
    const auto start = std::chrono::steady_clock::now();
    auto offers = candidates;
    // Include shared binding/preparation storage in each executable offer, so
    // candidate provenance agrees with the enclosing owner's resource charge.
    for (auto& offer : offers) offer.host_bytes = host_reservation;
    bool qualified = false, available = true;
#if defined(GENERATIVEQC_TEST_HOOKS)
    qualified =
        indexed ? symmetric_product_indexed_library_for_test : symmetric_product_library_for_test;
    available = !symmetric_product_unavailable_for_test;
#endif
    // Static legality precedes resource acquisition. Unknown endpoint costs do
    // not promote an optional implementation through an optimistic heuristic.
    const std::size_t alternative_index = indexed ? 2 : 1;
    offers[indexed ? 1 : 2].rejection = "materialization does not match the prepared output domain";
    auto& alternative = offers[alternative_index];
    alternative.provider_bytes = CudaContractionContext::kProviderAllowance;
    alternative.cache_bytes = indexed ? matrix_bytes : 0;
    if (!nonempty)
      alternative.rejection = "indexed output domain is empty";
    else if (!qualified)
      alternative.rejection = "rank-2k endpoint profile is not qualified";
    else if (columns > INT_MAX || reduction > INT_MAX)
      alternative.rejection = "rank-2k dimensions exceed the provider integer domain";
    else if (provider_budget < alternative.cache_bytes ||
             provider_budget - alternative.cache_bytes < CudaContractionContext::kProviderAllowance)
      alternative.rejection = "rank-2k cache and provider allowance are not admitted";
    else if (!available)
      alternative.rejection = "rank-2k provider is unavailable";
    if (alternative.rejection.empty() && !context_.prepare(stream))
      alternative.rejection = "rank-2k provider allocation unavailable";
    if (alternative.rejection.empty() && indexed) {
      bool host_oom = false;
      const auto status = runtime::resource_cuda_malloc(reinterpret_cast<void**>(&matrix_),
                                                        matrix_bytes, &host_oom);
      // A numeric-ledger/device miss can retain generated execution; failure
      // to record host ownership must not masquerade as a smaller device plan.
      if (status == cudaErrorMemoryAllocation && !host_oom) {
        (void)cudaGetLastError();
        // This is a live fallback, not destructor cleanup. A failure to drain
        // or destroy the optional provider must propagate rather than retry.
        {
          std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
          context_.release_locked();
        }
        alternative.rejection = "rank-2k compact output allocation unavailable";
      } else {
        generativeqc_tensor::cuda_check(status);
      }
    }
    try {
      const std::size_t incumbent = alternative.rejection.empty() ? alternative_index : 0;
      if (!incumbent) context_.prepare_generated(stream);
      const auto selection =
          runtime::select_native_lowering(request, offers, target, compilation, 1, incumbent);
      library_ = selection.selected == alternative_index;
      diagnostic_ = {
          offers[selection.selected],
          offers[library_ && indexed ? alternative_index : 0],
          alternative.rejection,
          host_reservation,
          library_ ? CudaContractionContext::kProviderAllowance : 0,
          context_.retained_bytes(),
          context_.provider_version(),
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
          library_ && indexed ? matrix_bytes : 0};
    } catch (...) {
      release_matrix();
      throw;
    }
  }
  ~CudaSymmetricProduct() override { release_matrix(); }

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
    if (!library_ || call.indexed != indexed_ || !call.columns) {
      call.generated(call.state);
      return;
    }
    if (indexed_ &&
        (!call.column_ids || call.output_columns < call.columns || call.output_columns > columns_))
      throw std::invalid_argument("symmetric product indexed output exceeds its prepared domain");
    const auto full = indexed_ ? call.output_columns : call.columns;
    const auto output_bytes = call.batches * full * full * sizeof(double);
    const auto panel_bytes = call.reduction * call.columns * sizeof(double);
    if (runtime::ranges_overlap(call.output, output_bytes, call.a, panel_bytes) ||
        runtime::ranges_overlap(call.output, output_bytes, call.b, call.batches * panel_bytes))
      throw std::invalid_argument("symmetric product output aliases its input panels");
    call.materialize(call.state);
    // The compact cache is overwritten per tile. Accumulation belongs to the
    // mapped destination and must never read stale compact scratch.
    const double alpha = 1.0, beta = call.accumulate && !indexed_ ? 1.0 : 0.0;
    const auto n = call.columns, k = call.reduction;
    for (std::size_t spin = 0; spin < call.batches; ++spin)
      // Column-major lower is row-major upper, the compiler's authoritative
      // triangle. Packed reduction axes need neither replication nor a copy.
      generativeqc_tensor::blas_check(
          cublasDsyr2k(context_.handle(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, static_cast<int>(n),
                       static_cast<int>(k), &alpha, call.a, static_cast<int>(n),
                       call.b + spin * k * n, static_cast<int>(n), &beta,
                       (indexed_ ? matrix_ : call.output) + spin * n * n, static_cast<int>(n)));
    if (indexed_)
      symmetric_product_detail::
          scatter<<<generativeqc_tensor::blocks(call.batches * n * n, 128), 128, 0, stream>>>(
              matrix_, call.output, call.column_ids, n, full, call.batches, call.accumulate,
              call.error);
    else
      symmetric_product_detail::
          publish<<<generativeqc_tensor::blocks(call.batches * n * n, 128), 128, 0, stream>>>(
              call.output, n, call.batches, call.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    call.epilogue(call.state);
  }
  const SymmetricProductDiagnostic& diagnostic() const noexcept override { return diagnostic_; }

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
  std::size_t columns_, reduction_, batches_;
  CudaContractionContext context_;
  SymmetricProductDiagnostic diagnostic_;
  double* matrix_{};
  bool library_{}, indexed_{};
};
}  // namespace generativeqc::tensor

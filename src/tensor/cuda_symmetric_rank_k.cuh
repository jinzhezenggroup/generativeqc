#pragma once

#include <chrono>
#include <climits>
#include <cmath>

#include "generated_symmetric_rank_k.cuh"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_cuda.cuh"
#include "tensor/cuda_contraction.cuh"

namespace generativeqc::tensor {

enum class RankKOrder { RowMajor, ColumnMajor };

/** One borrowed physical batch of C diag(w) C^T, repeated for `batches`.
 * Inputs and destination cannot alias.  The caller initializes error to zero
 * on the same stream before every execution, including graph replay.  On a
 * numerical error no output element is published.  Alpha/beta are host scalars
 * bound to the compiler-owned scalar update TensorIR program.
 */
struct SymmetricRankKInvocation {
  std::size_t n{}, k{}, batches{};
  const double* coefficients{};
  const double* weights{};
  double* output{};
  int* error{};
  double alpha{1.0}, beta{};
};

struct SymmetricRankKDiagnostic {
  runtime::NativeLoweringCandidate selected;
  std::string_view library_rejection;
  std::size_t host_bytes{}, temporary_bytes{}, provider_allowance{}, retained_provider_bytes{};
  int provider_version{}, runtime_version{};
  double prepare_seconds{};
};

namespace rank_k_detail {
__device__ inline std::size_t panel_index(std::size_t row, std::size_t orbital, std::size_t n,
                                          std::size_t k, RankKOrder order) {
  return order == RankKOrder::RowMajor ? row * k + orbital : row + orbital * n;
}

__global__ void scale(const double* coefficients, const double* weights, double* scaled,
                      std::size_t n, std::size_t k, std::size_t batches, RankKOrder order,
                      int* error) {
  const auto count = batches * n * k;
  for (std::size_t linear = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; linear < count;
       linear += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = linear / (n * k), physical = linear % (n * k);
    const auto orbital = order == RankKOrder::RowMajor ? physical % k : physical / n;
    const auto coefficient = coefficients[linear], weight = weights[batch * k + orbital];
    double value{};
    if (!rank_k_generated::rank_k_scale(coefficient, weight, value)) {
      atomicCAS(error, 0, 1);
      scaled[linear] = 0.0;
    } else {
      scaled[linear] = value;
    }
  }
}

__device__ inline bool generated_value(const SymmetricRankKInvocation call, std::size_t batch,
                                       std::size_t row, std::size_t col, RankKOrder order,
                                       double& result) {
  result = 0.0;
  const auto panel = call.coefficients + batch * call.n * call.k;
  const auto weights = call.weights + batch * call.k;
  for (std::size_t orbital = 0; orbital < call.k; ++orbital) {
    const auto left = panel[panel_index(row, orbital, call.n, call.k, order)];
    const auto right = panel[panel_index(col, orbital, call.n, call.k, order)];
    double weighted{}, updated{};
    if (!rank_k_generated::rank_k_scale(left, weights[orbital], weighted) ||
        !rank_k_generated::rank_k_update(weighted, right, result, updated))
      return false;
    result = updated;
  }
  return true;
}

__device__ inline std::size_t matrix_index(std::size_t row, std::size_t col, std::size_t n,
                                           RankKOrder order) {
  return order == RankKOrder::RowMajor ? row * n + col : row + col * n;
}

__device__ inline bool update_value(const SymmetricRankKInvocation call, double product,
                                    double old_output, double& updated) {
  return call.beta == 0.0 ? rank_k_generated::rank_k_alpha_overwrite(call.alpha, product, updated)
                          : rank_k_generated::rank_k_alpha_beta_update(
                                call.alpha, product, call.beta, old_output, updated);
}

__global__ void validate_generated(SymmetricRankKInvocation call, RankKOrder order) {
  const auto count = call.batches * call.n * call.n;
  for (std::size_t linear = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; linear < count;
       linear += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = linear / (call.n * call.n), local = linear % (call.n * call.n);
    const auto row = local / call.n, col = local % call.n;
    if (row > col) continue;
    const auto* panel = call.coefficients + batch * call.n * call.k;
    const auto* weights = call.weights + batch * call.k;
    for (std::size_t orbital = 0; orbital < call.k; ++orbital) {
      const auto left = panel[panel_index(row, orbital, call.n, call.k, order)];
      const auto right = panel[panel_index(col, orbital, call.n, call.k, order)];
      if (!isfinite(left) || !isfinite(right) || !isfinite(weights[orbital]))
        atomicCAS(call.error, 0, 1);
    }
    double value{};
    if (!generated_value(call, batch, row, col, order, value)) atomicCAS(call.error, 0, 1);
    const auto old = rank_k_generated::rank_k_bound_old_output(
        call.output, batch, call.n, row, col, order == RankKOrder::RowMajor, call.beta);
    double updated{};
    if (!update_value(call, value, old, updated)) atomicCAS(call.error, 0, 1);
  }
}

__global__ void validate_library(SymmetricRankKInvocation call, const double* product,
                                 RankKOrder order) {
  const auto count = call.batches * call.n * call.n;
  for (std::size_t linear = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; linear < count;
       linear += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = linear / (call.n * call.n), local = linear % (call.n * call.n);
    const auto row = local / call.n, col = local % call.n;
    if (row > col) continue;
    const auto address = batch * call.n * call.n + matrix_index(row, col, call.n, order);
    const auto value = product[address];
    const auto old = rank_k_generated::rank_k_bound_old_output(
        call.output, batch, call.n, row, col, order == RankKOrder::RowMajor, call.beta);
    double updated{};
    if (!update_value(call, value, old, updated)) atomicCAS(call.error, 0, 1);
  }
}

__global__ void publish(SymmetricRankKInvocation call, const double* product, RankKOrder order) {
  if (*call.error != 0) return;
  const auto count = call.batches * call.n * call.n;
  for (std::size_t linear = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; linear < count;
       linear += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = linear / (call.n * call.n), local = linear % (call.n * call.n);
    const auto row = local / call.n, col = local % call.n;
    if (row > col) continue;
    const auto offset = batch * call.n * call.n;
    double value{};
    if (product)
      value = product[offset + matrix_index(row, col, call.n, order)];
    else if (!generated_value(call, batch, row, col, order, value))
      return;
    const auto old = rank_k_generated::rank_k_bound_old_output(
        call.output, batch, call.n, row, col, order == RankKOrder::RowMajor, call.beta);
    double updated{};
    if (!update_value(call, value, old, updated)) return;
    call.output[offset + matrix_index(row, col, call.n, order)] = updated;
    call.output[offset + matrix_index(col, row, call.n, order)] = updated;
  }
}
}  // namespace rank_k_detail

/** Prepared rank-k qualification binding.  Library promotion requires an
 * explicit endpoint qualification input; ordinary construction retains the
 * executable generated fallback.  No allocation or selection occurs in execute.
 */
class CudaSymmetricRankK final {
 public:
  static constexpr std::size_t host_reservation = 16U << 10;

  template <std::size_t N>
  CudaSymmetricRankK(const runtime::NativeLoweringRequest& overwrite_request,
                     const std::array<runtime::NativeLoweringCandidate, N>& overwrite_candidates,
                     std::string_view overwrite_target, std::string_view overwrite_compilation,
                     const runtime::NativeLoweringRequest& update_request,
                     const std::array<runtime::NativeLoweringCandidate, N>& update_candidates,
                     std::string_view update_target, std::string_view update_compilation,
                     std::size_t compiled_n, std::size_t compiled_k, std::size_t n, std::size_t k,
                     std::size_t batches, RankKOrder order, cudaStream_t stream,
                     std::size_t provider_budget, bool library_qualified = false)
      : n_(n), k_(k), batches_(batches), order_(order) {
    static_assert(N == 2);
    static_assert(sizeof(CudaSymmetricRankK) + 2 * sizeof(overwrite_candidates) + 8192 <=
                  host_reservation);
#if !defined(GENERATIVEQC_TEST_HOOKS)
    if (library_qualified) throw std::invalid_argument("rank-k library qualification is test-only");
#endif
    if (!n || !k || !batches) throw std::invalid_argument("empty rank-k domain");
    const auto panel = contraction_product(contraction_product(n, k), batches);
    const auto matrix = contraction_product(contraction_product(n, n), batches);
    panel_bytes_ = contraction_product(panel, sizeof(double));
    matrix_bytes_ = contraction_product(matrix, sizeof(double));
    temporary_bytes_ = runtime::lowering_add(panel_bytes_, matrix_bytes_);
    const bool small_row_request =
        (overwrite_request.identity ==
             rank_k_generated::rank_k_density_row_overwrite_request.identity &&
         update_request.identity == rank_k_generated::rank_k_density_row_update_request.identity) ||
        (overwrite_request.identity ==
             rank_k_generated::rank_k_weighted_density_row_overwrite_request.identity &&
         update_request.identity ==
             rank_k_generated::rank_k_weighted_density_row_update_request.identity);
    const bool large_row_request =
        (overwrite_request.identity ==
             rank_k_generated::rank_k_density_n17_k9_row_overwrite_request.identity &&
         update_request.identity ==
             rank_k_generated::rank_k_density_n17_k9_row_update_request.identity) ||
        (overwrite_request.identity ==
             rank_k_generated::rank_k_weighted_density_n17_k9_row_overwrite_request.identity &&
         update_request.identity ==
             rank_k_generated::rank_k_weighted_density_n17_k9_row_update_request.identity);
    const bool small_column_request =
        (overwrite_request.identity ==
             rank_k_generated::rank_k_density_column_overwrite_request.identity &&
         update_request.identity ==
             rank_k_generated::rank_k_density_column_update_request.identity) ||
        (overwrite_request.identity ==
             rank_k_generated::rank_k_weighted_density_column_overwrite_request.identity &&
         update_request.identity ==
             rank_k_generated::rank_k_weighted_density_column_update_request.identity);
    const bool large_column_request =
        (overwrite_request.identity ==
             rank_k_generated::rank_k_density_n17_k9_column_overwrite_request.identity &&
         update_request.identity ==
             rank_k_generated::rank_k_density_n17_k9_column_update_request.identity) ||
        (overwrite_request.identity ==
             rank_k_generated::rank_k_weighted_density_n17_k9_column_overwrite_request.identity &&
         update_request.identity ==
             rank_k_generated::rank_k_weighted_density_n17_k9_column_update_request.identity);
    const bool row_request = small_row_request || large_row_request;
    const bool column_request = small_column_request || large_column_request;
    const bool large_request = large_row_request || large_column_request;
    const bool large_shape = compiled_n == 17 && compiled_k == 9;
    if ((order != RankKOrder::RowMajor && order != RankKOrder::ColumnMajor) ||
        row_request == column_request || row_request != (order == RankKOrder::RowMajor) ||
        (!large_shape && !(compiled_n == 3 && compiled_k == 5)) || large_request != large_shape ||
        compiled_n != n || compiled_k != k ||
        batches != rank_k_generated::rank_k_compiled_batches(overwrite_request.identity) ||
        batches != rank_k_generated::rank_k_compiled_batches(update_request.identity))
      throw std::invalid_argument("rank-k runtime shape/order differs from the compiled request");
    const auto valid = [](const auto& request, const auto& candidates, std::size_t inputs) {
      return request.dtype == runtime::PrecisionDtype::Fp64 &&
             request.accumulation_dtype == runtime::PrecisionDtype::Fp64 &&
             request.inputs == inputs && request.precisions.size() == 1 &&
             runtime::strict_requested_precision(request, request.precisions[0]) &&
             request.precisions[0].publication_dtype == runtime::PrecisionDtype::Fp64 &&
             request.precisions[0].casts.empty() && request.precisions[0].refinement.empty() &&
             request.precisions[0].audit.empty() && candidates[0].provider == "generated.cuda" &&
             candidates[1].provider == "cublas" &&
             candidates[0].algorithm == "symmetric-rank-k-generated" &&
             candidates[1].algorithm == "symmetric-rank-k-signed-gemm";
    };
    if (!valid(overwrite_request, overwrite_candidates, 3) ||
        !valid(update_request, update_candidates, 4))
      throw std::invalid_argument("rank-k requires its canonical strict-FP64 portfolio");
    const auto start = std::chrono::steady_clock::now();
    auto offers = update_candidates;
    auto overwrite_offers = overwrite_candidates;
    for (auto& offer : offers) offer.host_bytes = host_reservation;
    offers[1].temporary_bytes = temporary_bytes_;
    offers[1].provider_bytes = CudaContractionContext::kProviderAllowance;
    if (!library_qualified)
      offers[1].rejection = "rank-k complete endpoint is not qualified";
    else if (n > INT_MAX || k > INT_MAX)
      offers[1].rejection = "rank-k dimensions exceed provider integer range";
    else if (provider_budget < temporary_bytes_ ||
             provider_budget - temporary_bytes_ < CudaContractionContext::kProviderAllowance)
      offers[1].rejection = "rank-k scratch and provider allowance are not admitted";
    if (offers[1].rejection.empty() && !context_.prepare(stream))
      offers[1].rejection = "rank-k provider allocation unavailable";
    if (offers[1].rejection.empty()) {
      bool host_oom = false;
      const auto status = runtime::resource_cuda_malloc(reinterpret_cast<void**>(&scratch_),
                                                        temporary_bytes_, &host_oom);
      if (status == cudaErrorMemoryAllocation && !host_oom) {
        (void)cudaGetLastError();
        {
          std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
          context_.release_locked();
        }
        offers[1].rejection = "rank-k scratch allocation unavailable";
      } else {
        generativeqc_tensor::cuda_check(status);
      }
    }
    for (std::size_t candidate = 0; candidate < N; ++candidate) {
      overwrite_offers[candidate].host_bytes = offers[candidate].host_bytes;
      overwrite_offers[candidate].temporary_bytes = offers[candidate].temporary_bytes;
      overwrite_offers[candidate].provider_bytes = offers[candidate].provider_bytes;
      overwrite_offers[candidate].rejection = offers[candidate].rejection;
    }
    try {
      if (!offers[1].rejection.empty()) context_.prepare_generated(stream);
      const auto incumbent = offers[1].rejection.empty() ? 1 : 0;
      const auto selected = runtime::select_native_lowering(update_request, offers, update_target,
                                                            update_compilation, 1, incumbent);
      const auto overwrite_selected =
          runtime::select_native_lowering(overwrite_request, overwrite_offers, overwrite_target,
                                          overwrite_compilation, 1, incumbent);
      if (selected.selected != overwrite_selected.selected)
        throw std::logic_error("rank-k update domains selected different providers");
      library_ = selected.selected == 1;
      if (!library_ && scratch_)
        throw std::logic_error("rank-k selected generated with library scratch");
      const auto prepare_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      update_diagnostic_ = {offers[selected.selected],
                            offers[1].rejection,
                            host_reservation,
                            library_ ? temporary_bytes_ : 0,
                            library_ ? CudaContractionContext::kProviderAllowance : 0,
                            context_.retained_bytes(),
                            context_.provider_version(),
                            context_.runtime_version(),
                            prepare_seconds};
      overwrite_diagnostic_ = {overwrite_offers[overwrite_selected.selected],
                               overwrite_offers[1].rejection,
                               host_reservation,
                               library_ ? temporary_bytes_ : 0,
                               library_ ? CudaContractionContext::kProviderAllowance : 0,
                               context_.retained_bytes(),
                               context_.provider_version(),
                               context_.runtime_version(),
                               prepare_seconds};
    } catch (...) {
      release_scratch();
      throw;
    }
  }

  ~CudaSymmetricRankK() { release_scratch(); }
  CudaSymmetricRankK(const CudaSymmetricRankK&) = delete;
  CudaSymmetricRankK& operator=(const CudaSymmetricRankK&) = delete;

  void execute(cudaStream_t stream, const SymmetricRankKInvocation& call) const {
    if (stream != context_.stream()) throw std::invalid_argument("rank-k stream changed");
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != context_.device()) throw std::invalid_argument("rank-k device changed");
    if (call.n != n_ || call.k != k_ || call.batches != batches_ || !call.coefficients ||
        !call.weights || !call.output || !call.error || !std::isfinite(call.alpha) ||
        !std::isfinite(call.beta))
      throw std::invalid_argument("rank-k invocation exceeds prepared domain");
    const auto panel_bytes = contraction_product(
        contraction_product(contraction_product(call.n, call.k), call.batches), sizeof(double));
    const auto weights_bytes =
        contraction_product(contraction_product(call.k, call.batches), sizeof(double));
    const auto matrix_bytes = contraction_product(
        contraction_product(contraction_product(call.n, call.n), call.batches), sizeof(double));
    if (runtime::ranges_overlap(call.output, matrix_bytes, call.coefficients, panel_bytes) ||
        runtime::ranges_overlap(call.output, matrix_bytes, call.weights, weights_bytes) ||
        runtime::ranges_overlap(call.output, matrix_bytes, call.error, sizeof(int)) ||
        runtime::ranges_overlap(call.error, sizeof(int), call.coefficients, panel_bytes) ||
        runtime::ranges_overlap(call.error, sizeof(int), call.weights, weights_bytes) ||
        runtime::ranges_overlap(call.coefficients, panel_bytes, call.weights, weights_bytes))
      throw std::invalid_argument("rank-k output/error aliases its input storage");
    const auto total = contraction_product(contraction_product(call.batches, call.n), call.n);
    if (library_) {
      auto* scaled = scratch_;
      auto* product = scratch_ + panel_bytes_ / sizeof(double);
      const auto scale_count =
          contraction_product(contraction_product(call.batches, call.n), call.k);
      rank_k_detail::scale<<<generativeqc_tensor::blocks(scale_count, 128), 128, 0, stream>>>(
          call.coefficients, call.weights, scaled, call.n, call.k, call.batches, order_,
          call.error);
      const double one = 1.0, zero = 0.0;
      for (std::size_t batch = 0; batch < call.batches; ++batch) {
        const auto* input = call.coefficients + batch * call.n * call.k;
        const auto* weighted = scaled + batch * call.n * call.k;
        auto* result = product + batch * call.n * call.n;
        if (order_ == RankKOrder::RowMajor)
          generativeqc_tensor::blas_check(
              cublasDgemm(context_.handle(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(call.n),
                          static_cast<int>(call.n), static_cast<int>(call.k), &one, input,
                          static_cast<int>(call.k), weighted, static_cast<int>(call.k), &zero,
                          result, static_cast<int>(call.n)));
        else
          generativeqc_tensor::blas_check(
              cublasDgemm(context_.handle(), CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(call.n),
                          static_cast<int>(call.n), static_cast<int>(call.k), &one, input,
                          static_cast<int>(call.n), weighted, static_cast<int>(call.n), &zero,
                          result, static_cast<int>(call.n)));
      }
      rank_k_detail::validate_library<<<generativeqc_tensor::blocks(total, 128), 128, 0, stream>>>(
          call, product, order_);
      rank_k_detail::publish<<<generativeqc_tensor::blocks(total, 128), 128, 0, stream>>>(
          call, product, order_);
    } else {
      rank_k_detail::
          validate_generated<<<generativeqc_tensor::blocks(total, 128), 128, 0, stream>>>(call,
                                                                                          order_);
      rank_k_detail::publish<<<generativeqc_tensor::blocks(total, 128), 128, 0, stream>>>(
          call, nullptr, order_);
    }
    generativeqc_tensor::cuda_check(cudaGetLastError());
  }

  const SymmetricRankKDiagnostic& prepared_diagnostic(const SymmetricRankKInvocation& call) const {
    if (!std::isfinite(call.beta))
      throw std::invalid_argument("rank-k diagnostic requires a finite beta");
    return call.beta == 0.0 ? overwrite_diagnostic_ : update_diagnostic_;
  }

 private:
  void release_scratch() noexcept {
    if (!scratch_) return;
    int previous = context_.device();
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(context_.device());
    (void)cudaStreamSynchronize(context_.stream());
    (void)runtime::resource_cuda_free(scratch_);
    scratch_ = nullptr;
    (void)cudaSetDevice(previous);
  }

  std::size_t n_{}, k_{}, batches_{}, panel_bytes_{}, matrix_bytes_{}, temporary_bytes_{};
  RankKOrder order_{};
  CudaContractionContext context_;
  SymmetricRankKDiagnostic overwrite_diagnostic_, update_diagnostic_;
  double* scratch_{};
  bool library_{};
};
}  // namespace generativeqc::tensor

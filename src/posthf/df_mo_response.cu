#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

#include "df_mo_source_generated.hpp"
#include "posthf/capacity.hpp"
#include "posthf/df_mo_response.hpp"
#include "runtime/cuda_resources.cuh"

namespace generativeqc::posthf {
namespace {
std::size_t bytes(std::size_t count) { return checked_mul(count, sizeof(double)); }

}  // namespace

std::size_t df_mo_source_response_binding_capacity(std::size_t available) {
  return generated::DFMOSourceResponseExecution::plan(available).binding_bytes;
}

CudaDFMOSourceResponseDiagnostic pullback_df_mo_source_cuda(
    CudaDFMOSourceResponseView view, int device, cudaStream_t stream, const CudaDFSourceRead& read,
    const CudaDFSourceConsume& consume, const CudaDFSourceFinish& finish, std::size_t budget,
    std::size_t caller_bytes) {
  const auto n = view.nbf, q = view.naux;
  if (!n || !q || !view.coefficients || !view.inverse_root || !view.bar_whitened || device < 0 ||
      !stream || !read || !consume || !finish || !budget)
    throw std::invalid_argument("invalid native CUDA DF MO source response request");
  const auto work = generated::df_mo_source_response_work(n, q);
  const auto row = checked_mul(n, q), nn = checked_mul(n, n), qq = checked_mul(q, q);
  const auto full = checked_mul(n, row);
  if (std::max({n, q, nn, row}) > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF MO source response exceeds BLAS indexing");
  const auto small = checked_add(nn, qq), owned = checked_add(work.scratch_values, small);
  CudaDFMOSourceResponseDiagnostic result;
  result.owned_device_bytes = checked_add(bytes(owned), sizeof(int));
  result.numeric_capacity_bytes = checked_add(
      caller_bytes, checked_add(result.owned_device_bytes, bytes(checked_add(full, small))));
  if (result.numeric_capacity_bytes > budget)
    throw std::length_error("DF MO source response exceeds complete numeric budget");
  runtime::CudaDeviceScope scope(device);
  const auto admitted =
      generated::DFMOSourceResponseExecution::plan(budget - result.numeric_capacity_bytes);
  const auto prepare_start = std::chrono::steady_clock::now();
  generated::DFMOSourceResponseExecution execution(admitted, n, q, stream, result.gemms,
                                                   result.contraction_summands);
  result.preparation_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - prepare_start)
                              .count();
  result.binding_bytes = execution.selected().binding_bytes;
  result.numeric_capacity_bytes = checked_add(result.numeric_capacity_bytes, result.binding_bytes);
  result.provider_version = execution.provider_version();
  result.prepared_contractions = 8;
  const auto& candidate =
      generated::source_response_lowering_candidates[execution.selected().selected];
  result.provider = candidate.provider;
  result.candidate_identity = candidate.identity;
  const auto resources = execution.optional_resources();
  result.optional_workspace_bytes = resources.workspace_bytes;
  result.observed_provider_bytes = execution.retained_provider_bytes();
  // The host error destination outlives every stream-dependent allocation.
  int failed = 0;
  runtime::OwnedCudaBuffer<double> storage(device, owned, stream);
  runtime::OwnedCudaBuffer<int> error(device, 1, stream);
  auto* first = storage.get();
  auto* transformed = first + full;
  auto* raw_row = transformed + full;
  auto* bar_row = raw_row + row;
  auto* bar_c = bar_row + row;
  auto* bar_root = bar_c + nn;
  runtime::cuda_resource_check(cudaMemsetAsync(error.get(), 0, sizeof(int), stream));
  generated::pullback_df_mo_source_prepared(
      n, q, view.coefficients, view.inverse_root, view.bar_whitened, first, transformed, bar_row,
      bar_c, bar_root,
      [&](std::size_t mu) {
        read(mu, raw_row, stream);
        ++result.source_rows;
        result.source_values = checked_add(result.source_values, row);
        return raw_row;
      },
      [&](std::size_t mu, const double* values) {
        consume(mu, values, stream);
        ++result.output_rows;
        result.output_values = checked_add(result.output_values, row);
      },
      [&](std::size_t slot, const double* a, const double* b, double* c) {
        execution.execute(slot, stream, a, b, c, error.get());
      });
  if (result.source_rows != work.source_rows || result.source_values != work.raw_values ||
      result.output_rows != work.output_rows || result.output_values != work.output_values ||
      result.gemms != work.gemms || result.contraction_summands != work.contraction_summands)
    throw std::logic_error("DF MO source reverse execution differs from compiler work");
  finish(bar_c, bar_root, stream);
  runtime::cuda_resource_check(
      cudaMemcpyAsync(&failed, error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  runtime::cuda_resource_check(cudaStreamSynchronize(stream));
  result.scalar_d2h_bytes = sizeof(int);
  if (failed) throw std::runtime_error("nonfinite native DF MO source response arithmetic");
  return result;
}
}  // namespace generativeqc::posthf

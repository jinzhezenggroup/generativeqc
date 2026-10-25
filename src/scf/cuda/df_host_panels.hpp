#pragma once

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <string_view>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"

namespace generativeqc::scf::cuda_df {

/** Freeze only the requested policy, not changing machine headroom, in plan identity. */
inline bool host_panel_policy(std::size_t& limit, bool& automatic) noexcept {
  limit = std::numeric_limits<std::size_t>::max();
  const char* text = std::getenv("GENERATIVEQC_DF_HOST_MAX_BYTES");
  automatic = text == nullptr;
  if (automatic) return true;
  const std::string_view value(text);
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), limit);
  return !value.empty() && parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

/** Replay immutable raw A[pair,P] into an existing device tile. The owner keeps
 * pageable input alive through its stream drain. Capture retains on-device
 * generation; ordinary SCF uses eager execution when host values were admitted.
 * Metric whitening, densities and orbital projections are never cached here.
 */
inline generativeqc_status raw_panel(CudaDensityFittingJkPlan& plan, std::size_t system,
                                     std::size_t first, std::size_t pairs, std::size_t begin,
                                     std::size_t count, double* output, std::string& detail) {
  const auto cached_pairs = plan.naux ? plan.host_raw_three_center.size() / plan.naux : 0;
  if (system == 0 && pairs && count && first < cached_pairs && first < plan.matrix_elements &&
      pairs <= plan.matrix_elements - first && begin < plan.naux && count <= plan.naux - begin) {
    cudaStreamCaptureStatus capture{};
    auto error = cudaStreamIsCapturing(plan.stream, &capture);
    if (error != cudaSuccess) return cuda_failure(error, "inspect host DF replay", detail);
    if (capture == cudaStreamCaptureStatusNone) {
      // Split only the pair axis: every copied/generated rectangle keeps the
      // caller's contiguous auxiliary width and the original contraction order.
      const auto copied = std::min(pairs, cached_pairs - first);
      error = cudaMemcpy2DAsync(output, count * sizeof(double),
                                plan.host_raw_three_center.data() + first * plan.naux + begin,
                                plan.naux * sizeof(double), count * sizeof(double), copied,
                                cudaMemcpyHostToDevice, plan.stream);
      if (error != cudaSuccess) return cuda_failure(error, "replay host DF raw panel", detail);
      plan.host_panel_h2d_bytes += copied * count * sizeof(double);
      ++plan.host_panel_hits;
      runtime::cuda_trace::trace_counter("host_raw_panel_h2d_bytes",
                                         copied * count * sizeof(double));
      runtime::cuda_trace::trace_counter("host_raw_panel_hits", 1);
      first += copied;
      pairs -= copied;
      output += copied * count;
      if (!pairs) return GENERATIVEQC_STATUS_SUCCESS;
    }
  }
  runtime::cuda_trace::trace_counter("raw_panel_source_auxiliary_evaluations", pairs * count);
  return generate_cuda_density_fitting_raw_tile(plan.integral_source, system, first, pairs, begin,
                                                count, -1, plan.stream, output, detail);
}

}  // namespace generativeqc::scf::cuda_df

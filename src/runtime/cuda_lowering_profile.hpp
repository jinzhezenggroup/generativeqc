#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "runtime/cuda_fp64_matrix_policy.hpp"
#include "runtime/cuda_target_info.hpp"

namespace generativeqc::runtime {

/** Evidence-backed admission, not an inferred performance model.
 * The compiler adapter defines the axes and their independently qualified
 * bounds. Hardware facts have no product names; unknown FP64 capability cannot
 * qualify a matrix recipe. Costs, precision and simultaneous resource limits
 * remain inputs to select_native_lowering, not substitutes for these gates.
 */
struct CudaLoweringProfile {
  bool requires_fp64_matrix{};
  std::array<std::size_t, 4> minimum_extents{};
  int maximum_fp32_fp64_ratio{};
};

/** Check workload bounds before probing a GPU or preparing a provider. */
inline std::string_view cuda_lowering_workload_rejection(
    const CudaLoweringProfile& profile, const std::array<std::size_t, 4>& extents) noexcept {
  for (std::size_t axis = 0; axis != extents.size(); ++axis)
    if (extents[axis] < profile.minimum_extents[axis])
      return "workload is outside the qualified lowering profile";
  return {};
}

/** Apply hardware admission to an already bounded compiler workload.
 * Shared-memory, clock and memory-bus facts are available in the same target
 * record, but do not imply Tensor Core arithmetic support or endpoint speed.
 * Additional recipes need their own evidence rather than relaxed FP64 gates.
 */
inline std::string_view cuda_lowering_profile_rejection(
    const CudaLoweringProfile& profile, const CudaTargetInfo& target,
    const std::array<std::size_t, 4>& extents) noexcept {
  if (const auto rejection = cuda_lowering_workload_rejection(profile, extents); !rejection.empty())
    return rejection;
  if (profile.requires_fp64_matrix && !cuda_fp64_matrix_acceleration_supported(
                                          target.compute_capability_major, target.fp32_fp64_ratio))
    return "hardware is outside the qualified FP64 matrix profile";
  if (profile.maximum_fp32_fp64_ratio > 0 &&
      (target.fp32_fp64_ratio <= 0 || target.fp32_fp64_ratio > profile.maximum_fp32_fp64_ratio))
    return "hardware is outside the qualified FP64 throughput profile";
  return {};
}

}  // namespace generativeqc::runtime

#pragma once

namespace generativeqc::runtime {

/** Conservative admission for optional library FP64 matrix schedules.
 * A full-rate scalar FP64 device need not have FP64 matrix acceleration:
 * pre-Ampere devices therefore retain the incumbent even at ratio two.
 * Unknown/low FP64 throughput also retains the incumbent. This hardware
 * predicate is necessary, not sufficient: consumers still own independent
 * shape, precision, resource and complete-endpoint qualification gates.
 */
constexpr bool cuda_fp64_matrix_acceleration_supported(int compute_major,
                                                       int fp32_fp64_ratio) noexcept {
  return compute_major >= 8 && fp32_fp64_ratio > 0 && fp32_fp64_ratio <= 2;
}

}  // namespace generativeqc::runtime

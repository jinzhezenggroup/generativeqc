#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "runtime/cuda_target_info.hpp"

namespace generativeqc::tensor {

/** Primitive, exact-stack calibration for the retained FP64 product candidates.
 * This is provider evidence, not a scientific method option or a precision axis.
 */
struct CudaMatrixProductCalibration {
  double native_launch_nanoseconds{}, native_fp64_flops_per_nanosecond{},
      native_bytes_per_nanosecond{};
  double cublas_launch_nanoseconds{}, cublas_fp64_flops_per_nanosecond{},
      cublas_bytes_per_nanosecond{};
  bool operator==(const CudaMatrixProductCalibration& other) const noexcept {
    return native_launch_nanoseconds == other.native_launch_nanoseconds &&
           native_fp64_flops_per_nanosecond == other.native_fp64_flops_per_nanosecond &&
           native_bytes_per_nanosecond == other.native_bytes_per_nanosecond &&
           cublas_launch_nanoseconds == other.cublas_launch_nanoseconds &&
           cublas_fp64_flops_per_nanosecond == other.cublas_fp64_flops_per_nanosecond &&
           cublas_bytes_per_nanosecond == other.cublas_bytes_per_nanosecond;
  }
};
struct CudaMatrixProductProfile {
  std::size_t fallback_dimension_threshold{17};
  CudaMatrixProductCalibration calibration;
  /** Charge the complete measured first-use cost, without guessing SCF reuse. */
  double provider_setup_nanoseconds{};
};
struct CudaMatrixProductEstimate {
  std::uint64_t flops{}, native_semantic_bytes{}, provider_semantic_bytes{}, native_blocks{},
      native_waves{};
  bool operator==(const CudaMatrixProductEstimate& other) const noexcept {
    return flops == other.flops && native_semantic_bytes == other.native_semantic_bytes &&
           provider_semantic_bytes == other.provider_semantic_bytes &&
           native_blocks == other.native_blocks && native_waves == other.native_waves;
  }
};

namespace matrix_admission_detail {
inline std::uint64_t multiply(std::uint64_t first, std::uint64_t second) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return first && second > maximum / first ? maximum : first * second;
}
inline std::uint64_t add(std::uint64_t first, std::uint64_t second) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return second > maximum - first ? maximum : first + second;
}
inline std::uint64_t divide_up(std::uint64_t numerator, std::uint64_t denominator) noexcept {
  return denominator && numerator ? 1 + (numerator - 1) / denominator : 0;
}
inline bool positive(double value) noexcept { return value > 0 && std::isfinite(value); }
inline double roofline(double launch, std::uint64_t flops, std::uint64_t bytes, double flop_rate,
                       double byte_rate) noexcept {
  return launch +
         std::max(static_cast<double>(flops) / flop_rate, static_cast<double>(bytes) / byte_rate);
}
}  // namespace matrix_admission_detail

/** Provider-owned, bounded context admission before the immutable binding.
 * The native fallback needs no optional context. Creating an admitted context
 * does not select a new provider: CudaMatrixProductBinding still resolves each
 * canonical request and keeps its qualified incumbent/capture/resource checks.
 */
class CudaMatrixProductAdmission {
 public:
  bool requires_provider_context() const noexcept { return context_; }
  CudaMatrixProductAdmission available_if(bool available) const noexcept {
    auto admitted = *this;
    admitted.context_ = context_ && available;
    return admitted;
  }
  const CudaMatrixProductEstimate& estimate() const noexcept { return estimate_; }
  bool calibrated() const noexcept { return calibrated_; }
  double native_nanoseconds() const noexcept { return native_ns_; }
  double provider_nanoseconds() const noexcept { return provider_ns_; }
  bool operator==(const CudaMatrixProductAdmission& other) const noexcept {
    return estimate_ == other.estimate_ && context_ == other.context_ &&
           calibrated_ == other.calibrated_ && native_ns_ == other.native_ns_ &&
           provider_ns_ == other.provider_ns_;
  }

  static CudaMatrixProductAdmission resolve(const runtime::CudaTargetInfo& target,
                                            std::size_t dimension, std::size_t matrices,
                                            CudaMatrixProductProfile profile = {}) noexcept {
    using namespace matrix_admission_detail;
    CudaMatrixProductAdmission admission;
    if (!dimension || !matrices) return admission;
    const auto square = multiply(dimension, dimension);
    const auto cube = multiply(square, dimension);
    const auto outputs = multiply(matrices, square);
    auto& work = admission.estimate_;
    work.flops = multiply(multiply(2, matrices), cube);
    // The ordered kernel has one output per lane; tiled providers consume at
    // least A, B and C once. Calibrated effective rates include additional IO.
    work.native_semantic_bytes = multiply(outputs, add(multiply(16, dimension), 8));
    work.provider_semantic_bytes = multiply(multiply(24, matrices), square);
    constexpr std::uint64_t threads = 32;
    work.native_blocks = divide_up(outputs, threads);
    if (work.native_blocks && target.multiprocessor_count && target.maximum_blocks_per_sm &&
        target.maximum_threads_per_sm) {
      const auto resident =
          multiply(target.multiprocessor_count,
                   std::min<std::uint64_t>(target.maximum_blocks_per_sm,
                                           target.maximum_threads_per_sm / threads));
      work.native_waves = divide_up(work.native_blocks, resident);
    }
    const auto& rates = profile.calibration;
    admission.calibrated_ = positive(rates.native_launch_nanoseconds) &&
                            positive(rates.native_fp64_flops_per_nanosecond) &&
                            positive(rates.native_bytes_per_nanosecond) &&
                            positive(rates.cublas_launch_nanoseconds) &&
                            positive(rates.cublas_fp64_flops_per_nanosecond) &&
                            positive(rates.cublas_bytes_per_nanosecond) &&
                            std::isfinite(profile.provider_setup_nanoseconds) &&
                            profile.provider_setup_nanoseconds >= 0;
    if (!admission.calibrated_) {
      admission.context_ = dimension >= profile.fallback_dimension_threshold;
      return admission;
    }
    admission.native_ns_ =
        roofline(rates.native_launch_nanoseconds, work.flops, work.native_semantic_bytes,
                 rates.native_fp64_flops_per_nanosecond, rates.native_bytes_per_nanosecond);
    admission.provider_ns_ =
        profile.provider_setup_nanoseconds +
        roofline(rates.cublas_launch_nanoseconds, work.flops, work.provider_semantic_bytes,
                 rates.cublas_fp64_flops_per_nanosecond, rates.cublas_bytes_per_nanosecond);
    // Finite inputs can still overflow their estimates. An unrankable pair
    // must retain the incumbent rather than accidentally promote a candidate.
    if (!std::isfinite(admission.native_ns_) || !std::isfinite(admission.provider_ns_)) {
      admission.calibrated_ = false;
      admission.native_ns_ = admission.provider_ns_ = 0;
      admission.context_ = dimension >= profile.fallback_dimension_threshold;
      return admission;
    }
    admission.context_ = admission.provider_ns_ < admission.native_ns_;
    return admission;
  }

 private:
  CudaMatrixProductEstimate estimate_;
  bool context_{}, calibrated_{};
  double native_ns_{}, provider_ns_{};
};

}  // namespace generativeqc::tensor

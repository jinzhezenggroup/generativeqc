#pragma once

#include <cmath>
#include <span>
#include <stdexcept>

namespace generativeqc::tensor {

/** Streaming Neumaier sum of FP64 terms, in the caller's original order.
 *
 * The volatile term boundary forbids contraction of a caller's product into
 * the leading sum: the same rounded term must enter both sum and correction.
 * This owner requires ordinary IEEE arithmetic, without unsafe reassociation.
 * It allocates no storage, does not change point/product evaluation, and never
 * turns an overflowing or nonfinite leading sum back into a finite result.
 */
class CpuCompensatedSum {
 public:
  void add(double value) noexcept {
    const volatile double rounded_value = value;
    const double term = rounded_value;
    const double next = sum_ + term;
    if (std::isfinite(next)) {
      correction_ += std::abs(sum_) >= std::abs(term) ? (sum_ - next) + term : (term - next) + sum_;
    } else {
      // Keep the original IEEE nonfinite trajectory; callers retain their
      // existing finite checks and exception policy.
      correction_ = 0.0;
    }
    sum_ = next;
  }

  [[nodiscard]] double value() const noexcept { return sum_ + correction_; }

 private:
  double sum_{};
  double correction_{};
};

/** Signed FP64 dot product for production energy traces. Unlike a norm, this
 * preserves every term for indefinite/response densities and cancellation.
 * The scalar reference dot remains a separate, unchanged implementation.
 */
inline double cpu_compensated_dot(std::span<const double> a, std::span<const double> b) {
  if (a.size() != b.size()) throw std::invalid_argument("CPU compensated dot extents differ");
  CpuCompensatedSum result;
  for (std::size_t i = 0; i < a.size(); ++i) result.add(a[i] * b[i]);
  return result.value();
}

}  // namespace generativeqc::tensor

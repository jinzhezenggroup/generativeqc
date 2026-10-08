#pragma once

#include <cmath>
#include <cstdint>

namespace generativeqc::solver {

/** Operation outcomes are independent of any consumer's status ABI. */
enum class BroydenResult { success, invalid_argument, allocation_failed, numerical_failure };

/** Caller-selected Johnson modified-Broyden policy. Residual thresholds are
 * diagnostics; complete endpoint convergence remains the consumer's decision. */
struct BroydenPolicy {
  std::int64_t history_size{};
  double damping{};
  double rms_tolerance{};
  double maximum_tolerance{};

  [[nodiscard]] bool valid() const noexcept {
    return history_size > 0 && std::isfinite(damping) && damping > 0.0 && damping <= 1.0 &&
           std::isfinite(rms_tolerance) && rms_tolerance > 0.0 &&
           std::isfinite(maximum_tolerance) && maximum_tolerance > 0.0;
  }
};

/** Explicit encoding for caller-owned int32 status records. The provider does
 * not interpret other values written by a consumer. This preserves existing
 * storage representations without coupling the solver to a method's ABI. */
struct BroydenStatusEncoding {
  std::int32_t success{0};
  std::int32_t uninitialized{1};
  std::int32_t numerical_failure{2};

  [[nodiscard]] bool valid() const noexcept {
    return success != uninitialized && success != numerical_failure &&
           uninitialized != numerical_failure;
  }
  [[nodiscard]] bool operator==(const BroydenStatusEncoding& other) const noexcept {
    return success == other.success && uninitialized == other.uninitialized &&
           numerical_failure == other.numerical_failure;
  }
};

}  // namespace generativeqc::solver

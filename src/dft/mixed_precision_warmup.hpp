#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace generativeqc::dft {

/** An approximate SCF handoff, never a physical convergence certificate.
 * Zero controls preserve the incumbent approximate-stage termination criteria.
 * Positive controls bound approximate work or switch on the current physical
 * pre-DIIS maximum residual. The incumbent clears nonlinear history; a
 * separately admitted bounded history may supply proposals, never physical
 * stopping evidence. Every consumer must reset the energy epoch and certify
 * the final state using the original strict FP64 tolerances. */
struct MixedPrecisionWarmup {
  double switch_residual{};
  unsigned max_iterations{};

  bool enter_refinement(bool mixed_stage, bool converged, unsigned iterations,
                        unsigned target_limit, double residual_maximum) const noexcept {
    if (!mixed_stage || !std::isfinite(residual_maximum) || residual_maximum < 0.0) return false;
    const unsigned limit = max_iterations ? std::min(max_iterations, target_limit) : target_limit;
    return converged || iterations >= limit ||
           (switch_residual > 0.0 && iterations >= 2 && residual_maximum <= switch_residual);
  }
};

/** Parse immutable diagnostic controls before creating the prepared owner.
 * Malformed/nonfinite controls fail explicitly instead of changing arithmetic
 * silently. Strict-FP64 consumers need not resolve these optional controls. */
inline MixedPrecisionWarmup resolve_mixed_precision_warmup(const char* residual,
                                                           const char* iterations) {
  MixedPrecisionWarmup policy;
  if (residual) {
    const char* end = residual + std::strlen(residual);
    const auto parsed = std::from_chars(residual, end, policy.switch_residual);
    if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(policy.switch_residual) ||
        policy.switch_residual < 0.0)
      throw std::invalid_argument(
          "GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL requires a finite "
          "nonnegative residual (0 preserves convergence-only switching)");
  }
  if (iterations) {
    const char* end = iterations + std::strlen(iterations);
    const auto parsed = std::from_chars(iterations, end, policy.max_iterations);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
      throw std::invalid_argument(
          "GENERATIVEQC_KS_MIXED_MAX_ITERATIONS requires an unsigned "
          "iteration limit (0 preserves the original budget)");
  }
  return policy;
}

}  // namespace generativeqc::dft

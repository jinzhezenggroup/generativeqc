#pragma once

#include "dft/xc_point.hpp"

namespace generativeqc::dft::point {

/** Experimental same-equation PBE correlation lowering, not profile admission.
 * The caller must prove an equal-spin physical producer. Inputs, physical
 * normalization, exchange and returned coefficients remain FP64; the shared
 * first-derivative correlation graph uses FP32 registers and FP32 elementary
 * functions. Reductions are external and remain FP64. Response and final
 * stationarity checks must call the strict evaluator instead.
 *
 * The interior gate protects representability, not an observable error bound.
 * It cannot substitute for application-budget qualification. Extreme tails,
 * unsupported spin domains and nonfinite candidates use the unchanged FP64
 * scientific path, without clipping densities or changing the functional. */
struct MixedPointValue {
  Value value;
  bool fp32_correlation{};
};

/** Narrow both exchange and correlation algebra, with the same physical
 * normalization and tail contract as the strict consumer. No density clipping
 * or change of functional is permitted. Finite rounding still needs an
 * independent molecular force-budget gate. */
#if defined(__CUDACC__)
__host__ __device__
#endif
    inline MixedPointValue evaluate_pbe_full_mixed(bool pbe, const double rho[2],
                                                   const double gradient[2][3],
                                                   double exchange_scale = 1.0,
                                                   double correlation_scale = 1.0) {
  bool interior = true;
  for (unsigned spin = 0; spin < 2; ++spin) {
    interior = interior && rho[spin] >= 1e-12 && rho[spin] <= 1e6;
    for (unsigned axis = 0; axis < 3; ++axis)
      interior = interior && detail::finite(gradient[spin][axis]) &&
                 ::fabs(gradient[spin][axis]) <= 1e6 * rho[spin];
  }
  if (interior) {
    bool equal_spin = rho[0] == rho[1];
    for (unsigned axis = 0; axis < 3; ++axis)
      equal_spin = equal_spin && gradient[0][axis] == gradient[1][axis];
    const auto candidate = equal_spin ? evaluate_with_channels<4, true, float, float>(
                                            pbe, rho, gradient, exchange_scale, correlation_scale)
                                      : evaluate_with_channels<8, false, float, float>(
                                            pbe, rho, gradient, exchange_scale, correlation_scale);
    if (candidate.valid) return {candidate, true};
  }
  return {evaluate(pbe, rho, gradient, exchange_scale, correlation_scale), false};
}

#if defined(__CUDACC__)
__host__ __device__
#endif
    inline MixedPointValue evaluate_pbe_restricted_mixed(const double rho[2],
                                                         const double gradient[2][3],
                                                         double exchange_scale = 1.0,
                                                         double correlation_scale = 1.0) {
  bool interior = rho[0] == rho[1] && rho[0] >= 1e-12 && rho[0] <= 1e6 &&
                  detail::finite(exchange_scale) && detail::finite(correlation_scale) &&
                  exchange_scale >= 0.0 && correlation_scale > 0.0;
  for (unsigned axis = 0; axis < 3; ++axis)
    interior = interior && gradient[0][axis] == gradient[1][axis] &&
               detail::finite(gradient[0][axis]) && ::fabs(gradient[0][axis]) <= 1e6 * rho[0];
  if (interior) {
    const auto candidate = evaluate_with_channels<4, true, float>(
        true, rho, gradient, exchange_scale, correlation_scale);
    if (candidate.valid) return {candidate, true};
  }
  return {evaluate(true, rho, gradient, exchange_scale, correlation_scale), false};
}

}  // namespace generativeqc::dft::point

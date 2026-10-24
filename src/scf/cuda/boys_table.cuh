#pragma once

#include <cmath>
#include <type_traits>

#include "integrals/execution_qualifiers.hpp"
#include "integrals/range_moments.hpp"
#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained Boys table recurrence, shared by direct and one-electron
// evaluators with the existing precision and small-argument branches.
namespace generativeqc::scf::cuda_execution {

template <unsigned MaximumOrder, typename Scalar, bool ReuseReciprocal = false>
GENERATIVEQC_ERI_DEVICE inline void boys_values(Scalar argument, Scalar* values) {
  static_assert(MaximumOrder <= kMaximumCoulombResponseOrder);
  if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    // The positive incomplete-gamma series avoids the cancellation that made
    // the alternating table require FP64. Share its established mathematical
    // owner with range moments; only the mixed specialization narrows it.
    float moments[MaximumOrder + 1];
    const bool valid = generativeqc::integrals::range_detail::long_moments(
        MaximumOrder, argument.value, 1.0F, moments);
    // Preserve relative seed accuracy when the subsequent Coulomb multiplier
    // can amplify a subnormal moment's rounding. Normal tables remain FP32.
    if (valid && moments[MaximumOrder] < 0x1p-126F) {
      double accurate_values[MaximumOrder + 1];
      boys_values<MaximumOrder, double>(static_cast<double>(argument.value), accurate_values);
      for (unsigned order = 0; order <= MaximumOrder; ++order) {
        values[order] = MixedPrecisionFloat{accurate_values[order]};
      }
      return;
    }
    for (unsigned order = 0; order <= MaximumOrder; ++order) {
      values[order] = MixedPrecisionFloat{valid ? moments[order] : NAN};
    }
  } else {
    for (unsigned order = 0; order <= MaximumOrder; ++order) {
      values[order] = scalar<Scalar>(0.0);
    }
    // Low Boys orders tolerate upward recurrence much earlier than the generic
    // high-order path. Avoid a long alternating series for the dominant direct
    // s/p/d quartets once cancellation is bounded. These retained thresholds
    // require independent seed and Cartesian-derivative accuracy qualification.
    constexpr double series_threshold = MaximumOrder == 0   ? 1.0e-8
                                        : MaximumOrder == 1 ? 0.25
                                        : MaximumOrder == 2 ? 0.75
                                        : MaximumOrder == 3 ? 1.25
                                        : MaximumOrder == 4 ? 2.0
                                                            : 6.0;
    if (scalar_value(argument) < series_threshold) {
      // Evaluate only the highest requested Boys order by its convergent power
      // series. Lower orders follow from the stable downward recurrence, which
      // removes MaximumOrder duplicate series from every primitive quartet.
      Scalar term = scalar<Scalar>(1.0);
      Scalar sum = scalar<Scalar>(0.0);
      for (unsigned k = 0; k < 80; ++k) {
        sum = sum + term / static_cast<double>(2 * MaximumOrder + 2 * k + 1);
        term = term * (-1.0 * argument) / static_cast<double>(k + 1);
        if (fabs(scalar_value(term)) < 1.0e-18) break;
      }
      values[MaximumOrder] = sum;
      const Scalar exponential = qexp(-1.0 * argument);
      for (unsigned order = MaximumOrder; order > 0; --order) {
        values[order - 1] =
            (2.0 * argument * values[order] + exponential) / static_cast<double>(2 * order - 1);
      }
      return;
    }

    if constexpr (ReuseReciprocal) {
      // Opt-in arithmetic for compact MD J: one square root and reciprocal
      // serve all orders. The retained branch preserves other consumers'
      // established rounding while this FP64 variant is qualified separately.
      static_assert(std::is_same_v<Scalar, double>);
      const double root = sqrt(argument);
      values[0] = 0x1.c5bf891b4ef6bp-1 / root * erf(root);
      const double exponential = exp(-argument);
      const double inverse_twice_argument = 0.5 / argument;
      for (unsigned order = 1; order <= MaximumOrder; ++order)
        values[order] =
            ((2.0 * order - 1.0) * values[order - 1] - exponential) * inverse_twice_argument;
    } else {
      values[0] = 0.5 * qsqrt(scalar<Scalar>(kPi) / argument) * qerf(qsqrt(argument));
      const Scalar exponential = qexp(-1.0 * argument);
      for (unsigned order = 1; order <= MaximumOrder; ++order) {
        values[order] =
            ((2.0 * static_cast<double>(order) - 1.0) * values[order - 1] - exponential) /
            (2.0 * argument);
      }
    }
  }
}

}  // namespace generativeqc::scf::cuda_execution

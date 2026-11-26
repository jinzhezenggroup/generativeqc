#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "tensor/cpu_compensated_sum.hpp"

namespace generativeqc::core {

/** Geometry-independent focal-point coefficients, shared by native clients and
 * the Python endpoint. Basis identities/grouping and actual electronic work
 * remain the caller's responsibility; this plan is not a basis-error estimate. */
struct CbsExtrapolation {
  std::array<double, 2> hf_weights;
  std::array<double, 2> mp2_weights;
  std::array<double, 6> coefficients;

  /** Return total, HF limit, MP2 correlation limit and small-basis correction.
   * Subtract at the component level, before extrapolation: applying the six
   * expanded coefficients directly can amplify cancellation of total energies.
   * The force path uses the same original three-term addition order.
   */
  static std::array<double, 4> compose(const std::array<double, 2>& hf_weights,
                                       const std::array<double, 2>& mp2_weights,
                                       const std::array<double, 6>& values,
                                       bool compensated_energy) noexcept {
    tensor::CpuCompensatedSum hf_energy, mp2_energy;
    double hf = 0.0, mp2 = 0.0;
    for (std::size_t index = 0; index < 2; ++index) {
      const volatile double hf_term = hf_weights[index] * values[index];
      const volatile double mp2_term = mp2_weights[index] * (values[index + 2] - values[index]);
      if (compensated_energy) {
        hf_energy.add(hf_term);
        mp2_energy.add(mp2_term);
      } else {
        hf += hf_term;
        mp2 += mp2_term;
      }
    }
    const double delta = values[4] - values[5];
    if (!compensated_energy) return {(hf + mp2) + delta, hf, mp2, delta};
    hf = hf_energy.value();
    mp2 = mp2_energy.value();
    tensor::CpuCompensatedSum total;
    total.add(hf);
    total.add(mp2);
    total.add(delta);
    return {total.value(), hf, mp2, delta};
  }

  static CbsExtrapolation make(std::uint32_t low_cardinal, std::uint32_t high_cardinal,
                               double correlation_exponent, double hf_exponent,
                               bool exponential_hf) {
    if (low_cardinal < 2 || high_cardinal > 100 || low_cardinal >= high_cardinal)
      throw std::invalid_argument("CBS cardinal numbers must increase within [2,100]");
    for (const double exponent : {correlation_exponent, hf_exponent})
      if (!std::isfinite(exponent) || exponent <= 0)
        throw std::invalid_argument("CBS exponents must be finite and positive");
    const auto mp2 = limit_weights(-correlation_exponent *
                                   std::log(static_cast<double>(high_cardinal) / low_cardinal));
    const auto hf = exponential_hf ? limit_weights(-hf_exponent * (high_cardinal - low_cardinal))
                                   : std::array<double, 2>{0.0, 1.0};
    return {hf, mp2, {hf[0] - mp2[0], hf[1] - mp2[1], mp2[0], mp2[1], 1.0, -1.0}};
  }

 private:
  static std::array<double, 2> limit_weights(double log_ratio) {
    // expm1 preserves close-cardinal/small-exponent cancellation. A negative
    // infinite intermediate is the valid decaying-error limiting case.
    const double denominator = -std::expm1(log_ratio);
    if (!std::isfinite(denominator) || denominator <= 0)
      throw std::invalid_argument("CBS extrapolation parameters produce singular weights");
    const double high = 1.0 / denominator;
    const double low = -std::exp(log_ratio) / denominator;
    if (!std::isfinite(low) || !std::isfinite(high))
      throw std::invalid_argument("CBS extrapolation parameters produce nonfinite weights");
    return {low, high};
  }
};

}  // namespace generativeqc::core

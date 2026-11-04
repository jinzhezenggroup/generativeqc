#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace generativeqc::core {

/** Geometry-independent focal-point coefficients, shared by native clients and
 * the Python endpoint. Basis identities/grouping and actual electronic work
 * remain the caller's responsibility; this plan is not a basis-error estimate. */
struct CbsExtrapolation {
  std::array<double, 2> hf_weights;
  std::array<double, 2> mp2_weights;
  std::array<double, 6> coefficients;

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

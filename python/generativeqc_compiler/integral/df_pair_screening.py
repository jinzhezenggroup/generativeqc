"""Emit a spatial Coulomb-norm envelope for normalized Gaussian products.

For a nonnegative charge f, split its Coulomb potential at radius R:
V_f <= 2*pi*R^2*||f||inf + ||f||1/R. Minimizing R gives
||f||C^2 <= (3/2)*(4*pi)^(1/3)*||f||1^(5/3)*||f||inf^(1/3).
The Coulomb norm obeys the triangle inequality, so the envelope also bounds
signed contracted Cartesian/spherical expansions after absolute coefficients.

The Gaussian product theorem gives an isotropic majorant
exp(-mu*AB^2)*(r+PA)^la*(r+PB)^lb*exp(-p*r^2). Its radial L1 integral and
monomial supremum are elementary; no ERI, Boys function or auxiliary center
is required. Center derivatives use raised/lowered polynomial envelopes.
"""


def emit_df_pair_screening_cuda() -> str:
    """Emit cheap value and first-derivative envelopes for the through-f domain."""
    return r"""
#pragma once
#include <cmath>
#include <cstddef>
#ifndef GENERATIVEQC_DF_BOUND_DEVICE
#define GENERATIVEQC_DF_BOUND_DEVICE __host__ __device__
#endif
namespace generativeqc::scf::generated_df_screening {
// One extra degree covers a first center derivative of an orbital f shell.
// Invalid/overflowing envelopes retain work by returning infinity.
GENERATIVEQC_DF_BOUND_DEVICE inline double radial_pair_norm(
    unsigned la, unsigned lb, double alpha, double beta, double distance,
    double coefficient = 1.0) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double infinity = HUGE_VAL;
  if (la > 7 || lb > 7 || la + lb > 7 || !(alpha > 0) || beta < 0 || distance < 0 ||
      !std::isfinite(alpha) || !std::isfinite(beta) ||
      !std::isfinite(distance) || !std::isfinite(coefficient)) return infinity;
  if (coefficient == 0) return 0;
  const double p = alpha + beta;
  if (!std::isfinite(p)) return infinity;
  const double pa = distance * (beta / p), pb = distance * (alpha / p);
  double polynomial[8] = {1};
  unsigned degree = 0;
  for (unsigned center = 0; center < 2; ++center) {
    const unsigned angular = center == 0 ? la : lb;
    const double shift = center == 0 ? pa : pb;
    for (unsigned i = 0; i < angular; ++i) {
      ++degree;
      for (unsigned k = degree; k > 0; --k)
        polynomial[k] = polynomial[k] * shift + polynomial[k - 1];
      polynomial[0] *= shift;
    }
  }
  // Integral over R^3 of r^k exp(-p*r^2); even/odd recurrences avoid gamma.
  double moment[8] = {};
  moment[0] = pi * std::sqrt(pi) / (p * std::sqrt(p));
  moment[1] = 2 * pi / (p * p);
  for (unsigned k = 2; k <= degree; ++k)
    moment[k] = (k + 1) * moment[k - 2] / (2 * p);
  double mass = 0, peak = 0;
  for (unsigned k = 0; k <= degree; ++k) {
    mass += polynomial[k] * moment[k];
    const double maximum = k == 0 ? 1 : std::pow(k / (2 * p * std::exp(1.0)), .5 * k);
    peak += polynomial[k] * maximum;
  }
  if (!(mass > 0) || !(peak > 0) || !std::isfinite(mass) || !std::isfinite(peak))
    return infinity;
  const double gaussian = (alpha / p) * beta * distance * distance;
  if (!std::isfinite(gaussian)) return infinity;
  // Coefficients and the Gaussian attenuation enter in log space so a tiny
  // overlap cannot underflow before a large normalization factor is applied.
  // The factor 16 supplies FP64 headroom; this is an analytical envelope with
  // independently tested floating-point behavior, not interval arithmetic.
  const double log_norm = std::log(16.0) + .5 * std::log(1.5 * std::cbrt(4 * pi)) +
      (5.0 / 6.0) * std::log(mass) + std::log(peak) / 6 +
      std::log(std::abs(coefficient)) - gaussian;
  if (!std::isfinite(log_norm)) return infinity;
  return std::exp(std::fmax(log_norm, -640.0));
}

GENERATIVEQC_DF_BOUND_DEVICE inline double radial_pair_derivative_norm(
    unsigned la, unsigned lb, double alpha, double beta, double distance,
    double coefficient = 1.0) {
  if (la > 3 || lb > 3) return HUGE_VAL;
  // Bound the sum of the A and B derivative norms for ANY one Cartesian
  // direction. Translation then bounds C and every shared-atom accumulation.
  double result = 2 * alpha * radial_pair_norm(la + 1, lb, alpha, beta, distance, coefficient);
  result += 2 * beta * radial_pair_norm(la, lb + 1, alpha, beta, distance, coefficient);
  if (la) result += la * radial_pair_norm(la - 1, lb, alpha, beta, distance, coefficient);
  if (lb) result += lb * radial_pair_norm(la, lb - 1, alpha, beta, distance, coefficient);
  return result;
}
} // namespace generativeqc::scf::generated_df_screening
"""

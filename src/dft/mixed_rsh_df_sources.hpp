#ifndef GENERATIVEQC_DFT_MIXED_RSH_DF_SOURCES_HPP
#define GENERATIVEQC_DFT_MIXED_RSH_DF_SOURCES_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace generativeqc::dft {

/** Reserve the mixed bridge's derivative-only LR owner before preparation.
 * It remains additional to energy-only SCF storage even when reused. The
 * remaining allowance is for transient one-electron device work; compact
 * host publication keeps its independent original bound. DF J/K response
 * storage is not represented by this deliberately partial source budget. */
inline generativeqc_status mixed_rsh_df_primary_device_allowance(std::size_t maximum_bytes,
                                                                 std::size_t derivative_bytes,
                                                                 std::size_t& primary_bytes,
                                                                 std::string& detail) {
  primary_bytes = 0;
  if (!derivative_bytes || derivative_bytes >= maximum_bytes) {
    detail = "mixed RSH-DF LR derivative owner exceeds its additional device budget";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  primary_bytes = maximum_bytes - derivative_bytes;
  detail.clear();
  return GENERATIVEQC_STATUS_SUCCESS;
}

/** Recompose a fitted full-range primary and an exact Direct LR correction.
 *
 * Input rows are already multiplied by their *physical Fock-energy*
 * coefficients: [H', Pulay', J_DF', c_short K_DF(full)'] plus the independently
 * resolved (c_long-c_short) K_LR'.  In particular the correction is *not*
 * c_long K_LR', and DF never supplies an omega-dependent integral.
 *
 * Output rows match the ordinary RSH consumer:
 * [H', Pulay', J_DF', c_short (K_DF(full)'-K_LR'), c_long K_LR'].
 * Coefficients are taken from the resolved Fock terms, which already own
 * restricted/unrestricted spin factors and exchange signs.
 *
 * This pure source operation does not grant a public force capability.
 * A zero LR-correction coefficient cannot recover distinct SR/LR rows and
 * is deliberately not admitted here.
 */
inline generativeqc_status compose_mixed_rsh_df_integral_sources(
    std::span<const double> fitted_four, std::span<const double> direct_lr_correction,
    std::size_t coordinates, double fitted_exchange_coefficient,
    double correction_exchange_coefficient, std::vector<double>& output, std::string& detail) {
  output.clear();
  if (!coordinates || coordinates > std::numeric_limits<std::size_t>::max() / 5 ||
      fitted_four.size() != 4 * coordinates || direct_lr_correction.size() != coordinates ||
      !std::isfinite(fitted_exchange_coefficient) ||
      !std::isfinite(correction_exchange_coefficient) || correction_exchange_coefficient == 0.0) {
    detail = "mixed RSH-DF source shape or resolved exchange identity is invalid";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const double fraction = fitted_exchange_coefficient / correction_exchange_coefficient;
  const double long_fraction = 1.0 + fraction;
  if (!std::isfinite(fraction) || !std::isfinite(long_fraction)) {
    detail = "mixed RSH-DF resolved exchange ratio is nonfinite";
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
  std::vector<double> candidate(5 * coordinates);
  for (std::size_t i = 0; i < fitted_four.size(); ++i) {
    if (!std::isfinite(fitted_four[i])) {
      detail = "mixed RSH-DF fitted stationary source is nonfinite";
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    candidate[i] = fitted_four[i];
  }
  for (std::size_t i = 0; i < coordinates; ++i) {
    const double lr = direct_lr_correction[i];
    const double short_range = fitted_four[3 * coordinates + i] - fraction * lr;
    const double long_range = long_fraction * lr;
    if (!std::isfinite(lr) || !std::isfinite(short_range) || !std::isfinite(long_range)) {
      detail = "mixed RSH-DF recomposed exchange derivative is nonfinite";
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    candidate[3 * coordinates + i] = short_range;
    candidate[4 * coordinates + i] = long_range;
  }
  output = std::move(candidate);
  detail.clear();
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::dft

#endif  // GENERATIVEQC_DFT_MIXED_RSH_DF_SOURCES_HPP

#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace generativeqc::dft {
/** RKS uses D=2*Cocc*Cocc^T: the full-range physical exact-exchange
 * fraction alpha has the signed Fock coefficient -alpha/2. Authentication of
 * the full source/operator composition remains with the prepared KS owner. */
constexpr bool restricted_global_hybrid_exchange_coefficient(double coefficient) noexcept {
  return std::isfinite(coefficient) && coefficient < 0.0 && coefficient >= -0.5;
}

/** The resident final-state matrix algebra is functional-independent.
 * Direct global hybrids retain their 384-AO envelope. Pure-J models start at
 * 544 AOs, but exact-direct J stops below the failed 1088-AO moved endpoint.
 * Fitted J retains its separate qualification without that Direct-only hold.
 * All retain the strict RKS/FP64, token, library and scratch-lifetime guards. */
constexpr bool device_final_validation_default_eligible(
    std::size_t aos, unsigned spins, bool direct_global_hybrid, bool full_precision,
    bool full_range_coulomb_only = false, bool density_fitted_coulomb = false) noexcept {
  return spins == 1 && full_precision &&
         ((aos >= 384 && direct_global_hybrid) ||
          (aos >= 544 && full_range_coulomb_only && (density_fitted_coulomb || aos < 1088)));
}

/** An unset selector chooses the evidence-bound default; 0 is a durable opt-out.
 * Explicit 1 retains experimental admission outside the promoted default scope,
 * but cannot bypass resource, identity, lease or shared numerical gates. */
inline bool device_final_validation_requested(const char* setting, bool default_eligible) {
  if (!setting) return default_eligible;
  if (std::strcmp(setting, "0") == 0) return false;
  if (std::strcmp(setting, "1") == 0) return true;
  throw std::invalid_argument("GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION requires 0 or 1");
}
}  // namespace generativeqc::dft

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
 * Retain the measured 384-AO RKS/FP64/resource envelope and existing
 * strict token, library-handle and lifetime guards in the CUDA owner. */
constexpr bool device_final_validation_default_eligible(std::size_t aos, unsigned spins,
                                                        bool direct_global_hybrid,
                                                        bool full_precision) noexcept {
  return aos >= 384 && spins == 1 && direct_global_hybrid && full_precision;
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

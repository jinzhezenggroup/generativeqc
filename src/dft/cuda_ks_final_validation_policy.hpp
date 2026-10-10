#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace generativeqc::dft {
/** Promote the measured direct PBE0/RKS product owner, not every KS consumer.
 * The 384/768-AO endpoints qualify the lower size boundary. Larger owners use
 * the same nine-product schedule and bounded packets, with no new tile policy.
 * Resource and published-scratch-lease admission remain the caller's contract. */
constexpr bool device_final_validation_default_eligible(std::size_t aos, unsigned spins,
                                                        bool direct_pbe0,
                                                        bool full_precision) noexcept {
  return aos >= 384 && spins == 1 && direct_pbe0 && full_precision;
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

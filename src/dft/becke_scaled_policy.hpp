#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace generativeqc::dft {
/** Request the qualified representation unless explicitly disabled.
 * Resource/topology admission remains separate, and selection is captured at
 * owner creation rather than reread during force execution. */
inline bool binary_scaled_becke_requested(const char* setting) {
  if (!setting || std::strcmp(setting, "1") == 0) return true;
  if (std::strcmp(setting, "0") == 0) return false;
  throw std::invalid_argument("GENERATIVEQC_STATIONARY_BECKE_SCALED_PRODUCTS requires 0 or 1");
}

/** Field 7 cannot retain binary exponents and normalized-adjoint weights at
 * once. Reuse only admitted ordinary phased storage; never acquire more space. */
constexpr bool binary_scaled_becke_eligible(std::size_t atoms, bool phased, bool center_geometry,
                                            bool primitive_selected) noexcept {
  return atoms > 0 && atoms <= 128 && phased && center_geometry && !primitive_selected;
}

/** Conservative automatic crossover from complete endpoint qualification.
 * Smaller owners retain logarithmic products unless explicitly requested. */
constexpr bool binary_scaled_becke_automatic(std::size_t atoms) noexcept {
  return atoms >= 96 && atoms <= 128;
}
}  // namespace generativeqc::dft

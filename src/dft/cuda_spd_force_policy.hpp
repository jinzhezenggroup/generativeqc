#pragma once

#include <cstddef>

namespace generativeqc::dft {
/** Reuse spatial Coulomb states for qualified full-range Direct RKS hybrids.
 * The derivative recurrence is independent of the semilocal XC functional and
 * physical exchange fraction. Live source/geometry/resource guards still apply. */
constexpr bool spatial_spd_force_default_eligible(std::size_t aos, unsigned spins,
                                                  bool direct_global_hybrid,
                                                  bool full_precision) noexcept {
  return aos >= 384 && spins == 1 && direct_global_hybrid && full_precision;
}
}  // namespace generativeqc::dft

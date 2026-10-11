#pragma once

#include <cstddef>

namespace generativeqc::dft {
/** Default only the measured full-precision Direct PBE0/RKS consumer from
 * 384 public AOs. The method authenticates its composition separately; native
 * dispatch still requires the existing prefix/topology/resident leases. */
constexpr bool spatial_spd_force_default_eligible(std::size_t aos, unsigned spins, bool direct_pbe0,
                                                  bool full_precision) noexcept {
  return aos >= 384 && spins == 1 && direct_pbe0 && full_precision;
}
}  // namespace generativeqc::dft

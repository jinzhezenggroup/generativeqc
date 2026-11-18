#pragma once

#include <cstddef>

namespace generativeqc::dft::dispersion {

inline constexpr std::size_t kD3MaximumAtomsPerSystem = 4096;

/** Shared physical two-body BJ parameters; all distances are in bohr. */
struct D3Parameters {
  double s6{}, s8{}, a1{}, a2{}, s9{};
  // Zero means an unscreened reference path. Positive values are bohr.
  double cn_cutoff{}, pair_cutoff{}, pair_switch_width{};
};

}  // namespace generativeqc::dft::dispersion

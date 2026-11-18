#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace generativeqc::dft {

/** Validate Cartesian spin fields without changing the linked XC inputs.
 * AO products can round density to zero while a differentiated product remains
 * subnormal. The production graph owns its lazy vacuum branch, so those raw
 * underflow-scale gradients are admissible without clipping or a density floor.
 * Negative/nonfinite densities and normal-scale gradients at zero density still
 * refuse; this is not permission to project an arbitrary unphysical field.
 */
inline bool host_gga_spin_fields_admitted(double density, std::array<double, 3> gradient) {
  if (!std::isfinite(density) || density < 0) return false;
  return std::all_of(gradient.begin(), gradient.end(), [density](double component) {
    if (!std::isfinite(component)) return false;
    const auto classification = std::fpclassify(component);
    return density > 0 || classification == FP_ZERO || classification == FP_SUBNORMAL;
  });
}

}  // namespace generativeqc::dft

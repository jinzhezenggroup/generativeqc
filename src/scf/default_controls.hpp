#pragma once

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf::defaults {

/** Shared public SCF defaults, independent of backend, spin, model and grid.
 * The energy threshold matches GPU4PySCF's default conv_tol. Its orbital-
 * gradient threshold is sqrt(conv_tol), not a density tolerance: the density
 * value below is an additional native convergence control. Physical final-state
 * validation and stricter post-HF reference requirements remain authoritative.
 */
inline constexpr double energy_tolerance = 1e-9;
inline constexpr double density_tolerance = 1e-5;
inline constexpr double screening_tolerance = 1e-12;
inline constexpr unsigned policy_version = 2;

/** SCC has no GPU4PySCF counterpart; preserve its independently owned controls. */
struct ConvergenceControls {
  double energy, density;
};

inline constexpr ConvergenceControls for_method(generativeqc_method method) noexcept {
  return method == GENERATIVEQC_METHOD_GFN2_XTB
             ? ConvergenceControls{1e-10, 1e-8}
             : ConvergenceControls{energy_tolerance, density_tolerance};
}

}  // namespace generativeqc::scf::defaults

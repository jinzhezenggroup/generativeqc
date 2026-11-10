#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#define GENERATIVEQC_CONVERGENCE_HD __host__ __device__
#else
#define GENERATIVEQC_CONVERGENCE_HD
#endif

namespace generativeqc::scf {

/** Owner facts for admitting the relaxed Standard energy-only residual gate. */
struct HfStationarityContext {
  bool restricted;
  bool strict_fp64;
  bool all_electron;
  bool compute_forces;
  bool physical_reference;
};

/** Force publication retains the historical strict physical residual ceiling.
 * Loose public SCF controls alone do not establish derivative stationarity.
 * This admission is shared by iteration and final-state consumers; its norm
 * threshold is not itself a bound on independent force error.
 */
GENERATIVEQC_CONVERGENCE_HD constexpr bool standard_hf_stationarity_owner(
    const HfStationarityContext& context) noexcept {
  return context.restricted && context.strict_fp64 && context.all_electron &&
         !context.compute_forces && !context.physical_reference;
}

/** Standard endpoint controls, qualified against independent E/F references.
 * These are solver tolerances, not bounds on basis, fitting or quadrature error.
 * Either tighter public control retains the historical strict physical gates.
 * Fast's separately qualified mixed-arithmetic allowance is not changed here.
 */
GENERATIVEQC_CONVERGENCE_HD constexpr bool standard_scf_convergence_requested(
    double energy_tolerance, double density_tolerance) noexcept {
  return energy_tolerance >= 1e-9 && density_tolerance >= 1e-5 &&
         energy_tolerance <= std::numeric_limits<double>::max() &&
         density_tolerance <= std::numeric_limits<double>::max();
}

/** Keep iteration and final-state consumers on the same physical norm ceiling.
 * The AO maximum is deliberately not called GPU4PySCF's orbital-gradient norm.
 * Complete canonical proof and unchanged trace/idempotency checks remain owned
 * by the method. A looser request cannot grow this qualified ceiling.
 */
GENERATIVEQC_CONVERGENCE_HD constexpr double scf_stationarity_tolerance(
    double energy_tolerance, double density_tolerance, double strict_ceiling) noexcept {
  const double ceiling = standard_scf_convergence_requested(energy_tolerance, density_tolerance)
                             ? 1e-5
                             : strict_ceiling;
  return density_tolerance < ceiling ? density_tolerance : ceiling;
}

/** HF admission is an owner property, not a consequence of loose controls.
 * Explicit orbital-gradient owners instead compare the orthonormal restricted
 * physical norm with sqrt(energy_tolerance), never with an AO maximum ceiling.
 */
GENERATIVEQC_CONVERGENCE_HD constexpr double hf_stationarity_tolerance(
    double energy_tolerance, double density_tolerance, bool qualified_owner,
    bool orbital_gradient = false) noexcept {
  if (orbital_gradient) return std::sqrt(energy_tolerance);
  return qualified_owner ? scf_stationarity_tolerance(energy_tolerance, density_tolerance, 1e-8)
                         : (density_tolerance < 1e-8 ? density_tolerance : 1e-8);
}

/** Iteration and rebuilt force determinants must use the same norm comparison. */
GENERATIVEQC_CONVERGENCE_HD constexpr bool hf_physical_stationarity_met(
    double physical_norm, double energy_tolerance, double density_tolerance, bool qualified_owner,
    bool orbital_gradient = false) noexcept {
  const double tolerance = hf_stationarity_tolerance(energy_tolerance, density_tolerance,
                                                     qualified_owner, orbital_gradient);
  return physical_norm >= 0.0 &&
         (orbital_gradient ? physical_norm < tolerance : physical_norm <= tolerance);
}

}  // namespace generativeqc::scf

#undef GENERATIVEQC_CONVERGENCE_HD

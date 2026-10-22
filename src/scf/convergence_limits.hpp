#pragma once

#include <limits>

#if defined(__CUDACC__)
#define GENERATIVEQC_CONVERGENCE_HD __host__ __device__
#else
#define GENERATIVEQC_CONVERGENCE_HD
#endif

namespace generativeqc::scf {

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

/** HF admission is an owner property, not a consequence of loose controls. */
GENERATIVEQC_CONVERGENCE_HD constexpr double hf_stationarity_tolerance(
    double energy_tolerance, double density_tolerance, bool qualified_owner) noexcept {
  return qualified_owner ? scf_stationarity_tolerance(energy_tolerance, density_tolerance, 1e-8)
                         : (density_tolerance < 1e-8 ? density_tolerance : 1e-8);
}

}  // namespace generativeqc::scf

#undef GENERATIVEQC_CONVERGENCE_HD

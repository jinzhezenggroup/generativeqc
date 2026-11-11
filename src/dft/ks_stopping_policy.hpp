#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "dft/semilocal_family.hpp"

namespace generativeqc::dft {

/** Explicit opt-out of default orbital stopping, snapshotted by the owner.
 * This preserves stringent force-repeatability tasks without changing their
 * public E/D tolerances. Unknown values must not silently loosen convergence. */
inline bool strict_ks_convergence_requested(const char* value) {
  if (!value || !*value) return false;
  if (std::string_view(value) == "strict") return true;
  throw std::invalid_argument("GENERATIVEQC_CUDA_KS_CONVERGENCE requires empty or strict");
}

/** Default E/F endpoints share the restricted physical orbital-gradient gate.
 * Requesting forces must not silently change the SCF accuracy contract. A
 * full-range physical-reference export still needs its stricter canonical
 * contract; tighter public controls also retain the strict owner admission. */
constexpr bool standard_ks_stationarity_allowance(bool standard_owner, bool nonlocal_owner,
                                                  bool physical_reference) noexcept {
  return standard_owner && (nonlocal_owner || !physical_reference);
}

/** Complete compositions with independent default-grid early-stop evidence.
 * A point family alone is insufficient: custom mixing coefficients and generated
 * split hybrids retain the incumbent policy. In particular, the known M06-2X
 * default-grid force failure does not qualify an automatic stopping promotion.
 * Provider, precision, spin, resources and final canonicality are checked by the
 * prepared owner separately. Explicit opt-in remains an experimental control.
 */
constexpr bool early_final_probe_default_model(std::uint32_t functional, double semilocal_exchange,
                                               double semilocal_correlation,
                                               double exchange_coefficient,
                                               bool default_grid) noexcept {
  if (!default_grid || semilocal_correlation != 1.0) return false;
  if (functional == semilocal_family_code(SemilocalFamily::Pbe))
    return (semilocal_exchange == 1.0 && exchange_coefficient == 0.0) ||
           (semilocal_exchange == 0.75 && exchange_coefficient == -0.125);
  if (functional == semilocal_family_code(SemilocalFamily::R2scan))
    return semilocal_exchange == 1.0 && exchange_coefficient == 0.0;
  if (functional == semilocal_family_code(SemilocalFamily::B3lyp))
    return semilocal_exchange == 1.0 && exchange_coefficient == -0.1;
  return false;
}

/** Inputs from a completed physical F[D] evaluation, never a DIIS proposal. */
struct KsStoppingState {
  unsigned iterations{};
  bool warm_energy_baseline{}, full_energy_history{}, target_ready{};
  double energy_change{}, density_change{}, residual{}, maximum_residual{};
  double energy_tolerance{}, density_tolerance{}, stationarity_tolerance{};
};

/** Conservative stopping shared by the CUDA owner and executable host probes.
 * Strict comparisons also reject NaNs. A first cold iterate has no energy
 * history; a validated warm baseline may supply that history explicitly.
 */
constexpr bool ordinary_ks_converged(const KsStoppingState& state) noexcept {
  const bool history =
      state.iterations > 1 || (state.iterations == 1 && state.warm_energy_baseline);
  return history && state.full_energy_history && state.target_ready &&
         state.energy_change < state.energy_tolerance &&
         state.density_change < state.density_tolerance &&
         state.residual < state.stationarity_tolerance &&
         state.maximum_residual < state.stationarity_tolerance;
}

/** An early canonical proof must not reopen the energy-only AO-maximum
 * allowance for an imminent force/reference consumer. Its canonical-density
 * tolerance remains a separate final-state contract. */
constexpr bool early_ks_final_probe_stationary(const KsStoppingState& state,
                                               bool force_or_reference) noexcept {
  const double maximum_tolerance =
      std::min(force_or_reference ? state.stationarity_tolerance : 1e-8, state.density_tolerance);
  return state.residual < state.stationarity_tolerance &&
         state.maximum_residual < maximum_tolerance;
}

/** Physical MO stationarity shared by convergence and density confirmation.
 * AO RMS/maximum bounds do not imply this global occupied-virtual norm. A
 * confirmation hold must pass it before suppressing a new density proposal.
 */
inline bool orbital_ks_stationary(const KsStoppingState& state, double orbital_gradient) noexcept {
  return orbital_gradient >= 0.0 && orbital_gradient < std::sqrt(state.energy_tolerance);
}

/** Restricted molecular E/MO-gradient candidate, matching GPU4PySCF's norms.
 * The transformed physical commutator has Frobenius norm sqrt(2)*||2 F_vo||.
 * Density change is diagnostic here, not an additional SCF stopping condition.
 * Full-target energy history and the owner's canonical publication proof are
 * still mandatory; provisional/frozen VV10 operators cannot publish a state.
 */
inline bool orbital_ks_converged(const KsStoppingState& state, double orbital_gradient) noexcept {
  const bool history =
      state.iterations > 1 || (state.iterations == 1 && state.warm_energy_baseline);
  return history && state.full_energy_history && state.target_ready &&
         state.energy_change < state.energy_tolerance &&
         orbital_ks_stationary(state, orbital_gradient);
}

/** Hold a physical VV10 density only to establish its energy history.
 * The caller excludes provisional seed/frozen operators. Energy convergence
 * and a second target observation are deliberately not prerequisites: the
 * next unchanged-density evaluation supplies them. The active stationarity
 * gate must already pass, and a rejected canonical proof must advance the
 * owned proposal instead of repeatedly validating the same rejected state.
 * Strict/non-orbital consumers retain their original AO confirmation bounds.
 */
inline bool nonlocal_ks_density_confirmation(const KsStoppingState& state,
                                             double confirmation_tolerance, bool orbital_policy,
                                             double orbital_gradient,
                                             bool final_probe_rejected) noexcept {
  return !final_probe_rejected && state.density_change < state.density_tolerance &&
         state.residual < confirmation_tolerance &&
         state.maximum_residual < confirmation_tolerance &&
         (!orbital_policy || orbital_ks_stationary(state, orbital_gradient));
}

}  // namespace generativeqc::dft

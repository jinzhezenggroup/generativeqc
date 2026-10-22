#pragma once

#include <cstdint>

#include "dft/semilocal_family.hpp"

namespace generativeqc::dft {

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

}  // namespace generativeqc::dft

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "dft/cuda_stationary_binding.hpp"
#include "dft/semilocal_family.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::dft {

/** Stop proposing sub-tolerance density changes before a fresh energy check.
 * This selects a proposal, not convergence or an observable-error certificate.
 * Budget-Fock confirmation uses a residual one order below its 1e-4 physical
 * publication limit; strict callers retain 1e-9. Both the normalized and AO
 * maximum residual must pass, so the normalized diagnostic alone cannot select
 * confirmation. The caller
 * supplies an already-validated density tolerance, which is never weakened. */
inline double nonlocal_confirmation_tolerance(bool budget_stationarity,
                                              double density_tolerance) noexcept {
  return std::min(budget_stationarity ? 1e-5 : 1e-9, density_tolerance);
}

/** Frozen operators propose densities, never accepted physical states.
 * Oversolving that changing fixed-point problem below budget confirmation
 * accuracy adds strict J/K work without certifying the target operator.
 * Explicit strict solves retain their incumbent adaptive inner tolerance. */
inline double nonlocal_inner_tolerance(bool budget_stationarity, double maximum_residual,
                                       double density_tolerance) noexcept {
  if (budget_stationarity) return nonlocal_confirmation_tolerance(true, density_tolerance);
  const double target_gate = std::min(1e-9, density_tolerance);
  return std::clamp(maximum_residual * 1e-3, std::min(1e-12, target_gate * 0.1), 1e-7);
}

/** Private size-tagged Python transport. The original public KS descriptor
 * remains the prefix and retains its ABI/version. Only callers that verified
 * application_precision_version_v1 may pass the extended size; every pointer
 * still outlives preparation and the native owner copies the three allowances. */
struct ApplicationPrecisionKsOptions {
  generativeqc_ks_options base;
  double energy{}, maximum_force{}, rms_force{};
};
static_assert(offsetof(ApplicationPrecisionKsOptions, base) == 0);

/** Recognize the private request without reading past a legacy descriptor.
 * Full allowance validation remains with KS preparation. This early query
 * admits the unchanged strict seed provider, not any mixed target operator. */
inline bool application_precision_requested(const generativeqc_ks_options* options) noexcept {
  if (!options || options->abi_version != GENERATIVEQC_ABI_VERSION ||
      options->struct_size < sizeof(ApplicationPrecisionKsOptions))
    return false;
  const auto energy = reinterpret_cast<const ApplicationPrecisionKsOptions*>(options)->energy;
  return std::isfinite(energy) && energy > 0.0;
}

/** Budget admission for a provisional XC warmup, not a force-error estimator.
 * The published state still requires the incumbent FP64 physical convergence
 * and analytic gradient. These conservative qualification allowances keep
 * stricter requests off the approximate trajectory. No grid, basis, screening
 * threshold, convergence tolerance or final force arithmetic changes here. */
inline bool admit_application_xc_warmup(const std::array<double, 3>& allowances, std::size_t aos,
                                        unsigned maximum_angular, bool local_restricted_pbe,
                                        bool all_electron, bool exact_full_range) noexcept {
  for (const auto allowance : allowances)
    if (!std::isfinite(allowance) || allowance < 0.0) return false;
  return local_restricted_pbe && all_electron && exact_full_range && aos >= 24 && aos <= 256 &&
         maximum_angular <= 2 && allowances[0] >= 1e-6 &&
         (allowances[1] == 0.0 || allowances[1] >= 1e-5) &&
         (allowances[2] == 0.0 || allowances[2] >= 3e-6);
}

/** Shared public native-gradient shape domain, for strict and mixed execution.
 * The method owner separately authenticates the complete scientific model. */
inline bool admit_native_stationary_gradient_shape(std::size_t aos, std::size_t atoms,
                                                   std::size_t maximum_angular,
                                                   bool restricted) noexcept {
  return restricted && aos >= 24 && aos <= 384 && atoms >= 2 && atoms <= 48 && maximum_angular <= 2;
}

/** Budget-Fock candidate domain, independent of mixed-gradient capacity.
 * The prepared owner authenticates the requested model and the shared Direct
 * operator capabilities. A strict physical Fock closes the solve;
 * allowances select provisional arithmetic, not a per-molecule error bound. */
inline bool admit_application_fock(const std::array<double, 3>& allowances, std::size_t aos,
                                   std::size_t atoms, unsigned maximum_angular, bool restricted,
                                   bool all_electron, bool mixed_fock_available) noexcept {
  for (const auto allowance : allowances)
    if (!std::isfinite(allowance) || allowance < 0.0) return false;
  return restricted && all_electron && mixed_fock_available && aos >= 24 && aos <= 768 &&
         atoms >= 2 && atoms <= 96 && maximum_angular <= 2 && allowances[0] >= 1e-5 &&
         allowances[1] >= 1e-4 && (allowances[2] == 0.0 || allowances[2] >= 3e-5);
}

/** Independently qualified stationary RKS candidate at a strict final
 * density. These allowances select arithmetic, not a rigorous error bound for
 * an arbitrary molecule. Explicit observable targets still require an audit.
 * Zero force allowance cannot admit a mixed force, unlike an energy warmup. */
inline bool admit_application_gradient(const std::array<double, 3>& allowances, std::size_t aos,
                                       std::size_t atoms, std::size_t maximum_angular,
                                       bool qualified_restricted) noexcept {
  for (const auto allowance : allowances)
    if (!std::isfinite(allowance) || allowance < 0.0) return false;
  return admit_native_stationary_gradient_shape(aos, atoms, maximum_angular,
                                                qualified_restricted) &&
         allowances[0] >= 1e-6 && allowances[1] >= 1e-5 &&
         (allowances[2] == 0.0 || allowances[2] >= 3e-6);
}

/** Qualification identity for an already authenticated static binding.
 * These are complete full-range electronic models, not ingredient matches.
 * The new semilocal candidates retain their measured 24--96 AO domain until
 * larger complete endpoints are independently qualified. PBE0 keeps its
 * existing broader domain. The same native scheduling must be available to
 * ordinary FP64; a model identity alone never authorizes narrowed arithmetic. */
inline std::uint64_t stationary_gradient_model_qualification(const CudaStationaryBinding& binding,
                                                             std::size_t aos) noexcept {
  const auto& model = binding.model;
  if (model.spin_blocks != 1 || model.partition_iterations != 3 ||
      model.semilocal_correlation_scale != 1.0)
    return 0;
  if (model.functional == static_cast<std::uint32_t>(SemilocalFamily::Pbe) &&
      model.semilocal_exchange_scale == 0.75 && model.full_range_exchange == 0.25)
    return 1;
  if (aos > 96 || model.semilocal_exchange_scale != 1.0) return 0;
  if (model.functional == static_cast<std::uint32_t>(SemilocalFamily::Pbe) &&
      model.full_range_exchange == 0.0)
    return 2;
  if (model.functional == static_cast<std::uint32_t>(SemilocalFamily::B3lyp) &&
      model.full_range_exchange == 0.2)
    return 3;
  if (model.functional == static_cast<std::uint32_t>(SemilocalFamily::R2scan) &&
      model.full_range_exchange == 0.0)
    return 4;
  // Split-MGGA codes encode the pinned Libxc exchange registration. Require
  // the complete M06-2X work domain and exchange weight, not R2SCAN storage.
  if (model.functional == (0x20000U | 450U) && model.domain_version == 4 &&
      model.full_range_exchange == 0.54)
    return 6;
  return 0;
}

/** Narrow derivative arithmetic only after independent budget/shape admission.
 * A linked typed entry point is necessary, but never sufficient, evidence. */
inline std::uint64_t application_gradient_qualification(const CudaStationaryBinding& binding,
                                                        const std::array<double, 3>& allowances,
                                                        std::size_t aos, std::size_t atoms,
                                                        std::size_t maximum_angular) noexcept {
  if (!binding.geometry_fp32 || !admit_application_gradient(allowances, aos, atoms, maximum_angular,
                                                            binding.model.spin_blocks == 1))
    return 0;
  return stationary_gradient_model_qualification(binding, aos);
}

}  // namespace generativeqc::dft

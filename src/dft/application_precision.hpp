#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "generativeqc/generativeqc.h"

namespace generativeqc::dft {

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

/** Independently qualified stationary PBE0/RKS candidate at a strict final
 * density. These allowances select arithmetic, not a rigorous error bound for
 * an arbitrary molecule. Explicit observable targets still require an audit.
 * Zero force allowance cannot admit a mixed force, unlike an energy warmup. */
inline bool admit_application_gradient(const std::array<double, 3>& allowances, std::size_t aos,
                                       std::size_t atoms, std::size_t maximum_angular,
                                       bool pbe0_restricted) noexcept {
  for (const auto allowance : allowances)
    if (!std::isfinite(allowance) || allowance < 0.0) return false;
  return admit_native_stationary_gradient_shape(aos, atoms, maximum_angular, pbe0_restricted) &&
         allowances[0] >= 1e-6 && allowances[1] >= 1e-5 &&
         (allowances[2] == 0.0 || allowances[2] >= 3e-6);
}

}  // namespace generativeqc::dft

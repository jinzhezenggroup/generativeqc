#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "dft/cuda_stationary_binding.hpp"
#include "dft/semilocal_family.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::dft {

/** Force arithmetic and SCF arithmetic have separate admission contracts.
 * Budget-converged densities may feed the existing FP64 force consumer; only
 * strict SCF controls may authorize independently qualified mixed gradients. */
inline bool admit_stationary_force_controls(double energy, double density, double screening,
                                            bool budget_stationarity) noexcept {
  const std::array<double, 3> controls{energy, density, screening};
  const std::array<double, 3> ceilings = budget_stationarity
                                             ? std::array<double, 3>{1e-6, 1e-4, 1e-10}
                                             : std::array<double, 3>{1e-10, 1e-8, 1e-12};
  for (std::size_t index = 0; index < controls.size(); ++index)
    if (!std::isfinite(controls[index]) || controls[index] <= 0.0 ||
        controls[index] > ceilings[index])
      return false;
  return true;
}

inline constexpr unsigned nonlocal_proposal_iteration_limit = 16;
/** Preserve the incumbent twelve-Fock warmup's eleven initializer steps.
 * Extra mixed budget belongs to the target's frozen-potential proposal, not
 * to a more accurate solution of the deliberately incomplete seed operator. */
inline constexpr unsigned nonlocal_seed_iteration_limit = 11;

/** Bound one initializer, one mixed target, and one frozen-potential proposal.
 * Two iterations remain reserved for fresh strict targets. The provisional
 * inner residual selects a precision handoff, never physical convergence. */
inline unsigned nonlocal_mixed_proposal_limit(unsigned target_limit) noexcept {
  const auto proposal_limit = nonlocal_seed_iteration_limit + nonlocal_proposal_iteration_limit + 2;
  return std::min(proposal_limit, target_limit > 2 ? target_limit - 2 : 1U);
}

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

/** Executable native gradient shape; no empirical AO/atom upper bound here.
 * The linked profile, charged capacity, and physical final state gate execution.
 * The independently qualified no-audit population is tracked separately. */
inline bool admit_native_stationary_gradient_shape(std::size_t aos, std::size_t atoms,
                                                   std::size_t maximum_angular,
                                                   bool restricted) noexcept {
  return restricted && aos >= 24 && atoms >= 2 && maximum_angular <= 2;
}

/** Historical one-endpoint qualification, NOT a kernel or memory limit.
 * Larger supported shapes may execute FP32, but require a real FP64 audit
 * before Fast can publish their energy and forces. */
inline constexpr std::uint64_t kGradientAuditRequiredOffset = 100;
inline constexpr bool stationary_gradient_independently_qualified(
    std::uint64_t model, std::size_t aos, std::size_t atoms) noexcept {
  return model != 0 && aos <= (model == 1 ? 384U : 96U) &&
         atoms <= (model == 6 ? 32U : 48U);
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

/** Exact model match for an already authenticated compiled binding.
 * Shape-independent: numerical qualification is decided only at publication.
 * The shared native schedule and capacity must exist for either precision. */
inline std::uint64_t stationary_gradient_model_qualification(const CudaStationaryBinding& binding,
                                                             std::size_t /*aos*/) noexcept {
  const auto& model = binding.model;
  if (model.spin_blocks != 1 || model.partition_iterations != 3 ||
      model.semilocal_correlation_scale != 1.0)
    return 0;
  if (model.functional == static_cast<std::uint32_t>(SemilocalFamily::Pbe) &&
      model.semilocal_exchange_scale == 0.75 && model.full_range_exchange == 0.25)
    return 1;
  if (model.semilocal_exchange_scale != 1.0) return 0;
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

/** Select typed arithmetic only with complete model/budget/capability admission.
 * Distinguish historically qualified one-endpoint shapes from extrapolations
 * (including a relaxed-budget Fock density). Consumers MUST audit IDs >= 100
 * against a separate FP64 endpoint; neither shape nor finite values certify
 * an unmeasured gradient error. */
inline std::uint64_t application_gradient_qualification(const CudaStationaryBinding& binding,
                                                        const std::array<double, 3>& allowances,
                                                        std::size_t aos, std::size_t atoms,
                                                        std::size_t maximum_angular,
                                                        bool strict_density = true) noexcept {
  if (!binding.geometry_fp32 || !admit_application_gradient(allowances, aos, atoms, maximum_angular,
                                                            binding.model.spin_blocks == 1))
    return 0;
  const auto model = stationary_gradient_model_qualification(binding, aos);
  if (!model) return 0;
  return strict_density && stationary_gradient_independently_qualified(model, aos, atoms)
             ? model
             : model + kGradientAuditRequiredOffset;
}

}  // namespace generativeqc::dft

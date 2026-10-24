#include <limits>
#include <optional>
#include <span>

#include "api/error.hpp"
#include "generativeqc/generativeqc.h"
#include "scf/effort_policy.hpp"

extern "C" generativeqc_status generativeqc_scf_effort_decide_v1(
    const generativeqc_scf_effort_descriptor_v1* descriptor,
    generativeqc_scf_effort_decision_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (descriptor->flags & ~3u || descriptor->calibration_domain > 2 ||
      !descriptor->density_tolerances || !descriptor->level_count ||
      descriptor->level_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      descriptor->level_index >= descriptor->level_count)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    using namespace generativeqc::scf::effort;
    static_assert(static_cast<unsigned>(Action::tighten) == GENERATIVEQC_SCF_EFFORT_TIGHTEN);
    static_assert(static_cast<unsigned>(Reason::current_allowance) ==
                  GENERATIVEQC_SCF_EFFORT_CURRENT_ALLOWANCE);
    const Policy policy{descriptor->optimizer_force_tolerance,
                        descriptor->intermediate_energy_error,
                        descriptor->intermediate_force_fraction,
                        descriptor->max_intermediate_force_error,
                        descriptor->near_stationary_factor,
                        descriptor->relax_after,
                        bool(descriptor->flags & 1u)};
    const Calibration calibration{descriptor->energy_scale, descriptor->force_max_scale,
                                  descriptor->force_rms_scale, descriptor->diagnostic_floor};
    const auto decision =
        decide(policy, calibration,
               std::span(descriptor->density_tolerances,
                         static_cast<std::size_t>(descriptor->level_count)),
               {static_cast<std::size_t>(descriptor->level_index), descriptor->safe_streak},
               descriptor->flags & 2u ? std::optional(descriptor->current_force_max) : std::nullopt,
               descriptor->calibration_domain == 0);
    generativeqc_scf_effort_decision_v1 published{};
    published.struct_size = sizeof(published);
    published.abi_version = GENERATIVEQC_ABI_VERSION;
    published.action = static_cast<std::uint32_t>(decision.action);
    published.reason = static_cast<std::uint32_t>(decision.reason);
    published.level_index = decision.state.level_index;
    published.safe_streak = decision.state.safe_streak;
    published.allowed_energy_error = decision.allowed_energy_error;
    published.allowed_force_error = decision.allowed_force_error;
    if (decision.estimate) {
      published.has_estimate = 1;
      published.energy_abs = decision.estimate->energy_abs;
      published.force_max_abs = decision.estimate->force_max_abs;
      published.force_rms = decision.estimate->force_rms;
      published.diagnostic_value = decision.estimate->diagnostic_value;
    }
    *output = published;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

extern "C" generativeqc_status generativeqc_scf_effort_verify_v1(
    const generativeqc_scf_effort_final_descriptor_v1* descriptor,
    generativeqc_scf_effort_verification_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (descriptor->flags & ~7u || descriptor->reserved) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    const auto result = generativeqc::scf::effort::verify(
        {bool(descriptor->flags & 1u), bool(descriptor->flags & 2u), bool(descriptor->flags & 4u),
         descriptor->energy, descriptor->energy_change, descriptor->density_rms,
         descriptor->force_max_abs, descriptor->energy_tolerance, descriptor->density_tolerance,
         descriptor->optimizer_force_tolerance});
    *output = {sizeof(*output), GENERATIVEQC_ABI_VERSION, result.status, result.unmet_gates};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace generativeqc::scf::progressive {

/** Observed metadata, never a serialized final status. Identities and Python
 * object/array shape validation belong to the typed frontend; this shared
 * decision owns numerical gates, budget precedence and strict-final policy.
 * It does not rebuild a Fock matrix or certify an observable error estimate. */
struct VerificationInput {
  std::uint32_t flags{}, accuracy{};
  std::uint64_t iterations{}, iteration_limit{}, fock_builds{}, fock_limit{};
  double energy{}, energy_change{}, density_rms{}, residual_rms{};
  double audit_rms{}, audit_max{}, audit_energy_difference{};
  double energy_tolerance{}, density_tolerance{}, residual_tolerance{};
};

enum Observation : std::uint32_t {
  model_matches = 1u << 0,
  provider_matches = 1u << 1,
  hashes_match = 1u << 2,
  succeeded = 1u << 3,
  converged = 1u << 4,
  residual_present = 1u << 5,
  audit_present = 1u << 6,
  audit_target_matches = 1u << 7,
  audit_failed = 1u << 8,
  audit_budget_exhausted = 1u << 9,
  observables_complete = 1u << 10,
  precision_matches = 1u << 11,
  final_fp64 = 1u << 12,
  automatic_precision = 1u << 13,
  strict_refinement = 1u << 14,
  accuracy_present = 1u << 15,
  accuracy_target_matches = 1u << 16,
  fock_limit_present = 1u << 17,
  fock_count_present = 1u << 18,
};

/** Reasons follow public diagnostic order; budget reasons are kept separate
 * so a valid target can still report a resource/work-budget failure. */
struct VerificationDecision {
  std::uint32_t status{}, target_reasons{}, budget_reasons{}, cleanup{}, accuracy{};
};

inline bool valid(const VerificationInput& input) noexcept {
  if (input.flags >> 19 || input.accuracy > 6) return false;
  for (const auto tolerance :
       {input.energy_tolerance, input.density_tolerance, input.residual_tolerance})
    if (!std::isfinite(tolerance) || tolerance <= 0.0) return false;
  // The typed audit already requires finite, nonnegative diagnostics. Apply
  // the same admission at the C boundary before interpreting its values.
  if (input.flags & audit_present)
    for (const auto value : {input.audit_rms, input.audit_max, input.audit_energy_difference})
      if (!std::isfinite(value) || value < 0.0) return false;
  return true;
}

/** Preserve the existing fail-closed HF verification policy. Budget failures
 * take precedence in status, without erasing physical failures or independent
 * accuracy disposition. Audit max (not RMS) is the physical gate when present. */
inline VerificationDecision verify(const VerificationInput& input) noexcept {
  VerificationDecision result;
  const auto has = [&](Observation flag) { return (input.flags & flag) != 0; };
  const auto fail = [&](unsigned bit, bool unmet) {
    if (unmet) result.target_reasons |= 1u << bit;
  };
  fail(0, !has(model_matches));
  fail(1, !has(provider_matches));
  fail(2, !has(hashes_match));
  fail(3, !has(succeeded) || !has(converged));
  fail(4, !std::isfinite(input.energy));
  fail(5, !std::isfinite(input.energy_change) ||
              std::abs(input.energy_change) > input.energy_tolerance);
  fail(6, !std::isfinite(input.density_rms) || input.density_rms > input.density_tolerance);
  if (has(audit_present)) {
    fail(7, !has(audit_target_matches));
    fail(8, input.audit_energy_difference > std::max(1e-10, input.energy_tolerance));
  } else {
    fail(9, has(audit_failed));
  }
  const auto residual = has(audit_present) ? input.audit_rms : input.residual_rms;
  const bool missing = (!has(audit_present) && !has(residual_present)) || !std::isfinite(residual);
  fail(10, missing);
  fail(11,
       !missing && (has(audit_present) ? input.audit_max : residual) > input.residual_tolerance);
  fail(12, !has(observables_complete));
  if (!has(precision_matches)) {
    fail(13, true);
  } else if (!has(final_fp64)) {
    fail(14, true);
  } else if (has(automatic_precision)) {
    fail(15, !has(strict_refinement));
    result.cleanup = has(strict_refinement) ? 2 : 1;
  } else {
    result.cleanup = 3;
  }
  if (has(accuracy_present)) {
    fail(16, !has(accuracy_target_matches));
    if (has(accuracy_target_matches)) result.accuracy = input.accuracy;
  }
  if (has(audit_budget_exhausted)) result.budget_reasons |= 1u;
  if (input.iterations > input.iteration_limit) result.budget_reasons |= 2u;
  if (has(fock_limit_present)) {
    if (!has(fock_count_present))
      result.budget_reasons |= 4u;
    else if (input.fock_builds > input.fock_limit)
      result.budget_reasons |= 8u;
  }
  if (result.budget_reasons)
    result.status = 3;
  else if (result.target_reasons || (result.accuracy >= 1 && result.accuracy <= 3))
    result.status = 2;
  else if (result.accuracy == 4)
    result.status = 1;
  return result;
}

}  // namespace generativeqc::scf::progressive

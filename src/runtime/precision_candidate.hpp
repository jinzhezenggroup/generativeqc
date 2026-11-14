#pragma once

#include <algorithm>
#include <span>
#include <string_view>

#include "generativeqc/precision_admission.h"

namespace generativeqc::runtime {

/** Interpret only the incumbent detached gradient receipt domains. Producers
 * still own actual arithmetic/provider admission and physical state binding. */
inline bool qualified_candidate_gradient(
    const generativeqc_precision_candidate_context_v1& candidate, std::string_view name) noexcept {
  if (!candidate.gradient_present || !candidate.gradient_complete) return false;
  if (candidate.gradient_schema_version >= 1 && candidate.gradient_schema_version <= 3) {
    constexpr std::string_view names[]{"pbe0-rks-budget-fp32-v1",  "pbe-rks-budget-fp32-v1",
                                       "b3lyp-rks-budget-fp32-v1", "r2scan-rks-budget-fp32-v1",
                                       "rhf-budget-fp32-v1",       "rhf-df-shell-budget-fp32-v1",
                                       "m06-2x-rks-budget-fp32-v1"};
    return std::find(std::begin(names), std::end(names), name) != std::end(names);
  }
  if (candidate.gradient_schema_version != 4) return false;
  if (candidate.gradient_owner == 1) {
    constexpr std::string_view names[]{"pbe-df-rks-budget-fp32-v1",
                                       "pbe0-df-rks-budget-fp32-v1",
                                       "b3lyp-df-rks-budget-fp32-v1",
                                       "r2scan-df-rks-budget-fp32-v1",
                                       "m06-2x-df-rks-budget-fp32-v1",
                                       "pbe-df-rks-treutler-budget-fp32-v1",
                                       "pbe0-df-rks-treutler-budget-fp32-v1",
                                       "b3lyp-df-rks-treutler-budget-fp32-v1",
                                       "r2scan-df-rks-treutler-budget-fp32-v1",
                                       "m06-2x-df-rks-treutler-budget-fp32-v1",
                                       "pbe-df-rks-treutler-xc-budget-fp32-v1",
                                       "r2scan-df-rks-treutler-xc-budget-fp32-v1"};
    return std::find(std::begin(names), std::end(names), name) != std::end(names);
  }
  return candidate.gradient_owner == 2 &&
         (name == "wb97m-v-rks-budget-fp32-v1" || name == "wb97m-v-df-rks-budget-fp32-v1");
}

/** Bind electronic receipt interpretation to any complete geometry correction. */
inline generativeqc_precision_candidate_v1 candidate_gradient(
    const generativeqc_precision_candidate_context_v1& candidate,
    std::string_view canonical_composite_identity) noexcept {
  generativeqc_precision_candidate_v1 result{
      sizeof(result), GENERATIVEQC_ABI_VERSION, 1, 0, 0, 0, {}};
  const auto electronic = candidate.qualification_present
                              ? std::string_view(candidate.qualification)
                          : candidate.gradient_schema_version == 1 ? "pbe0-rks-budget-fp32-v1"
                                                                   : "unknown";
  auto qualified = qualified_candidate_gradient(candidate, electronic);
  std::string_view gradient = "no-application-qualified-fp32-gradient-route";
  if (qualified && candidate.correction_present) {
    const bool composite_electronic = electronic == "r2scan-rks-budget-fp32-v1" ||
                                      electronic == "r2scan-df-rks-budget-fp32-v1" ||
                                      electronic == "r2scan-df-rks-treutler-budget-fp32-v1" ||
                                      electronic == "r2scan-df-rks-treutler-xc-budget-fp32-v1";
    qualified = composite_electronic && !canonical_composite_identity.empty() &&
                candidate.correction_model_identity == canonical_composite_identity &&
                candidate.correction_result_flags == 63;
    if (!qualified) result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_CORRECTION_UNQUALIFIED;
  }
  result.qualified_gradient = qualified;
  if (qualified) {
    result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_GRADIENT_QUALIFIED;
    if (candidate.gradient_capability_fallback)
      result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_GEOMETRY_FALLBACK;
    if (candidate.correction_present) {
      constexpr std::string_view prefix = "r2scan-3c-";
      std::copy(prefix.begin(), prefix.end(), result.gradient_admission);
      gradient = electronic.substr(std::string_view("r2scan-").size());
      std::copy(gradient.begin(), gradient.end(), result.gradient_admission + prefix.size());
    } else {
      gradient = electronic;
      std::copy(gradient.begin(), gradient.end(), result.gradient_admission);
    }
  } else {
    std::copy(gradient.begin(), gradient.end(), result.gradient_admission);
  }
  return result;
}

/** Select a route after the shared work policy has consumed the same facts.
 * Explicit targets take precedence over all qualification shortcuts. Missing
 * census remains unknown, including the legacy strict-XC proposal path. */
inline generativeqc_precision_candidate_v1 precision_candidate(
    const generativeqc_precision_candidate_context_v1& candidate,
    const generativeqc_precision_admission_context_v1& context,
    const generativeqc_precision_provenance& work, const generativeqc_precision_work_detail& detail,
    std::span<const generativeqc_precision_operator_record> operators,
    const generativeqc_precision_admission_v1& admitted,
    generativeqc_precision_candidate_v1 result) noexcept {
  const bool qualified = result.qualified_gradient;
  const auto electronic = candidate.qualification_present
                              ? std::string_view(candidate.qualification)
                          : candidate.gradient_schema_version == 1 ? "pbe0-rks-budget-fp32-v1"
                                                                   : "unknown";
  if (candidate.explicit_target) return result;
  result.disposition = GENERATIVEQC_PRECISION_CANDIDATE_STRICT;
  // Presets resolve their allowance before this boundary. Neither the public
  // mode name nor relaxed SCF tolerances alone authenticate an observable budget.
  if (!candidate.budget_requested) return result;
  const bool complete = context.census_presence == 1 && work.operator_work_counters_valid &&
                        detail.complete && detail.operator_inventory_complete;
  if (candidate.budget_native && candidate.budget_fock &&
      (!candidate.gradient_present || candidate.gradient_returned_mode == 2 || qualified) &&
      (admitted.strict_warm || admitted.budget_fock)) {
    result.disposition = admitted.strict_warm ? GENERATIVEQC_PRECISION_CANDIDATE_WARM_FOCK
                                              : GENERATIVEQC_PRECISION_CANDIDATE_FOCK;
  } else if (candidate.budget_native && !candidate.budget_fock && candidate.forces_requested &&
             qualified &&
             ((candidate.gradient_schema_version >= 2 && candidate.gradient_schema_version <= 4) ||
              (complete && work.strict_stage_fock_builds && work.final_residual_audits)) &&
             (candidate.geometry_points ||
              (candidate.gradient_schema_version == 3 &&
               candidate.weighted_task_counters_collected && candidate.weighted_tasks &&
               candidate.primitive_sources)) &&
             (candidate.gradient_returned_mode == 1 || candidate.gradient_returned_mode == 2)) {
    result.disposition = candidate.gradient_returned_mode == 1
                             ? GENERATIVEQC_PRECISION_CANDIDATE_MIXED_GRADIENT
                             : GENERATIVEQC_PRECISION_CANDIDATE_STRICT_GRADIENT;
    if (candidate.gradient_retries)
      result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_GRADIENT_RETRIED;
    else if (candidate.gradient_schema_version != 4 && electronic != "rhf-budget-fp32-v1" &&
             electronic != "rhf-df-shell-budget-fp32-v1" && !candidate.grid_points)
      result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_GRID_UNAVAILABLE;
    if (candidate.execution_retries)
      result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_EXECUTION_RETRIED;
  } else if (candidate.budget_native && !candidate.budget_fock && !candidate.correction_present &&
             complete && candidate.mixed_fock_count_present && work.mixed_stage_fock_builds > 0 &&
             work.strict_stage_fock_builds >= 2 && work.strict_refinement_applied &&
             work.final_residual_audits > 0 &&
             std::any_of(operators.begin(), operators.end(), [](const auto& row) {
               return row.kind == GENERATIVEQC_PRECISION_OPERATOR_XC &&
                      row.compute_dtype == GENERATIVEQC_PRECISION_DTYPE_FP32 &&
                      row.accumulation_dtype == GENERATIVEQC_PRECISION_DTYPE_FP32 &&
                      row.arithmetic_mode == GENERATIVEQC_PRECISION_ARITHMETIC_MIXED && row.count;
             })) {
    result.disposition = GENERATIVEQC_PRECISION_CANDIDATE_STRICT_XC;
    if (candidate.execution_retries)
      result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_EXECUTION_RETRIED;
  } else if (admitted.standard_fallback) {
    result.disposition = GENERATIVEQC_PRECISION_CANDIDATE_STANDARD;
    result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_STANDARD_FALLBACK;
  } else {
    result.disposition = GENERATIVEQC_PRECISION_CANDIDATE_AUDIT;
    result.reasons |= GENERATIVEQC_PRECISION_CANDIDATE_BUDGET_AUDIT;
  }
  return result;
}

}  // namespace generativeqc::runtime

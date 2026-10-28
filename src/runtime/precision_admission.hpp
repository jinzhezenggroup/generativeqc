#pragma once

#include <cstdint>
#include <span>

#include "generativeqc/precision_admission.h"

namespace generativeqc::runtime {

/** Application evidence policy shared by language frontends. This consumes
 * execution-owned diagnostics; it neither schedules numerical work nor turns
 * historical qualification into an observable error bound. Unknown dtypes and
 * incomplete inventories cannot establish a strict returned state. */
inline generativeqc_precision_admission_v1 precision_admission(
    const generativeqc_precision_admission_context_v1& c,
    const generativeqc_precision_provenance& p, const generativeqc_precision_work_detail& detail,
    std::span<const generativeqc_precision_work_event> events,
    std::span<const generativeqc_precision_operator_record> operators) noexcept {
  constexpr auto fp64 = GENERATIVEQC_PRECISION_DTYPE_FP64;
  constexpr auto fp32 = GENERATIVEQC_PRECISION_DTYPE_FP32;
  constexpr auto strict = GENERATIVEQC_PRECISION_ARITHMETIC_STRICT;
  constexpr auto mixed = GENERATIVEQC_PRECISION_ARITHMETIC_MIXED;
  const auto bit = [](auto kind) { return std::uint32_t{1} << kind; };
  std::uint32_t mixed_work = 0, observed = 0;
  bool strict_j = true, has_j = false, mixed_proposal = false, all_strict = true;
  for (const auto& row : operators) {
    if (!row.count) continue;
    const bool strict_row = row.arithmetic_mode == strict && row.storage_dtype == fp64 &&
                            row.compute_dtype == fp64 && row.accumulation_dtype == fp64 &&
                            row.reduction_dtype == fp64;
    all_strict = all_strict && strict_row;
    // Unknown operator kinds cannot prove a required source. Do not shift by
    // an unchecked external enum; future non-required kinds remain harmless.
    if (row.kind > 0 && row.kind < 32) {
      observed |= bit(row.kind);
      if (row.compute_dtype == fp32 && row.accumulation_dtype == fp64 &&
          row.arithmetic_mode == mixed)
        mixed_work |= bit(row.kind);
    }
    if (row.kind == GENERATIVEQC_PRECISION_OPERATOR_COULOMB_J) {
      has_j = true;
      strict_j = strict_j && strict_row;
    }
    if ((row.kind == GENERATIVEQC_PRECISION_OPERATOR_XC ||
         row.kind == GENERATIVEQC_PRECISION_OPERATOR_MATRIX_PRODUCT ||
         row.kind == GENERATIVEQC_PRECISION_OPERATOR_DENSITY_BUILD) &&
        row.compute_dtype == fp32 && row.arithmetic_mode == mixed)
      mixed_proposal = true;
  }
  auto required_mixed = bit(GENERATIVEQC_PRECISION_OPERATOR_COULOMB_J) |
                        bit(GENERATIVEQC_PRECISION_OPERATOR_COULOMB_RECURRENCE);
  if (c.full_range_exchange) required_mixed |= bit(GENERATIVEQC_PRECISION_OPERATOR_EXCHANGE_K);
  const bool pure_strict_j =
      !c.full_range_exchange && !c.range_exchange && has_j && strict_j && mixed_proposal;
  const bool complete = c.census_presence == 1 && p.operator_work_counters_valid &&
                        detail.complete && detail.operator_inventory_complete;
  const bool budget_fock = c.ks_model && complete &&
                           ((mixed_work & required_mixed) == required_mixed || pure_strict_j) &&
                           p.strict_stage_fock_builds >= 2 && p.strict_refinement_applied &&
                           p.final_residual_audits > 0;

  auto required_strict = bit(GENERATIVEQC_PRECISION_OPERATOR_COULOMB_J) |
                         bit(GENERATIVEQC_PRECISION_OPERATOR_XC) |
                         bit(GENERATIVEQC_PRECISION_OPERATOR_FOCK_ASSEMBLY) |
                         bit(GENERATIVEQC_PRECISION_OPERATOR_PHYSICAL_RESIDUAL);
  if (c.full_range_exchange || c.range_exchange)
    required_strict |= bit(GENERATIVEQC_PRECISION_OPERATOR_EXCHANGE_K);
  if (c.nonlocal_correlation)
    required_strict |= bit(GENERATIVEQC_PRECISION_OPERATOR_NONLOCAL_CORRELATION);
  bool bound_fock = false, bound_audit = false;
  if (detail.owner_id && detail.returned_solve_epoch && detail.returned_state_generation) {
    for (const auto& event : events) {
      if (event.owner_id != detail.owner_id || event.solve_epoch != detail.returned_solve_epoch ||
          event.state_generation != detail.returned_state_generation)
        continue;
      bound_fock |= event.kind == GENERATIVEQC_PRECISION_EVENT_STRICT_FOCK;
      bound_audit |= event.kind == GENERATIVEQC_PRECISION_EVENT_FINAL_AUDIT;
    }
  }
  const bool strict_warm = c.ks_model && c.warm_start_used && !c.warm_start_fallback && complete &&
                           p.effective_bits == 64 && p.mixed_stage_fock_builds == 0 &&
                           p.strict_stage_fock_builds >= 1 && p.final_residual_audits >= 1 &&
                           (!c.gradient_present || c.gradient_fp64 || c.gradient_qualified) &&
                           all_strict && (observed & required_strict) == required_strict &&
                           bound_fock && bound_audit;

  bool controls = true;
  for (unsigned index = 0; index != 3; ++index)
    controls = controls && c.requested_controls[index] > 0 &&
               c.requested_controls[index] <= c.standard_controls[index];
  const bool strict_work =
      c.census_presence == 1 && p.operator_work_counters_valid && p.strict_stage_fock_builds > 0 &&
      (p.final_residual_audits > 0 || (c.hartree_fock && p.skipped_final_fock_builds > 0));
  // Fitted HF's established physical-final contract predates the optional
  // census. Explicitly unavailable counters must remain different from absent
  // metadata; this exception never manufactures a work count.
  const bool fitted_hf = c.hartree_fock && c.density_fitted && c.census_presence == 2 &&
                         !p.operator_work_counters_valid && !c.gradient_present;
  const bool standard = c.fp64_candidate && c.converged && c.cuda_execution &&
                        c.energy_abs >= 1e-5 && c.force_max_abs >= 1e-4 &&
                        (c.force_rms == 0 || c.force_rms >= 3e-5) && controls &&
                        p.requested_mode == GENERATIVEQC_PRECISION_FP64 && p.effective_bits == 64 &&
                        p.mixed_stage_fock_builds == 0 && (strict_work || fitted_hf) &&
                        (!c.gradient_present || c.gradient_fp64);
  return {sizeof(generativeqc_precision_admission_v1),
          GENERATIVEQC_ABI_VERSION,
          1,
          budget_fock,
          strict_warm,
          standard};
}

}  // namespace generativeqc::runtime

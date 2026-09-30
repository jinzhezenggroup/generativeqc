#ifndef GENERATIVEQC_API_PRECISION_HPP
#define GENERATIVEQC_API_PRECISION_HPP

#include <limits>

#include "api/error.hpp"
#include "scf/precision_work.hpp"
#include "scf/types.hpp"

namespace generativeqc::api {

/** Share descriptor validation and copy-out across single and batched queries. */
inline generativeqc_status copy_precision_provenance(const scf::PrecisionProvenance& source,
                                                     generativeqc_precision_provenance* out) {
  if (out == nullptr) return GENERATIVEQC_STATUS_SUCCESS;
  if (!valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;

  generativeqc_precision_provenance record{};
  record.struct_size = sizeof(record);
  record.abi_version = GENERATIVEQC_ABI_VERSION;
  record.policy_version = source.policy_version;
  record.requested_mode = source.requested_mode;
  record.effective_bits = source.effective_bits;
  record.mixed_precision_fock_threshold = source.mixed_precision_fock_threshold;
  record.strict_refinement_applied = source.strict_refinement_applied ? 1 : 0;
  record.mixed_precision_reserved_error = source.mixed_precision_reserved_error;
  record.refinement_iterations = static_cast<int32_t>(source.refinement_iterations);
  record.mixed_stage_fock_builds = source.mixed_stage_fock_builds;
  record.strict_stage_fock_builds = source.strict_stage_fock_builds;
  record.post_scf_fock_builds = source.post_scf_fock_builds;
  record.execution_retries = source.execution_retries;
  record.mixed_admission_census = source.mixed_admission_census;
  record.final_residual_audits = source.final_residual_audits;
  record.skipped_final_fock_builds = source.skipped_final_fock_builds;
  record.operator_work_counters_valid = source.operator_work_counters_valid;
  *out = record;
  return GENERATIVEQC_STATUS_SUCCESS;
}

inline generativeqc_status copy_incremental_direct_jk_diagnostic(
    const scf::IncrementalDirectJkDiagnostic& source,
    generativeqc_incremental_direct_jk_diagnostic* out) {
  if (out == nullptr) return GENERATIVEQC_STATUS_SUCCESS;
  if (!valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  *out = {sizeof(*out),
          GENERATIVEQC_ABI_VERSION,
          source.policy_version,
          source.requested ? 1 : 0,
          source.active ? 1 : 0,
          source.quartet_work_counters_valid ? 1 : 0,
          source.anchor_full_builds,
          source.delta_builds,
          source.periodic_rebuilds,
          source.bypass_full_builds,
          source.post_scf_full_builds,
          source.anchor_updates,
          source.max_abs_delta_density,
          source.full_candidate_shell_quartets,
          source.full_rejected_shell_quartets,
          source.full_admitted_shell_quartets,
          source.full_admitted_quartet_tiles,
          source.delta_candidate_shell_quartets,
          source.delta_rejected_shell_quartets,
          source.delta_admitted_shell_quartets,
          source.delta_admitted_quartet_tiles};
  return GENERATIVEQC_STATUS_SUCCESS;
}

/** Validate every destination before copying so a rejected variable-length
 * query cannot leave a summary or prefix that looks authoritative. */
inline generativeqc_status copy_precision_work(const scf::PrecisionWork& source,
                                               std::uint32_t detail_version,
                                               generativeqc_precision_work_detail* out,
                                               generativeqc_precision_work_event* events,
                                               std::uint32_t event_capacity,
                                               generativeqc_precision_operator_record* operators,
                                               std::uint32_t operator_capacity) {
  if (detail_version != GENERATIVEQC_PRECISION_WORK_DETAIL_VERSION ||
      source.detail_version != detail_version) {
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (out != nullptr && !valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if ((events == nullptr && event_capacity != 0) ||
      (operators == nullptr && operator_capacity != 0)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  if (source.events.size() > std::numeric_limits<std::uint32_t>::max() ||
      source.operators.size() > std::numeric_limits<std::uint32_t>::max()) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
  if (events != nullptr) {
    if (event_capacity < source.events.size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.events.size(); ++index) {
      if (!valid_descriptor(events + index)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
    }
  }
  if (operators != nullptr) {
    if (operator_capacity < source.operators.size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.operators.size(); ++index) {
      if (!valid_descriptor(operators + index)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
    }
  }

  if (out != nullptr) {
    *out = {sizeof(*out),
            GENERATIVEQC_ABI_VERSION,
            source.detail_version,
            source.complete ? 1 : 0,
            source.operator_inventory_complete ? 1 : 0,
            static_cast<std::uint32_t>(source.events.size()),
            static_cast<std::uint32_t>(source.operators.size()),
            source.conversion_count,
            source.fallback_count,
            source.owner_id,
            source.returned_solve_epoch,
            source.returned_state_generation};
  }
  if (events != nullptr) {
    for (std::size_t index = 0; index < source.events.size(); ++index) {
      const auto& event = source.events[index];
      events[index] = {sizeof(events[index]),
                       GENERATIVEQC_ABI_VERSION,
                       static_cast<generativeqc_precision_work_event_kind>(event.kind),
                       static_cast<generativeqc_precision_work_phase>(event.phase),
                       event.sequence,
                       event.iteration,
                       event.owner_id,
                       event.solve_epoch,
                       event.state_generation};
    }
  }
  if (operators != nullptr) {
    for (std::size_t index = 0; index < source.operators.size(); ++index) {
      const auto& value = source.operators[index];
      operators[index] = {
          sizeof(operators[index]),
          GENERATIVEQC_ABI_VERSION,
          static_cast<generativeqc_precision_operator_kind>(value.kind),
          static_cast<generativeqc_precision_dtype>(value.storage),
          static_cast<generativeqc_precision_dtype>(value.compute),
          static_cast<generativeqc_precision_dtype>(value.accumulation),
          static_cast<generativeqc_precision_dtype>(value.reduction),
          static_cast<generativeqc_precision_arithmetic_mode>(value.arithmetic_mode),
          value.count};
    }
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::api

#endif

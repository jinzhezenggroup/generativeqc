#ifndef VIBEQC_API_PRECISION_HPP
#define VIBEQC_API_PRECISION_HPP

#include <limits>

#include "api/error.hpp"
#include "scf/precision_work.hpp"
#include "scf/types.hpp"

namespace vibeqc::api {

/** Share descriptor validation and copy-out across single and batched queries. */
inline vibeqc_status copy_precision_provenance(const scf::PrecisionProvenance& source,
                                               vibeqc_precision_provenance* out) {
  if (out == nullptr) return VIBEQC_STATUS_SUCCESS;
  if (!valid_descriptor(out)) return VIBEQC_STATUS_ABI_MISMATCH;

  vibeqc_precision_provenance record{};
  record.struct_size = sizeof(record);
  record.abi_version = VIBEQC_ABI_VERSION;
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
  return VIBEQC_STATUS_SUCCESS;
}

/** Validate every destination before copying so a rejected variable-length
 * query cannot leave a summary or prefix that looks authoritative. */
inline vibeqc_status copy_precision_work(const scf::PrecisionWork& source,
                                         std::uint32_t detail_version,
                                         vibeqc_precision_work_detail* out,
                                         vibeqc_precision_work_event* events,
                                         std::uint32_t event_capacity,
                                         vibeqc_precision_operator_record* operators,
                                         std::uint32_t operator_capacity) {
  if (detail_version != VIBEQC_PRECISION_WORK_DETAIL_VERSION ||
      source.detail_version != detail_version) {
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (out != nullptr && !valid_descriptor(out)) return VIBEQC_STATUS_ABI_MISMATCH;
  if ((events == nullptr && event_capacity != 0) ||
      (operators == nullptr && operator_capacity != 0)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (source.events.size() > std::numeric_limits<std::uint32_t>::max() ||
      source.operators.size() > std::numeric_limits<std::uint32_t>::max()) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
  if (events != nullptr) {
    if (event_capacity < source.events.size()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.events.size(); ++index) {
      if (!valid_descriptor(events + index)) return VIBEQC_STATUS_ABI_MISMATCH;
    }
  }
  if (operators != nullptr) {
    if (operator_capacity < source.operators.size()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.operators.size(); ++index) {
      if (!valid_descriptor(operators + index)) return VIBEQC_STATUS_ABI_MISMATCH;
    }
  }

  if (out != nullptr) {
    *out = {sizeof(*out),
            VIBEQC_ABI_VERSION,
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
                       VIBEQC_ABI_VERSION,
                       static_cast<vibeqc_precision_work_event_kind>(event.kind),
                       static_cast<vibeqc_precision_work_phase>(event.phase),
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
      operators[index] = {sizeof(operators[index]),
                          VIBEQC_ABI_VERSION,
                          static_cast<vibeqc_precision_operator_kind>(value.kind),
                          static_cast<vibeqc_precision_dtype>(value.storage),
                          static_cast<vibeqc_precision_dtype>(value.compute),
                          static_cast<vibeqc_precision_dtype>(value.accumulation),
                          static_cast<vibeqc_precision_dtype>(value.reduction),
                          static_cast<vibeqc_precision_arithmetic_mode>(value.arithmetic_mode),
                          value.count};
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::api

#endif

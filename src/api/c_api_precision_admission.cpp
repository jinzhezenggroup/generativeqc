#include <cmath>
#include <span>

#include "api/error.hpp"
#include "runtime/precision_admission.hpp"

extern "C" generativeqc_status generativeqc_precision_admission_query_v1(
    const generativeqc_precision_admission_context_v1* context,
    const generativeqc_precision_provenance* provenance,
    const generativeqc_precision_work_detail* detail,
    const generativeqc_precision_work_event* events,
    const generativeqc_precision_operator_record* operators,
    generativeqc_precision_admission_v1* output) {
  using generativeqc::api::valid_descriptor;
  if (!context || !provenance || !detail || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!valid_descriptor(context) || !valid_descriptor(provenance) || !valid_descriptor(detail) ||
      !valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (detail->detail_version != GENERATIVEQC_PRECISION_WORK_DETAIL_VERSION)
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if ((!events && detail->event_count) || (!operators && detail->operator_count) ||
      context->reserved || context->census_presence > 2 ||
      provenance->operator_work_counters_valid > 1)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (const auto flag :
       {context->ks_model, context->full_range_exchange, context->range_exchange,
        context->nonlocal_correlation, context->hartree_fock, context->density_fitted,
        context->warm_start_used, context->warm_start_fallback, context->gradient_present,
        context->gradient_fp64, context->gradient_qualified, context->converged,
        context->cuda_execution, context->fp64_candidate})
    if (flag > 1) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (const auto flag : {detail->complete, detail->operator_inventory_complete,
                          provenance->strict_refinement_applied})
    if (flag != 0 && flag != 1) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (const auto value : {context->energy_abs, context->force_max_abs, context->force_rms,
                           context->requested_controls[0], context->requested_controls[1],
                           context->requested_controls[2], context->standard_controls[0],
                           context->standard_controls[1], context->standard_controls[2]})
    if (!std::isfinite(value) || value < 0) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (std::uint32_t index = 0; index != detail->event_count; ++index)
    if (!valid_descriptor(events + index)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  for (std::uint32_t index = 0; index != detail->operator_count; ++index)
    if (!valid_descriptor(operators + index)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  *output = generativeqc::runtime::precision_admission(
      *context, *provenance, *detail, std::span(events, detail->event_count),
      std::span(operators, detail->operator_count));
  return GENERATIVEQC_STATUS_SUCCESS;
}

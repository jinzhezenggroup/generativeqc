#include <algorithm>
#include <span>
#include <string_view>

#include "api/error.hpp"
#include "methods/named_ks_catalog.hpp"
#include "runtime/precision_candidate.hpp"

extern "C" generativeqc_status generativeqc_precision_candidate_query_v1(
    const generativeqc_precision_candidate_context_v1* candidate,
    const generativeqc_precision_admission_context_v1* context,
    const generativeqc_precision_provenance* provenance,
    const generativeqc_precision_work_detail* detail,
    const generativeqc_precision_work_event* events,
    const generativeqc_precision_operator_record* operators,
    generativeqc_precision_candidate_v1* output) {
  if (!candidate || !context || !provenance || !detail || !output)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(candidate) ||
      !generativeqc::api::valid_descriptor(context) || !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  const auto copied = *candidate;
  if (copied.reserved || copied.gradient_owner > 2 || copied.gradient_returned_mode > 2 ||
      copied.correction_result_flags > 63 ||
      std::find(std::begin(copied.qualification), std::end(copied.qualification), '\0') ==
          std::end(copied.qualification) ||
      std::find(std::begin(copied.correction_model_identity),
                std::end(copied.correction_model_identity),
                '\0') == std::end(copied.correction_model_identity))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (const auto flag :
       {copied.budget_requested, copied.explicit_target, copied.budget_native, copied.budget_fock,
        copied.forces_requested, copied.gradient_present, copied.gradient_complete,
        copied.qualification_present, copied.gradient_capability_fallback,
        copied.weighted_task_counters_collected, copied.correction_present,
        copied.mixed_fock_count_present})
    if (flag > 1) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto model = *context;
  model.gradient_present = copied.gradient_present;
  model.gradient_fp64 = copied.gradient_returned_mode == 2;
  const auto* composite = generativeqc::methods::named_ks::find_method("r2scan-3c-rks");
  auto interpreted = generativeqc::runtime::candidate_gradient(
      copied, composite ? composite->identity : std::string_view{});
  model.gradient_qualified = interpreted.qualified_gradient;
  generativeqc_precision_admission_v1 admitted{sizeof(admitted), GENERATIVEQC_ABI_VERSION};
  const auto status = generativeqc_precision_admission_query_v1(&model, provenance, detail, events,
                                                                operators, &admitted);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  interpreted = generativeqc::runtime::precision_candidate(
      copied, model, *provenance, *detail, std::span(operators, detail->operator_count), admitted,
      interpreted);
  *output = interpreted;
  return GENERATIVEQC_STATUS_SUCCESS;
}

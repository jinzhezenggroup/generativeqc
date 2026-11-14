#include <cstring>
#include <limits>

#include "generativeqc/precision_admission.h"
#include "native_test_require.hpp"

int main() {
  generativeqc_precision_candidate_context_v1 candidate{sizeof(candidate),
                                                        GENERATIVEQC_ABI_VERSION};
  generativeqc_precision_admission_context_v1 context{sizeof(context), GENERATIVEQC_ABI_VERSION};
  generativeqc_precision_provenance work{sizeof(work), GENERATIVEQC_ABI_VERSION};
  generativeqc_precision_work_detail detail{sizeof(detail), GENERATIVEQC_ABI_VERSION};
  detail.detail_version = GENERATIVEQC_PRECISION_WORK_DETAIL_VERSION;
  generativeqc_precision_candidate_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION};
  const auto query = [&] {
    return generativeqc_precision_candidate_query_v1(&candidate, &context, &work, &detail, nullptr,
                                                     nullptr, &output);
  };
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && output.policy_version == 1 &&
               output.disposition == GENERATIVEQC_PRECISION_CANDIDATE_STRICT);
  candidate.budget_requested = candidate.budget_native = candidate.forces_requested = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS &&
               output.disposition == GENERATIVEQC_PRECISION_CANDIDATE_AUDIT &&
               output.reasons == GENERATIVEQC_PRECISION_CANDIDATE_BUDGET_AUDIT);
  candidate.gradient_present = candidate.gradient_complete = 1;
  candidate.gradient_schema_version = 2;
  candidate.qualification_present = 1;
  std::strcpy(candidate.qualification, "pbe0-rks-budget-fp32-v1");
  candidate.geometry_points = candidate.grid_points = 1;
  candidate.gradient_returned_mode = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && output.qualified_gradient &&
               output.disposition == GENERATIVEQC_PRECISION_CANDIDATE_MIXED_GRADIENT &&
               std::strcmp(output.gradient_admission, candidate.qualification) == 0);
  candidate.explicit_target = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && output.qualified_gradient &&
               output.disposition == GENERATIVEQC_PRECISION_CANDIDATE_AUDIT &&
               !(output.reasons & GENERATIVEQC_PRECISION_CANDIDATE_BUDGET_AUDIT));
  const auto before = output;
  candidate.reserved = 1;
  require_test(query() == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
               std::memcmp(&before, &output, sizeof(output)) == 0);
  candidate.reserved = 0;
  context.energy_abs = std::numeric_limits<double>::quiet_NaN();
  require_test(query() == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
               std::memcmp(&before, &output, sizeof(output)) == 0);
  context.energy_abs = 0;
  candidate.abi_version += 1;
  require_test(query() == GENERATIVEQC_STATUS_ABI_MISMATCH &&
               std::memcmp(&before, &output, sizeof(output)) == 0);
}

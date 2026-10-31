#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "generativeqc/precision_admission.h"
#include "native_test_require.hpp"

int main() {
  generativeqc_precision_admission_context_v1 context{};
  context.struct_size = sizeof(context);
  context.abi_version = GENERATIVEQC_ABI_VERSION;
  context.ks_model = context.census_presence = context.warm_start_used = 1;
  generativeqc_precision_provenance provenance{};
  provenance.struct_size = sizeof(provenance);
  provenance.abi_version = GENERATIVEQC_ABI_VERSION;
  provenance.effective_bits = 64;
  provenance.operator_work_counters_valid = 1;
  provenance.strict_stage_fock_builds = provenance.final_residual_audits = 1;
  generativeqc_precision_work_detail detail{};
  detail.struct_size = sizeof(detail);
  detail.abi_version = GENERATIVEQC_ABI_VERSION;
  detail.detail_version = GENERATIVEQC_PRECISION_WORK_DETAIL_VERSION;
  detail.complete = detail.operator_inventory_complete = 1;
  detail.owner_id = 11;
  detail.returned_solve_epoch = 3;
  detail.returned_state_generation = 8;
  std::array<generativeqc_precision_work_event, 2> events{};
  for (auto& event : events) {
    event.struct_size = sizeof(event);
    event.abi_version = GENERATIVEQC_ABI_VERSION;
    event.owner_id = detail.owner_id;
    event.solve_epoch = detail.returned_solve_epoch;
    event.state_generation = detail.returned_state_generation;
  }
  events[0].kind = GENERATIVEQC_PRECISION_EVENT_STRICT_FOCK;
  events[1].kind = GENERATIVEQC_PRECISION_EVENT_FINAL_AUDIT;
  std::array<generativeqc_precision_operator_record, 4> operators{};
  const std::array kinds{GENERATIVEQC_PRECISION_OPERATOR_COULOMB_J,
                         GENERATIVEQC_PRECISION_OPERATOR_XC,
                         GENERATIVEQC_PRECISION_OPERATOR_FOCK_ASSEMBLY,
                         GENERATIVEQC_PRECISION_OPERATOR_PHYSICAL_RESIDUAL};
  for (unsigned i = 0; i != operators.size(); ++i) {
    auto& row = operators[i];
    row.struct_size = sizeof(row);
    row.abi_version = GENERATIVEQC_ABI_VERSION;
    row.kind = kinds[i];
    row.storage_dtype = row.compute_dtype = row.accumulation_dtype = row.reduction_dtype =
        GENERATIVEQC_PRECISION_DTYPE_FP64;
    row.arithmetic_mode = GENERATIVEQC_PRECISION_ARITHMETIC_STRICT;
    row.count = 1;
  }
  detail.event_count = events.size();
  detail.operator_count = operators.size();
  generativeqc_precision_admission_v1 decision{sizeof(decision), GENERATIVEQC_ABI_VERSION};
  auto query = [&] {
    return generativeqc_precision_admission_query_v1(&context, &provenance, &detail, events.data(),
                                                     operators.data(), &decision);
  };
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && decision.strict_warm &&
               !decision.budget_fock);
  // A fresh audit on another density, owner, or execution cannot certify this
  // returned state, even when all aggregate counters remain positive.
  for (auto member : {&generativeqc_precision_work_event::owner_id,
                      &generativeqc_precision_work_event::solve_epoch,
                      &generativeqc_precision_work_event::state_generation}) {
    events[1].*member += 1;
    require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.strict_warm);
    events[1].*member -= 1;
  }
  context.range_exchange = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.strict_warm);
  context.range_exchange = 0;
  context.nonlocal_correlation = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.strict_warm);
  context.nonlocal_correlation = 0;
  provenance.mixed_stage_fock_builds = std::numeric_limits<std::uint64_t>::max();
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.strict_warm);
  provenance.mixed_stage_fock_builds = 0;
  context.gradient_present = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.strict_warm);
  context.gradient_qualified = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && decision.strict_warm);
  context.gradient_present = context.gradient_qualified = 0;

  // Pure-J mixed proposals may retain strict J; range/exact exchange needs its
  // own actually executed operator evidence rather than a method label.
  operators[1].compute_dtype = GENERATIVEQC_PRECISION_DTYPE_FP32;
  operators[1].arithmetic_mode = GENERATIVEQC_PRECISION_ARITHMETIC_MIXED;
  provenance.strict_stage_fock_builds = 2;
  provenance.strict_refinement_applied = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && decision.budget_fock &&
               !decision.strict_warm);
  context.range_exchange = 1;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.budget_fock);
  context.range_exchange = 0;

  context.ks_model = 0;
  context.hartree_fock = context.density_fitted = context.converged = context.cuda_execution =
      context.fp64_candidate = 1;
  context.census_presence = 2;
  provenance.operator_work_counters_valid = 0;
  provenance.requested_mode = GENERATIVEQC_PRECISION_FP64;
  context.energy_abs = 1e-5;
  context.force_max_abs = 1e-4;
  for (unsigned i = 0; i != 3; ++i)
    context.requested_controls[i] = context.standard_controls[i] = 1e-10;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && decision.standard_fallback);
  context.census_presence = 0;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.standard_fallback);
  context.census_presence = 2;
  context.force_rms = std::nextafter(3e-5, 0.0);
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.standard_fallback);
  context.force_rms = 3e-5;
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && decision.standard_fallback);
  context.requested_controls[0] = std::nextafter(context.standard_controls[0], 1.0);
  require_test(query() == GENERATIVEQC_STATUS_SUCCESS && !decision.standard_fallback);

  // All schema and input failures are transactional, including late array
  // descriptor validation after otherwise admissible metadata.
  const auto before = decision;
  context.reserved = 1;
  require_test(query() == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(std::memcmp(&before, &decision, sizeof(decision)) == 0);
  context.reserved = 0;
  context.force_rms = std::numeric_limits<double>::quiet_NaN();
  require_test(query() == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(std::memcmp(&before, &decision, sizeof(decision)) == 0);
  context.force_rms = 0;
  operators.back().abi_version += 1;
  require_test(query() == GENERATIVEQC_STATUS_ABI_MISMATCH);
  require_test(std::memcmp(&before, &decision, sizeof(decision)) == 0);
  return 0;
}

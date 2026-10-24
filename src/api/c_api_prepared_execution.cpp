#include <algorithm>
#include <iterator>

#include "api/handles.hpp"

namespace {
/** Private observation ABI. Counts describe host-control transactions only;
 * uninstrumented allocation, transfer, fence and launch counts are not zero. */
generativeqc_status lifecycle_record(const generativeqc::runtime::CompiledExecutionRegion& region,
                                     std::uint64_t* record, std::size_t count) {
  if (!record || count != 9) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto& binding = region.binding();
  const auto& metrics = region.metrics();
  const std::uint64_t values[] = {static_cast<std::uint64_t>(region.state()),
                                  static_cast<std::uint64_t>(binding.backend),
                                  static_cast<std::uint64_t>(binding.device),
                                  metrics.bindings,
                                  metrics.invalidations,
                                  metrics.submissions,
                                  metrics.executions,
                                  metrics.failures,
                                  metrics.recoveries};
  std::copy(std::begin(values), std::end(values), record);
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace

extern "C" {
generativeqc_status generativeqc_calculation_execution_lifecycle_v1(
    const generativeqc_calculation* calculation, std::uint64_t* record, std::size_t count) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  return lifecycle_record(calculation->plan->execution_lifecycle(), record, count);
}

generativeqc_status generativeqc_batch_execution_lifecycle_v1(const generativeqc_batch* batch,
                                                              std::uint64_t* record,
                                                              std::size_t count) {
  if (!batch) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  return lifecycle_record(batch->plan->execution_lifecycle(), record, count);
}
}

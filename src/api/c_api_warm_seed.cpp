#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/warm_seed.h"
#include "methods/warm_seed_capacity.hpp"

extern "C" generativeqc_status generativeqc_batch_query_warm_seed_capacity_v1(
    const generativeqc_batch* batch, generativeqc_warm_seed_capacity_v1* output) {
  if (!batch || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(output)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto* owner =
        dynamic_cast<const generativeqc::methods::BatchWarmStarts*>(batch->plan.get());
    if (!owner) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto capacity = generativeqc::methods::warm_seed_capacity(*owner, batch->atom_counts);
    *output = {sizeof(*output),           GENERATIVEQC_ABI_VERSION, capacity.density_bytes,
               capacity.coordinate_bytes, capacity.total_bytes,     capacity.largest_density_bytes};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

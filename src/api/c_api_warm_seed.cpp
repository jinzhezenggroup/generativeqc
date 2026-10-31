#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/warm_seed.h"
#include "methods/warm_seed_capacity.hpp"
#include "methods/warm_seed_snapshot.hpp"
#include "runtime/resource_ledger.hpp"

/** C handle owns only the detached method-level host snapshot. */
struct generativeqc_warm_seed_snapshot {
  generativeqc::methods::WarmSeedSnapshot value;
};

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

extern "C" generativeqc_status generativeqc_batch_capture_warm_seeds_v1(
    const generativeqc_batch* batch, uint64_t maximum_numeric_bytes,
    generativeqc_warm_seed_snapshot_info_v1* info, generativeqc_warm_seed_snapshot** output) {
  if (!batch || !info || !output || !(batch->flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(info)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto* owner =
        dynamic_cast<const generativeqc::methods::BatchWarmStarts*>(batch->plan.get());
    if (!owner) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    auto snapshot = std::make_unique<generativeqc_warm_seed_snapshot>();
    snapshot->value = generativeqc::methods::capture_warm_seeds(*owner, batch->atom_counts,
                                                                maximum_numeric_bytes);
    const auto& value = snapshot->value;
    *info = {sizeof(*info),
             GENERATIVEQC_ABI_VERSION,
             static_cast<uint32_t>(value.states.size()),
             0,
             value.numeric_bytes,
             value.largest_density_bytes};
    *output = snapshot.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_batch_restore_warm_seeds_v1(
    generativeqc_batch* batch, const generativeqc_warm_seed_snapshot* snapshot) {
  if (!batch || !snapshot || !(batch->flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    auto* owner = dynamic_cast<generativeqc::methods::BatchWarmStarts*>(batch->plan.get());
    if (!owner) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    // Physical restoration may allocate eigensolver/provider scratch. The SDK
    // must retain the target's admission even without a Python observation.
    generativeqc::runtime::DeviceResourceScope resources(batch->resource_ledger);
    generativeqc::methods::restore_warm_seeds(*owner, batch->atom_counts, snapshot->value);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_warm_seed_snapshot_read_v1(
    const generativeqc_warm_seed_snapshot* snapshot, uint32_t index,
    generativeqc_warm_seed_view_v1* view) {
  if (!snapshot || !view || index >= snapshot->value.states.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(view)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  generativeqc_warm_seed_view_v1 result{sizeof(result), GENERATIVEQC_ABI_VERSION};
  if (const auto& state = snapshot->value.states[index]) {
    result.density = state->density.data();
    result.density_count = state->density.size();
    result.coordinates = state->coordinates.data();
    result.coordinate_count = state->coordinates.size();
    result.energy = state->energy;
    result.energy_change = state->energy_change;
    result.density_rms = state->density_rms;
    result.iterations = state->iterations;
    result.present = 1;
  }
  *view = result;
  return GENERATIVEQC_STATUS_SUCCESS;
}

extern "C" void generativeqc_warm_seed_snapshot_destroy(generativeqc_warm_seed_snapshot* snapshot) {
  delete snapshot;
}

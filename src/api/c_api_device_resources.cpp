#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "api/device_resources.hpp"
#include "api/error.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/prepared_device_resources.hpp"

extern "C" {

generativeqc_status generativeqc_device_resources_create(
    const generativeqc_device_resource_descriptor* descriptor,
    generativeqc_device_resources** output) {
  using namespace generativeqc::runtime;
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(descriptor)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    if (!descriptor->buffers || !descriptor->buffer_count ||
        ranges_overlap(descriptor, sizeof(*descriptor), output, sizeof(*output)) ||
        ranges_overlap(descriptor->buffers,
                       size_mul(descriptor->buffer_count, sizeof(*descriptor->buffers)), output,
                       sizeof(*output)) ||
        *output)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    std::vector<resources::DeviceLedgerBuffer> capacities;
    capacities.reserve(descriptor->buffer_count);
    for (std::uint32_t index = 0; index < descriptor->buffer_count; ++index) {
      const auto& row = descriptor->buffers[index];
      if ((row.space != resources::kPinnedSpace && row.space != resources::kPageableSpace &&
           row.space != resources::kFirstDeviceSpace) ||
          row.runtime_allowance > 1)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      capacities.push_back({row.bytes, static_cast<std::size_t>(row.space), row.device,
                            row.first_phase, row.last_phase, bool(row.runtime_allowance)});
    }
    auto owner = resources::prepare_device_resources(capacities);
    if (!owner) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    auto handle = std::make_unique<generativeqc::api::DeviceResourceHandle>(std::move(owner));
    *output = generativeqc::api::device_resource_handle(handle.release());
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception(nullptr);
  }
}

void generativeqc_device_resources_destroy(generativeqc_device_resources* resources) {
  delete reinterpret_cast<generativeqc::api::DeviceResourceHandle*>(resources);
}

generativeqc_status generativeqc_device_resources_read(
    const generativeqc_device_resources* resources, generativeqc_device_resource_diagnostic* out) {
  if (!resources || !out) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  const auto& ledger = generativeqc::api::device_resource_owner(resources);
  std::lock_guard<std::mutex> lock(generativeqc::runtime::device_resource_mutex);
  generativeqc_device_resource_diagnostic snapshot{};
  snapshot.struct_size = out->struct_size;
  snapshot.abi_version = out->abi_version;
  snapshot.numeric_limit = ledger->limit;
  snapshot.declared_device_peak = ledger->declared_peak;
  snapshot.numeric_live_bytes = ledger->live;
  snapshot.numeric_peak_bytes = ledger->peak;
  snapshot.allocations = ledger->allocations;
  snapshot.rejected_allocations = ledger->rejected;
  snapshot.requested_bytes_available = !ledger->requested_bytes_overflow;
  snapshot.requested_bytes = snapshot.requested_bytes_available ? ledger->requested_bytes : 0;
  snapshot.provider_reserved_bytes = ledger->provider_reserved;
  snapshot.provider_reservation_peak_bytes = ledger->provider_reservation_peak;
  snapshot.combined_peak_bytes = ledger->admission_peak;
  snapshot.sampled_host_peak_bytes = ledger->native_host_peak;
  snapshot.host_samples = ledger->native_host_samples;
  snapshot.sampled_cuda_arena_peak_bytes = ledger->native_cuda_arena_peak;
  snapshot.cuda_arena_samples = ledger->native_cuda_arena_samples;
  snapshot.device = ledger->device;
  *out = snapshot;
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // extern "C"

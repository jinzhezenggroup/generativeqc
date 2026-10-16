#include <limits>
#include <memory>
#include <stdexcept>

#include "api/error.hpp"
#include "generativeqc/generativeqc.h"
#include "response/recycle.hpp"

struct generativeqc_response_recycle {
  generativeqc::response::ResponseRecycleSpace data;
};

namespace {

template <class Descriptor>
bool abi_valid(const Descriptor& descriptor) {
  return descriptor.struct_size >= sizeof(descriptor) &&
         descriptor.abi_version == GENERATIVEQC_ABI_VERSION;
}

std::size_t native_count(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("recycle count exceeds native size_t");
  return static_cast<std::size_t>(value);
}

generativeqc::response::ResponseColumns columns_for(
    const generativeqc_response_columns_v1& descriptor) {
  const auto count = native_count(descriptor.element_count);
  if (count && !descriptor.storage) throw std::invalid_argument("recycle storage is NULL");
  return {std::span(descriptor.storage, count),
          native_count(descriptor.rows),
          native_count(descriptor.columns),
          native_count(descriptor.origin),
          descriptor.row_stride,
          descriptor.column_stride};
}

template <class Action>
generativeqc_status guarded(Action action) {
  try {
    action();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const generativeqc::response::RecycleWorkspaceLimit&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::overflow_error&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

}  // namespace

extern "C" {

generativeqc_status generativeqc_response_recycle_capacity_v1_get(
    const generativeqc_response_recycle_descriptor_v1* descriptor,
    generativeqc_response_recycle_capacity_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!abi_valid(*descriptor) || !abi_valid(*output)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  return guarded([&] {
    const auto capacity = generativeqc::response::response_recycle_capacity(
        native_count(descriptor->dimension), native_count(descriptor->max_vectors),
        native_count(descriptor->max_retained_bytes));
    *output = {sizeof(*output),
               GENERATIVEQC_ABI_VERSION,
               capacity.dimension,
               capacity.vector_capacity,
               capacity.retained_bytes,
               capacity.projection_workspace_bytes,
               capacity.update_workspace_bytes};
  });
}

generativeqc_status generativeqc_response_recycle_create_v1(
    const generativeqc_response_recycle_descriptor_v1* descriptor,
    generativeqc_response_recycle** output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *output = nullptr;
  if (!abi_valid(*descriptor)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  return guarded([&] {
    auto owner = std::make_unique<generativeqc_response_recycle>(
        generativeqc_response_recycle{generativeqc::response::ResponseRecycleSpace(
            native_count(descriptor->dimension), native_count(descriptor->max_vectors),
            native_count(descriptor->max_retained_bytes))});
    *output = owner.release();
  });
}

generativeqc_status generativeqc_response_recycle_state_v1_get(
    const generativeqc_response_recycle* owner, generativeqc_response_recycle_state_v1* output) {
  if (!owner || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!abi_valid(*output)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  return guarded([&] {
    const auto measured = owner->data.allocations();
    *output = {sizeof(*output),
               GENERATIVEQC_ABI_VERSION,
               owner->data.capacity().dimension,
               owner->data.vector_count(),
               owner->data.generation(),
               owner->data.storage_bytes(),
               measured.live_bytes,
               measured.peak_bytes,
               measured.allocation_count};
  });
}

generativeqc_status generativeqc_response_recycle_project_v1(
    const generativeqc_response_recycle* owner, std::uint64_t expected_generation,
    const generativeqc_response_columns_v1* rhs, double* output, std::uint64_t output_count,
    std::uint64_t max_workspace_bytes) {
  if (!owner || !rhs || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!abi_valid(*rhs)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  return guarded([&] {
    if (output_count != owner->data.capacity().dimension)
      throw std::invalid_argument("recycle projection output size mismatch");
    const auto projected = owner->data.project(expected_generation, columns_for(*rhs),
                                               native_count(max_workspace_bytes));
    std::copy(projected.begin(), projected.end(), output);
  });
}

generativeqc_status generativeqc_response_recycle_update_v1(
    generativeqc_response_recycle* owner, std::uint64_t expected_generation,
    const generativeqc_response_columns_v1* solution, const generativeqc_response_columns_v1* basis,
    std::uint64_t max_vectors, std::uint64_t max_retained_bytes,
    std::uint64_t max_workspace_bytes) {
  if (!owner || !solution || !basis) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!abi_valid(*solution) || !abi_valid(*basis)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  return guarded([&] {
    owner->data.update(expected_generation, columns_for(*solution), columns_for(*basis),
                       native_count(max_vectors), native_count(max_retained_bytes),
                       native_count(max_workspace_bytes));
  });
}

generativeqc_status generativeqc_response_recycle_copy_v1(
    const generativeqc_response_recycle* owner, std::uint64_t expected_generation,
    std::uint64_t first_column, std::uint64_t column_count, double* output,
    std::uint64_t output_count) {
  if (!owner || (!output && output_count)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return guarded([&] {
    owner->data.copy(expected_generation, native_count(first_column), native_count(column_count),
                     std::span(output, native_count(output_count)));
  });
}

void generativeqc_response_recycle_destroy(generativeqc_response_recycle* owner) { delete owner; }

}  // extern "C"

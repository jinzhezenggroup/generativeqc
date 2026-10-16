#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "integrals/s_integrals.hpp"
#include "scf/occupied_projection.hpp"

namespace {
using generativeqc::scf::projection::Input;
using generativeqc::scf::projection::Policy;

bool metadata(const generativeqc_basis_projection_descriptor_v1& descriptor, Input& input,
              Policy& policy) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  if (descriptor.input_kind > 1 || descriptor.source_ao > maximum ||
      descriptor.target_ao > maximum || descriptor.occupied > maximum ||
      descriptor.maximum_ao > maximum || descriptor.maximum_workspace_bytes > maximum)
    return false;
  input = {static_cast<std::size_t>(descriptor.source_ao),
           static_cast<std::size_t>(descriptor.target_ao),
           static_cast<std::size_t>(descriptor.occupied),
           descriptor.occupation,
           {},
           {},
           {},
           {}};
  policy = {descriptor.relative_threshold, descriptor.validation_tolerance,
            descriptor.maximum_residual, static_cast<std::size_t>(descriptor.maximum_ao),
            static_cast<std::size_t>(descriptor.maximum_workspace_bytes)};
  return true;
}

struct Range {
  std::uintptr_t begin{}, end{};
};

bool range(const void* pointer, std::size_t count, std::size_t element_size, std::size_t alignment,
           Range& output) {
  if (!count) {
    output = {};
    return true;
  }
  if (!pointer || count > std::numeric_limits<std::size_t>::max() / element_size) return false;
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const auto bytes = count * element_size;
  if (address % alignment || address > std::numeric_limits<std::uintptr_t>::max() - bytes)
    return false;
  output = {address, address + bytes};
  return true;
}

bool disjoint(Range left, Range right) {
  return left.end <= right.begin || right.end <= left.begin;
}
}  // namespace

extern "C" generativeqc_status generativeqc_basis_projection_query_v1(
    const generativeqc_basis_projection_descriptor_v1* descriptor,
    generativeqc_basis_projection_capacity_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(descriptor) ||
      !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  Range input_record, output_record;
  if (!range(descriptor, 1, descriptor->struct_size, alignof(decltype(*descriptor)),
             input_record) ||
      !range(output, 1, output->struct_size, alignof(decltype(*output)), output_record) ||
      !disjoint(input_record, output_record))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    Input input;
    Policy policy;
    if (!metadata(*descriptor, input, policy)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto required = generativeqc::scf::projection::capacity(input, policy);
    *output = {sizeof(*output), GENERATIVEQC_ABI_VERSION, required.coefficient_count,
               required.density_count, required.numeric_workspace_bytes};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception();
  }
}

extern "C" generativeqc_status generativeqc_basis_project_cpu_v1(
    generativeqc_context* context, const generativeqc_basis_projection_descriptor_v1* descriptor,
    generativeqc_basis_projection_output_v1* output) {
  if (!context || !descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(descriptor) ||
      !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    Input input;
    Policy policy;
    if (!metadata(*descriptor, input, policy)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto required = generativeqc::scf::projection::capacity(input, policy);
    if (policy.maximum_workspace_bytes &&
        required.numeric_workspace_bytes > policy.maximum_workspace_bytes)
      throw generativeqc::scf::projection::BudgetExceeded{};
    const auto source_count = input.source_ao * input.source_ao;
    const auto target_count = input.target_ao * input.target_ao;
    const auto cross_count = input.target_ao * input.source_ao;
    const auto state_count =
        descriptor->input_kind ? source_count : input.source_ao * input.occupied;
    constexpr auto maximum_count = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (descriptor->source_overlap_count != source_count ||
        descriptor->target_overlap_count != target_count ||
        descriptor->cross_overlap_count != cross_count ||
        descriptor->source_state_count != state_count ||
        output->coefficient_capacity < required.coefficient_count ||
        output->density_capacity < required.density_count ||
        output->coefficient_capacity > maximum_count || output->density_capacity > maximum_count)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    Range inputs[6], coefficients, density;
    if (!range(descriptor->source_overlap, source_count, sizeof(double), alignof(double),
               inputs[0]) ||
        !range(descriptor->target_overlap, target_count, sizeof(double), alignof(double),
               inputs[1]) ||
        !range(descriptor->cross_overlap, cross_count, sizeof(double), alignof(double),
               inputs[2]) ||
        !range(descriptor->source_state, state_count, sizeof(double), alignof(double), inputs[3]) ||
        !range(descriptor, 1, descriptor->struct_size, alignof(decltype(*descriptor)), inputs[4]) ||
        !range(output, 1, output->struct_size, alignof(decltype(*output)), inputs[5]) ||
        !range(output->coefficients, required.coefficient_count, sizeof(double), alignof(double),
               coefficients) ||
        !range(output->density, required.density_count, sizeof(double), alignof(double), density) ||
        !disjoint(coefficients, density))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < 5; ++index)
      if (!disjoint(inputs[5], inputs[index])) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (const auto borrowed : inputs)
      if (!disjoint(coefficients, borrowed) || !disjoint(density, borrowed))
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    input.source_overlap = {descriptor->source_overlap, source_count};
    input.target_overlap = {descriptor->target_overlap, target_count};
    input.cross_overlap = {descriptor->cross_overlap, cross_count};
    input.source_state = {descriptor->source_state, state_count};
    const auto result = descriptor->input_kind
                            ? generativeqc::scf::projection::project_density(input, policy)
                            : generativeqc::scf::projection::project_occupied(input, policy);
    auto published = *output;
    const auto& diagnostic = result.diagnostics;
    published.source_metric_rank = diagnostic.source_metric_rank;
    published.target_metric_rank = diagnostic.target_metric_rank;
    published.occupied_rank = diagnostic.occupied_rank;
    published.minimum_projected_norm_squared = diagnostic.minimum_projected_norm_squared;
    published.projection_residual = diagnostic.projection_residual;
    published.normal_equation_residual = diagnostic.normal_equation_residual;
    published.metric_orthogonality_error = diagnostic.metric_orthogonality_error;
    published.electron_trace = diagnostic.electron_trace;
    published.metric_idempotency_error = diagnostic.metric_idempotency_error;
    if (!result.coefficients.empty())
      std::copy(result.coefficients.begin(), result.coefficients.end(), output->coefficients);
    std::copy(result.density.begin(), result.density.end(), output->density);
    *output = published;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_system_cross_overlap_cpu(
    generativeqc_context* context, const generativeqc_system* target,
    const generativeqc_system* source, double* output, size_t output_count) {
  if (context == nullptr || target == nullptr || source == nullptr || output == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    generativeqc::integrals::cross_overlap(target->data, source->data, {output, output_count});
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

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
#include "scf/system_projection.hpp"

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

namespace {
namespace system_projection = generativeqc::scf::system_projection;

system_projection::Controls system_controls(
    const generativeqc_hf_system_projection_controls_v1& controls) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  if (controls.unrestricted > 1 || controls.reserved || controls.maximum_ao > maximum ||
      controls.maximum_host_bytes > maximum)
    throw std::invalid_argument("invalid system projection controls");
  return {{controls.relative_threshold, controls.validation_tolerance, controls.maximum_residual,
           static_cast<std::size_t>(controls.maximum_ao), 0},
          bool(controls.unrestricted),
          static_cast<std::size_t>(controls.maximum_host_bytes)};
}

generativeqc_status system_projection_failure(std::string* detail = nullptr) {
  try {
    throw;
  } catch (const system_projection::Unsupported& error) {
    if (detail) *detail = error.what();
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  } catch (const std::overflow_error& error) {
    if (detail) *detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return generativeqc::api::map_exception(detail);
  }
}
}  // namespace

extern "C" generativeqc_status generativeqc_hf_system_projection_query_v1(
    const generativeqc_system* target, const generativeqc_system* source,
    const generativeqc_hf_system_projection_controls_v1* controls,
    generativeqc_hf_system_projection_capacity_v1* output) {
  if (!target || !source || !controls || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(controls) ||
      !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  Range borrowed, publication;
  if (!range(controls, 1, controls->struct_size, alignof(decltype(*controls)), borrowed) ||
      !range(output, 1, output->struct_size, alignof(decltype(*output)), publication) ||
      !disjoint(borrowed, publication))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    const auto required =
        system_projection::capacity(target->data, source->data, system_controls(*controls));
    *output = {sizeof(*output),
               GENERATIVEQC_ABI_VERSION,
               required.source_ao,
               required.target_ao,
               required.spins,
               required.occupied[0],
               required.occupied[1],
               required.source_density_count,
               required.target_density_count,
               required.numeric_workspace_bytes};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return system_projection_failure();
  }
}

extern "C" generativeqc_status generativeqc_hf_system_project_cpu_v1(
    generativeqc_context* context, const generativeqc_system* target,
    const generativeqc_system* source,
    const generativeqc_hf_system_projection_controls_v1* controls, const double* density,
    std::uint64_t density_count, generativeqc_basis_projection_output_v1* outputs,
    std::uint64_t output_count) {
  if (!context || !target || !source || !controls || !density || !outputs)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(controls)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  context->last_detail.clear();
  try {
    if (context->state.executed_backend != generativeqc::core::kCpuReferenceBackend)
      throw system_projection::Unsupported("system projection requires an explicit CPU context");
    const auto selected = system_controls(*controls);
    const auto required = system_projection::capacity(target->data, source->data, selected);
    if (density_count != required.source_density_count || output_count != required.spins)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    Range borrowed[4]{}, written[4]{};
    if (!range(controls, 1, controls->struct_size, alignof(decltype(*controls)), borrowed[0]) ||
        !range(density, density_count, sizeof(double), alignof(double), borrowed[1]) ||
        !range(outputs, output_count, sizeof(*outputs), alignof(decltype(*outputs)), borrowed[2]) ||
        !disjoint(borrowed[2], borrowed[0]) || !disjoint(borrowed[2], borrowed[1]))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t spin = 0; spin < required.spins; ++spin) {
      const auto& output = outputs[spin];
      if (!generativeqc::api::valid_descriptor(&output)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
      if (!range(&output, 1, output.struct_size, alignof(decltype(output)), borrowed[2 + spin]))
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      for (std::size_t prior = 0; prior < 2 + spin; ++prior)
        if (!disjoint(borrowed[2 + spin], borrowed[prior]))
          return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      const auto coefficients = required.target_ao * required.occupied[spin];
      const auto target_count = required.target_ao * required.target_ao;
      constexpr auto maximum_count = std::numeric_limits<std::size_t>::max() / sizeof(double);
      if (output.coefficient_capacity < coefficients || output.density_capacity < target_count ||
          output.coefficient_capacity > maximum_count || output.density_capacity > maximum_count ||
          !range(output.coefficients, coefficients, sizeof(double), alignof(double),
                 written[2 * spin]) ||
          !range(output.density, target_count, sizeof(double), alignof(double),
                 written[2 * spin + 1]))
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < 2 * required.spins; ++index) {
      for (auto input : borrowed)
        if (!disjoint(written[index], input)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      for (std::size_t prior = 0; prior < index; ++prior)
        if (!disjoint(written[index], written[prior])) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto result = system_projection::evaluate(
        target->data, source->data, selected, {density, static_cast<std::size_t>(density_count)});
    // Stage descriptors before copying any output. evaluate() has already
    // accepted every spin; the remaining trivial copies cannot fail halfway.
    std::array<generativeqc_basis_projection_output_v1, 2> published;
    for (std::size_t spin = 0; spin < required.spins; ++spin) {
      auto& output = published[spin];
      output = outputs[spin];
      const auto& diagnostic = result[spin].diagnostics;
      output.source_metric_rank = diagnostic.source_metric_rank;
      output.target_metric_rank = diagnostic.target_metric_rank;
      output.occupied_rank = diagnostic.occupied_rank;
      output.minimum_projected_norm_squared = diagnostic.minimum_projected_norm_squared;
      output.projection_residual = diagnostic.projection_residual;
      output.normal_equation_residual = diagnostic.normal_equation_residual;
      output.metric_orthogonality_error = diagnostic.metric_orthogonality_error;
      output.electron_trace = diagnostic.electron_trace;
      output.metric_idempotency_error = diagnostic.metric_idempotency_error;
    }
    for (std::size_t spin = 0; spin < required.spins; ++spin) {
      const auto& value = result[spin];
      if (!value.coefficients.empty())
        std::copy(value.coefficients.begin(), value.coefficients.end(), outputs[spin].coefficients);
      std::copy(value.density.begin(), value.density.end(), outputs[spin].density);
      outputs[spin] = published[spin];
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return system_projection_failure(&context->last_detail);
  }
}

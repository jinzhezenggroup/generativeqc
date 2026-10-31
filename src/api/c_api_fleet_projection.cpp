#include <algorithm>
#include <limits>
#include <mutex>
#include <span>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/generativeqc.h"
#include "methods/batch_capabilities.hpp"
#include "scf/fleet.hpp"
#include "scf/system_projection.hpp"

namespace {
namespace transfer = generativeqc::scf::fleet_projection;
using Facet = generativeqc::methods::BatchHfProjection;

/** Serialize opposing transfers without deadlock; a shared context is locked
 * only once. Prepared lifecycle checks also reject same-thread callback reentry. */
class LockedOwners {
 public:
  LockedOwners(const generativeqc_batch& target, const generativeqc_batch& source)
      : target_(target.context->mutex, std::defer_lock),
        source_(source.context->mutex, std::defer_lock) {
    if (target.context == source.context)
      target_.lock();
    else
      std::lock(target_, source_);
    for (const auto* owner : {target.plan.get(), source.plan.get()}) {
      const auto& state = owner->execution_lifecycle();
      if (state.running() || state.closed())
        throw std::invalid_argument("projection requires idle live prepared HF owners");
    }
  }

 private:
  std::unique_lock<std::recursive_mutex> target_, source_;
};

transfer::Controls controls(const generativeqc_hf_fleet_projection_controls_v1& input) {
  if (input.strict > 1 || input.reserved ||
      input.maximum_ao > std::numeric_limits<std::size_t>::max() ||
      input.maximum_host_bytes > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("invalid native fleet projection controls");
  return {{input.relative_threshold, input.validation_tolerance, input.maximum_residual,
           static_cast<std::size_t>(input.maximum_ao), 0},
          static_cast<std::size_t>(input.maximum_host_bytes),
          input.strict != 0};
}

transfer::Coordinates coordinates(const generativeqc_batch& target,
                                  const generativeqc_batch_input_descriptor* inputs,
                                  std::uint32_t count) {
  transfer::Coordinates result;
  if (!inputs && !count) return result;
  if (!inputs || count != target.plan->size())
    throw std::invalid_argument("projection coordinates require one descriptor per target item");
  result.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto& item = inputs[index];
    if (!generativeqc::api::valid_descriptor(&item))
      throw generativeqc::methods::MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH,
                                               "projection coordinate ABI mismatch");
    if (!item.coordinates && !item.coordinate_count) continue;
    // Never size an allocation or read an array from an untrusted payload count.
    // An invalid item remains a per-item rejection under non-strict policy.
    if (!item.coordinates || item.coordinate_count != std::size_t(target.atom_counts[index]) * 3)
      result[index] = std::span<const double>{};
    else
      result[index] = std::span<const double>{item.coordinates, item.coordinate_count};
  }
  return result;
}

const Facet& facet(const generativeqc_batch& batch) {
  const auto* result = dynamic_cast<const Facet*>(batch.plan.get());
  if (!result)
    throw generativeqc::scf::system_projection::Unsupported(
        "prepared owner has no HF projection facet");
  return *result;
}

void publish(const transfer::Item& item,
             generativeqc_hf_fleet_projection_item_v1& output) noexcept {
  output = {};
  output.struct_size = sizeof(output);
  output.abi_version = GENERATIVEQC_ABI_VERSION;
  output.accepted = item.accepted;
  output.spin_count = item.spins;
  output.numeric_workspace_bytes = item.numeric_workspace_bytes;
  output.source_energy = item.source_energy;
  output.source_iterations = item.source_iterations;
  std::copy_n(item.reason.data(), std::min(item.reason.size(), sizeof(output.reason) - 1),
              output.reason);
  for (std::size_t spin = 0; spin < item.spins; ++spin) {
    const auto& value = item.diagnostics[spin];
    output.spins[spin] = {value.source_metric_rank,
                          value.target_metric_rank,
                          value.occupied_rank,
                          value.minimum_projected_norm_squared,
                          value.projection_residual,
                          value.normal_equation_residual,
                          value.metric_orthogonality_error,
                          value.electron_trace,
                          value.metric_idempotency_error};
  }
}
}  // namespace

extern "C" generativeqc_status generativeqc_hf_fleet_projection_query_v1(
    const generativeqc_batch* target, const generativeqc_batch* source,
    const generativeqc_hf_fleet_projection_controls_v1* descriptor,
    const generativeqc_batch_input_descriptor* inputs, std::uint32_t input_count,
    generativeqc_hf_fleet_projection_capacity_v1* output) {
  if (!target || !source || !descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(descriptor) ||
      !generativeqc::api::valid_descriptor(output))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    LockedOwners locked(*target, *source);
    const auto policy = controls(*descriptor);
    const auto request = coordinates(*target, inputs, input_count);
    const auto required = facet(*target).projection_fleet().projection_capacity(
        facet(*source).projection_fleet(), request, policy);
    *output = {sizeof(*output), GENERATIVEQC_ABI_VERSION, required.numeric_peak_bytes,
               required.retained_bytes, required.largest_density_bytes};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const generativeqc::scf::system_projection::Unsupported& error) {
    std::lock_guard<std::recursive_mutex> lock(target->context->mutex);
    target->context->last_detail = error.what();
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  } catch (...) {
    std::lock_guard<std::recursive_mutex> lock(target->context->mutex);
    return generativeqc::api::map_exception(&target->context->last_detail);
  }
}

extern "C" generativeqc_status generativeqc_hf_fleet_initialize_cpu_v1(
    generativeqc_batch* target, const generativeqc_batch* source,
    const generativeqc_hf_fleet_projection_controls_v1* descriptor,
    const generativeqc_batch_input_descriptor* inputs, std::uint32_t input_count,
    generativeqc_hf_fleet_projection_item_v1* outputs, std::uint32_t output_count) {
  if (!target || !source || !descriptor || !outputs || output_count != target->plan->size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(descriptor)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  for (std::size_t index = 0; index < output_count; ++index)
    if (!generativeqc::api::valid_descriptor(&outputs[index]))
      return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    LockedOwners locked(*target, *source);
    const auto policy = controls(*descriptor);
    const auto request = coordinates(*target, inputs, input_count);
    (void)facet(*target);
    auto& owner = dynamic_cast<Facet&>(*target->plan).projection_fleet();
    // Allocate all presentation storage before the seed transaction. After
    // native commit, publication uses only non-throwing fixed-size copies.
    std::vector<generativeqc_hf_fleet_projection_item_v1> published(output_count);
    const auto result = owner.initialize_from(facet(*source).projection_fleet(), request, policy);
    for (std::size_t index = 0; index < output_count; ++index)
      publish(result.items[index], published[index]);
    std::copy(published.begin(), published.end(), outputs);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const generativeqc::scf::system_projection::Unsupported& error) {
    std::lock_guard<std::recursive_mutex> lock(target->context->mutex);
    target->context->last_detail = error.what();
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  } catch (...) {
    std::lock_guard<std::recursive_mutex> lock(target->context->mutex);
    return generativeqc::api::map_exception(&target->context->last_detail);
  }
}

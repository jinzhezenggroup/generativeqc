#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "api/device_resources.hpp"
#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/initial_guess_diagnostic.hpp"
#include "api/ks_diagnostic.hpp"
#include "api/ks_dispersion.hpp"
#include "api/named_ks_inputs.hpp"
#include "api/native_method_inputs.hpp"
#include "api/precision.hpp"
#include "api/prepared_publication.hpp"
#include "generativeqc/generativeqc.h"
#include "methods/batch_capabilities.hpp"
#include "methods/method.hpp"
#include "runtime/host_component_trace.hpp"

extern "C" {

generativeqc_status generativeqc_ks_batch_composite_correction_result_v1(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_ks_composite_correction_result_v1* out) {
  if (!batch || !out || index >= batch->dispersion.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (batch->plan->execution_lifecycle().state() !=
          generativeqc::runtime::CompiledExecutionState::Warm ||
      !batch->dispersion[index] || !batch->dispersion[index]->composite)
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  const auto& result = *batch->dispersion[index]->composite;
  const auto count = 3ull * batch->atom_counts[index];
  if ((!out->gradient && out->gradient_count) || (!out->gcp_gradient && out->gcp_gradient_count) ||
      (out->gradient && out->gradient_count != count) ||
      (out->gcp_gradient && out->gcp_gradient_count != count))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if ((out->gradient && result.gradient.size() != count) ||
      (out->gcp_gradient && result.gcp_gradient.size() != count))
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (out->gradient) std::copy(result.gradient.begin(), result.gradient.end(), out->gradient);
  if (out->gcp_gradient)
    std::copy(result.gcp_gradient.begin(), result.gcp_gradient.end(), out->gcp_gradient);
  out->energy = result.energy;
  out->gcp_energy = result.gcp_energy;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_ks_batch_dispersion_result_v1(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_d4_batch_item_result_descriptor* out) {
  if (!batch || !out || index >= batch->dispersion.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (batch->plan->execution_lifecycle().state() !=
          generativeqc::runtime::CompiledExecutionState::Warm ||
      !batch->dispersion[index])
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  const auto& result = *batch->dispersion[index];
  if ((!out->gradient && out->gradient_count) || (!out->charges && out->charge_count) ||
      (out->gradient && out->gradient_count != 3 * batch->atom_counts[index]) ||
      (out->charges && out->charge_count != batch->atom_counts[index]))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if ((out->gradient && result.gradient.size() != out->gradient_count) ||
      (out->charges && result.charges.size() != out->charge_count))
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (out->gradient) std::copy(result.gradient.begin(), result.gradient.end(), out->gradient);
  if (out->charges) std::copy(result.charges.begin(), result.charges.end(), out->charges);
  out->status = GENERATIVEQC_STATUS_SUCCESS;
  out->energy = result.two_body_energy + result.atm_energy;
  out->two_body_energy = result.two_body_energy;
  out->atm_energy = result.atm_energy;
  out->executed_backend = batch->context->state.executed_backend;
  return GENERATIVEQC_STATUS_SUCCESS;
}

const char* generativeqc_ks_batch_dispersion_variant_v1(const generativeqc_batch* batch,
                                                        uint32_t index) {
  if (!batch || index >= batch->dispersion.size()) return nullptr;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  return batch->plan->execution_lifecycle().state() ==
                     generativeqc::runtime::CompiledExecutionState::Warm &&
                 batch->dispersion[index]
             ? batch->dispersion[index]->variant
             : nullptr;
}

generativeqc_status generativeqc_ks_batch_d3_diagnostic_v1(
    const generativeqc_batch* batch, uint32_t index, generativeqc_d3_runtime_diagnostic* out) {
  if (!batch || !out) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto* diagnostic =
      dynamic_cast<const generativeqc::methods::BatchDispersionDiagnostics*>(batch->plan.get());
  return diagnostic ? diagnostic->dispersion_diagnostic(index, out, nullptr)
                    : GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status generativeqc_ks_batch_d4_diagnostic_v1(
    const generativeqc_batch* batch, uint32_t index, generativeqc_d4_runtime_diagnostic* out) {
  if (!batch || !out) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto* diagnostic =
      dynamic_cast<const generativeqc::methods::BatchDispersionDiagnostics*>(batch->plan.get());
  return diagnostic ? diagnostic->dispersion_diagnostic(index, nullptr, out)
                    : GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

/** Private v1 bridge: expose the *prepared item* capability rather than
 * copying method/backend/DF/spin qualification logic into Python. No mutation,
 * SCF, compiler or DFT source evaluation. Unqualified items return ENERGY
 * without a FORCES bit; non-KS methods reject this private query. */
generativeqc_status generativeqc_ks_batch_supported_properties_v1(
    const generativeqc_batch* batch, std::uint32_t index, generativeqc_property_flags* output) {
  if (!batch || !output || index >= batch->plan->size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto value = batch->plan->supported_properties(index);
  if (!value) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  *output = *value;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_batch_get_initial_guess_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_initial_guess_diagnostic* out) {
  if (!batch || index >= batch->plan->size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  return generativeqc::api::copy_initial_guess_diagnostic(batch->initial_guesses[index], out);
}

static generativeqc_status prepare_batch(
    generativeqc_context* context, const generativeqc_system* const* systems, uint32_t system_count,
    const generativeqc_method_descriptor* descriptor, generativeqc_batch_flags flags,
    const std::shared_ptr<generativeqc::runtime::DeviceResourceLedger>& resource_owner,
    generativeqc_batch** batch, const generativeqc_named_ks_descriptor* named) {
  generativeqc::runtime::host_trace::Region trace("batch_prepare");
  if (context == nullptr || systems == nullptr || system_count == 0 || descriptor == nullptr ||
      batch == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *batch = nullptr;
  if (!generativeqc::api::valid_descriptor(descriptor) ||
      (descriptor->initial_guess &&
       !generativeqc::api::valid_descriptor(descriptor->initial_guess))) {
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  }

  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    if (resource_owner) {
      if (context->state.executed_backend != generativeqc::core::kCudaBackend)
        return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
      if (context->state.device_id != resource_owner->device)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    generativeqc::runtime::DeviceResourceScope resources(resource_owner);
    std::optional<generativeqc::api::NamedKsInputs> named_inputs;
    if (named) {
      named_inputs.emplace(*descriptor, *named, context->state.executed_backend);
      descriptor = &named_inputs->method.base;
    }

    std::vector<generativeqc::core::System> native_systems;
    native_systems.reserve(system_count);
    std::vector<std::uint32_t> atom_counts;
    atom_counts.reserve(system_count);
    for (std::uint32_t i = 0; i < system_count; ++i) {
      if (systems[i] == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      if (named_inputs)
        generativeqc::methods::named_ks::validate_bound_basis(*named_inputs->program,
                                                              systems[i]->data);
      if (!generativeqc::api::valid_system_roles(*systems[i], *descriptor,
                                                 context->state.executed_backend)) {
        context->last_detail =
            "auxiliary handle role requires its matching method and density-fitting backend";
        return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
      }
      native_systems.push_back(systems[i]->data);
      atom_counts.push_back(static_cast<std::uint32_t>(systems[i]->data.atoms.size()));
    }
    auto candidate = std::make_unique<generativeqc_batch>();
    candidate->context = context;
    candidate->resource_ledger = resource_owner;
    candidate->flags = flags;
    candidate->atom_counts = std::move(atom_counts);
    candidate->last_fock_builds.resize(system_count);
    candidate->precision.resize(system_count);
    candidate->incremental_direct_jk.resize(system_count);
    candidate->precision_work.resize(system_count);
    candidate->scf_diagnostics.resize(system_count);
    candidate->ks_diagnostics.resize(system_count);
    candidate->initial_guesses.resize(system_count);
    candidate->dispersion.resize(system_count);
    candidate->plan = generativeqc::methods::prepare_batch(
        context->state, std::move(native_systems), *descriptor, flags,
        generativeqc::api::native_auxiliary_basis(*descriptor),
        named_inputs && named_inputs->program->dispersion.kind ? &named_inputs->program->dispersion
                                                               : nullptr,
        named ? named->dispersion_maximum_bytes : 0);
    *batch = candidate.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_batch_prepare(generativeqc_context* context,
                                               const generativeqc_system* const* systems,
                                               uint32_t system_count,
                                               const generativeqc_method_descriptor* descriptor,
                                               generativeqc_batch_flags flags,
                                               generativeqc_batch** batch) {
  return prepare_batch(context, systems, system_count, descriptor, flags, {}, batch, nullptr);
}

generativeqc_status generativeqc_batch_prepare_named_ks_v1(
    generativeqc_context* context, const generativeqc_system* const* systems, uint32_t system_count,
    const generativeqc_method_descriptor* descriptor, const generativeqc_named_ks_descriptor* name,
    generativeqc_batch_flags flags, generativeqc_batch** batch) {
  if (batch) *batch = nullptr;
  if (!name) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return prepare_batch(context, systems, system_count, descriptor, flags, {}, batch, name);
}

generativeqc_status generativeqc_batch_prepare_with_device_resources(
    generativeqc_context* context, const generativeqc_system* const* systems, uint32_t system_count,
    const generativeqc_method_descriptor* descriptor, generativeqc_batch_flags flags,
    const generativeqc_device_resources* resources, generativeqc_batch** batch) {
  if (!resources) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return prepare_batch(context, systems, system_count, descriptor, flags,
                       generativeqc::api::device_resource_owner(resources), batch, nullptr);
}

generativeqc_status generativeqc_batch_get_device_resources(
    const generativeqc_batch* batch, generativeqc_device_resources** resources) {
  if (!batch || !resources || *resources) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (!batch->resource_ledger) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  try {
    auto handle = std::make_unique<generativeqc::api::DeviceResourceHandle>(batch->resource_ledger);
    *resources = generativeqc::api::device_resource_handle(handle.release());
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

void generativeqc_batch_destroy(generativeqc_batch* batch) {
  generativeqc::runtime::host_trace::Region trace("batch_destroy");
  delete batch;
}

uint32_t generativeqc_batch_get_system_count(const generativeqc_batch* batch) {
  return batch == nullptr ? 0 : static_cast<std::uint32_t>(batch->plan->size());
}

generativeqc_status generativeqc_batch_get_scf_diagnostic(const generativeqc_batch* batch,
                                                          uint32_t index,
                                                          generativeqc_scf_diagnostic* out) {
  if (!batch || index >= batch->plan->size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (out && !generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (!batch->scf_diagnostics[index]) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (out) *out = *batch->scf_diagnostics[index];
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_batch_get_correlation_diagnostic(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_correlation_diagnostic* diagnostic) {
  if (!batch || !diagnostic || index >= batch->plan->size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(diagnostic)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchCorrelationDiagnostics*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto value = capability->correlation_diagnostic(index);
    if (!value) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    *diagnostic = *value;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_cc_performance_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_cc_performance_diagnostic* out) {
  if (!batch || index >= batch->plan->size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (out && !generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchCcPerformanceDiagnostics*>(
            batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto value = capability->cc_performance_diagnostic(index);
    if (!value) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (out)
      *out = {sizeof(*out),
              GENERATIVEQC_ABI_VERSION,
              value->reference_seconds,
              value->problem_seconds,
              value->provider_seconds,
              value->source_seconds,
              value->solver_seconds,
              value->iteration_seconds,
              value->replay_seconds,
              value->update_seconds,
              value->diis_seconds,
              value->triples_seconds,
              value->source_scans,
              value->source_reads,
              value->source_values,
              value->transform_fmas,
              value->transform_stages,
              value->mo_blocks,
              value->cuda_transform_calls,
              value->cuda_batch_calls,
              value->iteration_graph_calls,
              value->replay_graph_calls,
              value->update_calls,
              value->generated_error_checks,
              value->diis_gram_calls,
              value->diis_coefficient_calls,
              value->diis_combine_calls};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_ks_diagnostic(const generativeqc_batch* batch,
                                                         uint32_t index,
                                                         generativeqc_ks_diagnostic* out,
                                                         generativeqc_ks_iteration* history,
                                                         uint32_t history_capacity) {
  if (!batch || index >= batch->plan->size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  return generativeqc::api::copy_ks_diagnostic(batch->ks_diagnostics[index], out, history,
                                               history_capacity);
}

generativeqc_status generativeqc_batch_get_ks_coulomb_exchange_work_v1(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_ks_coulomb_exchange_work_v1* out) {
  if (!batch || index >= batch->ks_diagnostics.size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  return generativeqc::api::copy_ks_coulomb_exchange_work(batch->ks_diagnostics[index], out);
}

/** Copy the optional VV10 work receipt without another device read or fence. */
extern "C" GENERATIVEQC_API generativeqc_status generativeqc_internal_batch_get_nonlocal_work_v1(
    const generativeqc_batch* batch, uint32_t index, uint64_t* values, size_t count) {
  if (!batch || !values || count != 6 || index >= batch->ks_diagnostics.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto& diagnostic = batch->ks_diagnostics[index];
  if (!diagnostic || !diagnostic->nonlocal_work[0]) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  std::copy(diagnostic->nonlocal_work.begin(), diagnostic->nonlocal_work.end(), values);
  return GENERATIVEQC_STATUS_SUCCESS;
}

/** Preserve interleaved provisional/physical history without changing public ABI. */
extern "C" GENERATIVEQC_API generativeqc_status
generativeqc_internal_batch_get_nonlocal_history_scopes_v1(const generativeqc_batch* batch,
                                                           uint32_t index, uint64_t* values,
                                                           size_t count) {
  if (!batch || !values || index >= batch->ks_diagnostics.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto& diagnostic = batch->ks_diagnostics[index];
  if (!diagnostic || !diagnostic->nonlocal_work[0]) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (count != diagnostic->history.size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  for (size_t i = 0; i < count; ++i) values[i] = diagnostic->history[i].nonlocal_operator_scope;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_batch_get_ks_ao_selection_diagnostic_v1(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_ks_ao_selection_diagnostic_v1* out) {
  if (!batch || !out || index >= batch->ks_diagnostics.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (!batch->ks_diagnostics[index]) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  const auto& work = batch->ks_diagnostics[index]->cuda_ao_selection;
  // CPU/host-unfused owners have no device XC submission evidence. Preserve
  // unavailable rather than publishing their default-initialized zero counts.
  if (!work.xc_evaluations) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  *out = {sizeof(*out),
          GENERATIVEQC_ABI_VERSION,
          work.requested,
          work.selected,
          work.tiles,
          work.empty_tiles,
          work.min_active,
          work.max_active,
          work.active_sum,
          work.discovery_ao_jet_values,
          work.point_ao_visits,
          work.point_ao_square_sum,
          work.dense_point_ao_square_sum,
          work.discovery_d2h_bytes,
          work.reserved_device_bytes,
          work.host_peak_bytes,
          work.xc_evaluations,
          work.cutoff,
          work.discovery_seconds};
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_batch_get_ks_transport_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_ks_transport_diagnostic* out) {
  if (!batch || index >= batch->plan->size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (out && !generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchKsTransportDiagnostics*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto source = capability->ks_transport_diagnostic(index);
    if (!source) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (out)
      *out = {sizeof(*out),
              GENERATIVEQC_ABI_VERSION,
              source->setup_h2d_bytes,
              source->density_h2d_bytes,
              source->scalar_d2h_bytes,
              source->matrix_d2h_bytes,
              source->synchronizations,
              source->iterations,
              source->occupation_stabilized_proposals};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_last_shell_class_profile(
    const generativeqc_batch* batch, generativeqc_shell_class_profile_entry* entries,
    uint32_t entry_count) {
  if (batch == nullptr || entries == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchDirectQueueProfiles*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto profile = capability->last_direct_shell_class_profile();
    if (!profile.has_value()) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (entry_count < profile->size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < profile->size(); ++index) {
      const generativeqc::methods::DirectShellClassProfileEntry& source = (*profile)[index];
      entries[index] = {source.shell_quartets, source.tiles, source.ao_quartets,
                        source.primitive_quartets};
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_last_ppps_queue_profile(
    const generativeqc_batch* batch, generativeqc_ppps_queue_profile* profile) {
  if (batch == nullptr || profile == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchDirectQueueProfiles*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto source = capability->last_direct_ppps_queue_profile();
    if (!source.has_value()) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    *profile = {};
    profile->descriptor_slots = source->descriptor_slots;
    profile->non_empty_descriptors = source->non_empty_descriptors;
    profile->empty_descriptors = source->empty_descriptors;
    profile->tasks = source->tasks;
    profile->primitive_work = source->primitive_work;
    profile->ket_count_min = source->ket_count_min;
    profile->ket_count_median = source->ket_count_median;
    profile->ket_count_p90 = source->ket_count_p90;
    profile->ket_count_p99 = source->ket_count_p99;
    profile->ket_count_max = source->ket_count_max;
    std::copy(source->lane_efficiency.begin(), source->lane_efficiency.end(),
              profile->lane_efficiency);
    profile->primitive_warp_efficiency = source->primitive_warp_efficiency;
    std::copy(source->task_tail_imbalance.begin(), source->task_tail_imbalance.end(),
              profile->task_tail_imbalance);
    std::copy(source->primitive_tail_imbalance.begin(), source->primitive_tail_imbalance.end(),
              profile->primitive_tail_imbalance);
    std::copy(source->orientation_tasks.begin(), source->orientation_tasks.end(),
              profile->orientation_tasks);
    std::copy(source->orientation_primitive_work.begin(), source->orientation_primitive_work.end(),
              profile->orientation_primitive_work);
    std::copy(source->bra_primitive_tasks.begin(), source->bra_primitive_tasks.end(),
              profile->bra_primitive_tasks);
    std::copy(source->bra_primitive_work.begin(), source->bra_primitive_work.end(),
              profile->bra_primitive_work);
    std::copy(source->ket_primitive_tasks.begin(), source->ket_primitive_tasks.end(),
              profile->ket_primitive_tasks);
    std::copy(source->ket_primitive_work.begin(), source->ket_primitive_work.end(),
              profile->ket_primitive_work);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_last_eigensolver_diagnostics(
    const generativeqc_batch* batch, generativeqc_eigensolver_diagnostic* entries,
    uint32_t entry_count, uint32_t* written_count) {
  if (batch == nullptr || written_count == nullptr || (entries == nullptr && entry_count != 0U)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *written_count = 0U;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchEigensolverProfiles*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto source = capability->last_eigensolver_diagnostics();
    if (source.empty()) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (source.size() > std::numeric_limits<std::uint32_t>::max()) {
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    *written_count = static_cast<std::uint32_t>(source.size());
    if (entries == nullptr) return GENERATIVEQC_STATUS_SUCCESS;
    if (entry_count < source.size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.size(); ++index) {
      const generativeqc::methods::EigensolverDiagnostic& input = source[index];
      generativeqc_eigensolver_diagnostic& output = entries[index];
      output = {};
      output.bucket_id = input.bucket_id;
      output.ordinary_family = static_cast<std::int32_t>(input.ordinary_family);
      output.graph_family = static_cast<std::int32_t>(input.graph_family);
      output.selection_source = static_cast<std::int32_t>(input.selection_source);
      output.matrix_dimension = input.matrix_dimension;
      output.physical_system_count = input.physical_system_count;
      output.solver_batch_count = input.solver_batch_count;
      output.api_eligible = input.api_eligible ? 1 : 0;
      output.api_reason = static_cast<std::int32_t>(input.api_reason);
      output.matrix_batch_product = input.matrix_batch_product;
      output.probe_failure_stage = static_cast<std::int32_t>(input.probe_failure_stage);
      output.device_workspace_bytes = input.device_workspace_bytes;
      output.host_workspace_bytes = input.host_workspace_bytes;
      output.available_device_bytes = input.available_device_bytes;
      output.device_id = input.device_id;
      std::copy(input.device_uuid.begin(), input.device_uuid.end(), output.device_uuid);
      std::copy(input.device_name.begin(), input.device_name.end(), output.device_name);
      output.compute_capability_major = input.compute_capability_major;
      output.compute_capability_minor = input.compute_capability_minor;
      output.cuda_runtime_version = input.cuda_runtime_version;
      output.cuda_driver_version = input.cuda_driver_version;
      output.cusolver_version = input.cusolver_version;
      output.cuda_error = input.cuda_error;
      output.cusolver_error = input.cusolver_error;
      output.ordinary_execution_passed = input.ordinary_execution_passed ? 1 : 0;
      output.graph_capture_passed = input.graph_capture_passed ? 1 : 0;
      output.host_graph_replay_passed = input.host_graph_replay_passed ? 1 : 0;
      output.device_tail_replay_passed = input.device_tail_replay_passed ? 1 : 0;
      output.graph_eligible = input.graph_eligible ? 1 : 0;
      output.maximum_eigenvalue_error = input.maximum_eigenvalue_error;
      output.maximum_residual = input.maximum_residual;
      output.maximum_orthogonality_error = input.maximum_orthogonality_error;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_last_density_fitting_metric_diagnostics(
    const generativeqc_batch* batch, generativeqc_density_fitting_metric_diagnostic* entries,
    uint32_t entry_count, uint32_t* written_count) {
  if (batch == nullptr || written_count == nullptr || (entries == nullptr && entry_count != 0U)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *written_count = 0U;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchDfMetricDiagnostics*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto& source = capability->last_density_fitting_metric_diagnostics();
    if (source.empty()) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (source.size() > std::numeric_limits<uint32_t>::max()) {
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    *written_count = static_cast<uint32_t>(source.size());
    if (entries == nullptr) return GENERATIVEQC_STATUS_SUCCESS;
    if (entry_count < source.size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.size(); ++index) {
      const auto& input = source[index];
      auto& output = entries[index];
      output = {};
      output.bucket_id = static_cast<uint32_t>(input.bucket_id);
      output.system_index = static_cast<uint32_t>(input.system_index);
      output.effective_rank = input.effective_rank;
      output.absolute_threshold = input.absolute_threshold;
      output.condition_number = input.condition_number;
      output.solver_device_workspace_bytes = input.solver_device_workspace_bytes;
      output.solver_host_workspace_bytes = input.solver_host_workspace_bytes;
      output.device_resident_bytes = input.device_resident_bytes;
      output.peak_device_bytes = input.peak_device_bytes;
      output.host_resident_bytes = input.host_resident_bytes;
      output.peak_host_bytes = input.peak_host_bytes;
      output.auxiliary_tile = input.auxiliary_tile;
      output.streamed = input.streamed ? 1 : 0;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_last_inactive_eigensolver_profile(
    const generativeqc_batch* batch, generativeqc_inactive_eigensolver_profile_entry* entries,
    uint32_t entry_count, uint32_t* written_count) {
  if (batch == nullptr || written_count == nullptr || (entries == nullptr && entry_count != 0U)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *written_count = 0U;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchEigensolverProfiles*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto source = capability->last_inactive_eigensolver_profile();
    if (source.empty()) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (source.size() > std::numeric_limits<std::uint32_t>::max()) {
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    *written_count = static_cast<std::uint32_t>(source.size());
    if (entries == nullptr) return GENERATIVEQC_STATUS_SUCCESS;
    if (entry_count < source.size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.size(); ++index) {
      const generativeqc::methods::InactiveEigensolverProfileEntry& input = source[index];
      generativeqc_inactive_eigensolver_profile_entry& output = entries[index];
      output = {};
      output.bucket_id = input.bucket_id;
      output.iteration = input.iteration;
      output.family = static_cast<std::int32_t>(input.family);
      output.physical_system_count = input.physical_system_count;
      output.solver_batch_count = input.solver_batch_count;
      output.active_physical_count = input.active_physical_count;
      output.active_solver_count = input.active_solver_count;
      output.solver_elapsed_nanoseconds = input.solver_elapsed_nanoseconds;
      output.inactive_input_nonfinite_count = input.inactive_input_nonfinite_count;
      output.inactive_submission_nonfinite_count = input.inactive_submission_nonfinite_count;
      output.inactive_info_nonzero_count = input.inactive_info_nonzero_count;
      output.inactive_touch_flags = input.inactive_touch_flags;
      output.provider_invoked = input.provider_invoked ? 1 : 0;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_clear_warm_starts(generativeqc_batch* batch) {
  if (batch == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  try {
    batch->plan->reset_execution();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_set_warm_start_updates(generativeqc_batch* batch,
                                                              int32_t enabled) {
  if (batch == nullptr || (enabled != 0 && enabled != 1)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  try {
    auto* capability = dynamic_cast<generativeqc::methods::BatchWarmStarts*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    capability->set_warm_start_updates(enabled != 0);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_get_hf_warm_state(const generativeqc_batch* batch,
                                                         uint32_t index,
                                                         generativeqc_hf_warm_state* state) {
  if (!batch || !state || index >= batch->plan->size()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(state)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto* capability =
        dynamic_cast<const generativeqc::methods::BatchWarmStarts*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto& source = capability->warm_state(index);
    if (!source) {
      state->present = 0;
      state->density_count = state->coordinate_count = 0;
      return GENERATIVEQC_STATUS_SUCCESS;
    }
    const bool query = !state->density && !state->coordinates;
    if (!query &&
        (!state->density || !state->coordinates || state->density_count < source->density.size() ||
         state->coordinate_count < source->coordinates.size()))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    state->present = 1;
    state->density_count = source->density.size();
    state->coordinate_count = source->coordinates.size();
    state->energy = source->energy;
    state->energy_change = source->energy_change;
    state->density_rms = source->density_rms;
    state->iterations = source->iterations;
    if (!query) {
      std::copy(source->density.begin(), source->density.end(), state->density);
      std::copy(source->coordinates.begin(), source->coordinates.end(), state->coordinates);
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_restore_hf_warm_states(
    generativeqc_batch* batch, const generativeqc_hf_warm_state* states, uint32_t count) {
  if (!batch || !states || count != batch->plan->size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    auto* capability = dynamic_cast<generativeqc::methods::BatchWarmStarts*>(batch->plan.get());
    if (!capability) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    // Validate dimensions against the trusted prepared topology before any
    // caller-controlled allocation or pointer arithmetic.
    for (uint32_t i = 0; i < count; ++i) {
      const auto& state = states[i];
      if (!generativeqc::api::valid_descriptor(&state)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
      if (state.present != 0 && state.present != 1) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      if (!state.present) continue;
      if (!state.density || !state.coordinates ||
          state.density_count != capability->warm_density_size(i) ||
          state.coordinate_count != std::size_t(batch->atom_counts[i]) * 3)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    std::vector<std::optional<generativeqc::scf::HfWarmState>> candidates(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& state = states[i];
      if (!state.present) continue;
      candidates[i] = generativeqc::scf::HfWarmState{
          {state.density, state.density + state.density_count},
          {state.coordinates, state.coordinates + state.coordinate_count},
          state.energy,
          state.energy_change,
          state.density_rms,
          state.iterations};
    }
    capability->restore_warm_states(std::move(candidates));
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

static generativeqc_status execute_batch(generativeqc_batch* batch,
                                         const generativeqc_batch_input_descriptor* inputs,
                                         uint32_t input_count,
                                         generativeqc_batch_item_result_descriptor* results,
                                         uint32_t result_count, bool selective_forces) {
  generativeqc::runtime::host_trace::Region trace("batch_execute");
  if (batch == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  // Invalidation mutates shared diagnostics even when validation rejects the
  // replay, so it belongs to the same serialized operation as execution.
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  std::optional<generativeqc::runtime::DeviceResourceScope> resource_scope;
  generativeqc::api::PreparedPublication publication(*batch->plan);
  std::fill(batch->precision_work.begin(), batch->precision_work.end(), std::nullopt);
  std::fill(batch->initial_guesses.begin(), batch->initial_guesses.end(), std::nullopt);
  std::fill(batch->last_fock_builds.begin(), batch->last_fock_builds.end(), 0);
  // Revoke host records before method invalidation or argument rejection can
  // return. A failed replay must not expose diagnostics from the previous run.
  std::fill(batch->precision.begin(), batch->precision.end(), std::nullopt);
  std::fill(batch->incremental_direct_jk.begin(), batch->incremental_direct_jk.end(), std::nullopt);
  std::fill(batch->scf_diagnostics.begin(), batch->scf_diagnostics.end(), std::nullopt);
  std::fill(batch->ks_diagnostics.begin(), batch->ks_diagnostics.end(), std::nullopt);
  std::fill(batch->dispersion.begin(), batch->dispersion.end(), std::nullopt);
  // Method-owned tokens must follow the same invalidation boundary as the
  // cached diagnostics, including malformed descriptors and output counts.
  try {
    resource_scope.emplace(batch->resource_ledger);
    batch->plan->invalidate_result();
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
  if (results == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const std::uint32_t system_count = generativeqc_batch_get_system_count(batch);
  if (result_count != system_count || ((inputs == nullptr) != (input_count == 0)) ||
      (inputs != nullptr && input_count != system_count)) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::uint32_t i = 0; i < result_count; ++i) {
    if (!generativeqc::api::valid_descriptor(&results[i])) {
      return GENERATIVEQC_STATUS_ABI_MISMATCH;
    }
  }
  if (inputs != nullptr) {
    for (std::uint32_t i = 0; i < input_count; ++i) {
      if (!generativeqc::api::valid_descriptor(&inputs[i])) {
        return GENERATIVEQC_STATUS_ABI_MISMATCH;
      }
    }
  }

  try {
    generativeqc::methods::Coordinates coordinates;
    if (inputs != nullptr) {
      coordinates.resize(system_count);
      for (std::uint32_t i = 0; i < system_count; ++i) {
        if (inputs[i].coordinates == nullptr && inputs[i].coordinate_count == 0) {
          continue;
        }
        if (inputs[i].coordinates == nullptr) {
          // Preserve item-level failure isolation for a malformed coordinate
          // payload without rejecting structurally valid neighboring systems.
          coordinates[i] = std::vector<double>{std::numeric_limits<double>::quiet_NaN()};
          continue;
        }
        coordinates[i] = std::vector<double>(inputs[i].coordinates,
                                             inputs[i].coordinates + inputs[i].coordinate_count);
      }
    }

    // All omitted force buffers request a true energy-only replay. The legacy
    // entry retains its whole-fleet schedule for mixed outputs; the selective
    // bridge requests only the indicated slots. Both validate buffers below.
    const bool compute_forces = std::any_of(results, results + result_count, [](const auto& item) {
      return item.forces != nullptr || item.force_count != 0;
    });
    std::vector<generativeqc::methods::BatchItemResult> native;
    if (selective_forces) {
      // The additive bridge uses the existing output descriptors as per-item
      // work requests. Keep the legacy whole-fleet ABI schedule unchanged.
      std::vector<std::uint8_t> requests(system_count);
      for (std::uint32_t i = 0; i < system_count; ++i)
        requests[i] = results[i].forces != nullptr || results[i].force_count != 0;
      native = batch->plan->execute_selected_forces(coordinates, requests);
    } else {
      native = batch->plan->execute(coordinates, compute_forces);
    }
    if (native.size() != system_count) {
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    for (std::uint32_t i = 0; i < system_count; ++i) {
      generativeqc_batch_item_result_descriptor& output = results[i];
      generativeqc::methods::BatchItemResult& item = native[i];
      if (item.status == GENERATIVEQC_STATUS_SUCCESS ||
          item.status == GENERATIVEQC_STATUS_NOT_CONVERGED) {
        batch->ks_diagnostics[i] = std::move(item.calculation.ks_diagnostic);
        if (item.calculation.preliminary_guess.requested_kind)
          batch->initial_guesses[i] = item.calculation.preliminary_guess;
        batch->precision[i] = item.calculation.precision;
        batch->incremental_direct_jk[i] = item.calculation.incremental_direct_jk;
        if (item.calculation.physical_residual_rms)
          batch->scf_diagnostics[i] = generativeqc_scf_diagnostic{
              sizeof(generativeqc_scf_diagnostic), GENERATIVEQC_ABI_VERSION,
              item.calculation.convergence.residual_rms, *item.calculation.physical_residual_rms};
      }
      // A retry-aware owner publishes only its complete execution-local total.
      // Missing work must not fall through to a successful attempt's count.
      // Legacy owners still require the final census and no discarded attempt.
      const auto& work = item.calculation.precision_work;
      const bool counted_cuda_ks = item.calculation.executed_backend == GENERATIVEQC_BACKEND_CUDA &&
                                   batch->ks_diagnostics[i].has_value() && work.complete &&
                                   work.operator_inventory_complete;
      if (item.target_fock_work) {
        batch->last_fock_builds[i] = item.target_fock_work->total().value_or(0);
      } else if (!item.warm_start_fallback &&
                 (item.calculation.executed_backend == GENERATIVEQC_BACKEND_CPU_REFERENCE ||
                  counted_cuda_ks)) {
        batch->last_fock_builds[i] = item.calculation.fock_builds;
      }
      const std::uint32_t required_forces = batch->atom_counts[i] * 3;
      const bool omit_forces = output.forces == nullptr && output.force_count == 0;
      const bool valid_force_buffer =
          omit_forces || (output.forces != nullptr && output.force_count >= required_forces);
      output.status = valid_force_buffer ? item.status : GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      if (output.status == GENERATIVEQC_STATUS_SUCCESS)
        batch->dispersion[i] = std::move(item.calculation.dispersion);
      if (output.status == GENERATIVEQC_STATUS_SUCCESS ||
          output.status == GENERATIVEQC_STATUS_NOT_CONVERGED) {
        batch->precision_work[i] = std::move(item.calculation.precision_work);
      }
      if ((batch->flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0) {
        output.warm_start_used = item.warm_start_used ? 1 : 0;
        output.warm_start_fallback = item.warm_start_fallback ? 1 : 0;
      }
      if (output.status != GENERATIVEQC_STATUS_SUCCESS &&
          output.status != GENERATIVEQC_STATUS_NOT_CONVERGED)
        continue;
      output.energy = item.calculation.energy;
      output.iterations = item.calculation.convergence.iterations;
      output.energy_change = item.calculation.convergence.energy_change;
      output.density_rms = item.calculation.convergence.residual_rms;
      output.converged = item.calculation.convergence.converged ? 1 : 0;
      output.executed_backend = item.calculation.executed_backend;
      output.bucket_id = static_cast<std::uint32_t>(item.bucket_id);
      output.warm_start_used = item.warm_start_used ? 1 : 0;
      output.warm_start_fallback = item.warm_start_fallback ? 1 : 0;
      if (!omit_forces && item.status == GENERATIVEQC_STATUS_SUCCESS) {
        std::copy(item.calculation.forces.begin(), item.calculation.forces.end(), output.forces);
      }
    }
    publication.commit();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
}

generativeqc_status generativeqc_batch_execute(generativeqc_batch* batch,
                                               const generativeqc_batch_input_descriptor* inputs,
                                               uint32_t input_count,
                                               generativeqc_batch_item_result_descriptor* results,
                                               uint32_t result_count) {
  return execute_batch(batch, inputs, input_count, results, result_count, false);
}

/** Private additive bridge for Python/native KS mixed force consumers. NULL/0
 * force storage requests only energy for that slot; requested forces remain
 * subject to the prepared native capability. Uses the same invalidation,
 * diagnostics and publication transaction as the legacy batch entry point. */
generativeqc_status generativeqc_ks_batch_execute_selected_forces_v1(
    generativeqc_batch* batch, const generativeqc_batch_input_descriptor* inputs,
    uint32_t input_count, generativeqc_batch_item_result_descriptor* results,
    uint32_t result_count) {
  return execute_batch(batch, inputs, input_count, results, result_count, true);
}

generativeqc_status generativeqc_batch_get_precision_provenance(
    const generativeqc_batch* batch, uint32_t index, generativeqc_precision_provenance* out) {
  if (batch == nullptr || index >= batch->precision.size()) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  if (!batch->precision[index].has_value()) return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  return generativeqc::api::copy_precision_provenance(*batch->precision[index], out);
}

generativeqc_status generativeqc_batch_get_incremental_direct_jk_diagnostic(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_incremental_direct_jk_diagnostic* out) {
  if (batch == nullptr || index >= batch->incremental_direct_jk.size()) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  if (!batch->incremental_direct_jk[index].has_value()) {
    return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  }
  return generativeqc::api::copy_incremental_direct_jk_diagnostic(
      *batch->incremental_direct_jk[index], out);
}

generativeqc_status generativeqc_batch_get_precision_work(
    const generativeqc_batch* batch, uint32_t index, uint32_t detail_version,
    generativeqc_precision_work_detail* out, generativeqc_precision_work_event* events,
    uint32_t event_capacity, generativeqc_precision_operator_record* operators,
    uint32_t operator_capacity) {
  if (batch == nullptr || index >= batch->precision_work.size()) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  if (!batch->precision_work[index].has_value()) return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  return generativeqc::api::copy_precision_work(*batch->precision_work[index], detail_version, out,
                                                events, event_capacity, operators,
                                                operator_capacity);
}

generativeqc_status generativeqc_batch_get_last_fock_builds(const generativeqc_batch* batch,
                                                            uint32_t index, uint64_t* builds) {
  if (!batch || !builds || index >= batch->last_fock_builds.size())
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(batch->context->mutex);
  *builds = batch->last_fock_builds[index];
  return *builds ? GENERATIVEQC_STATUS_SUCCESS : GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

}  // extern "C"

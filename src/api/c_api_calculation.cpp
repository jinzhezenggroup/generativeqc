#include <algorithm>
#include <memory>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/initial_guess_diagnostic.hpp"
#include "api/ks_diagnostic.hpp"
#include "api/native_method_inputs.hpp"
#include "api/precision.hpp"
#include "api/prepared_publication.hpp"
#include "generativeqc/generativeqc.h"
#include "methods/method.hpp"
#include "runtime/host_component_trace.hpp"
#include "scf/initial_guess/automatic_policy.hpp"

extern "C" {

uint32_t generativeqc_initial_guess_options_version(void) { return 1; }

uint32_t generativeqc_initial_guess_capabilities_v1(void) {
  return GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HF | GENERATIVEQC_INITIAL_GUESS_CAPABILITY_LDA |
         GENERATIVEQC_INITIAL_GUESS_CAPABILITY_MINAO |
         GENERATIVEQC_INITIAL_GUESS_CAPABILITY_AUTO_CPU |
         GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL |
         GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL_CUDA_RHF |
         GENERATIVEQC_INITIAL_GUESS_CAPABILITY_HUCKEL_DF |
         GENERATIVEQC_INITIAL_GUESS_CAPABILITY_AUTO_HUCKEL_CPU;
}

generativeqc_status generativeqc_initial_guess_auto_kind_v1(
    const generativeqc_initial_guess_auto_policy* request, uint32_t* resolved_kind) {
  if (!request || !resolved_kind) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(request)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if ((request->backend != GENERATIVEQC_BACKEND_CPU_REFERENCE &&
       request->backend != GENERATIVEQC_BACKEND_CUDA) ||
      (request->precision_mode != GENERATIVEQC_PRECISION_FP64 &&
       request->precision_mode != GENERATIVEQC_PRECISION_AUTO) ||
      (request->density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE &&
       request->density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE &&
       request->density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CUDA &&
       request->density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_AUTO &&
       request->density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CORRELATION_CUDA &&
       request->density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_FULL_CUDA) ||
      request->restricted > 1 || request->has_ecp > 1 ||
      (request->atomic_numbers == nullptr) != (request->atom_count == 0))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto* capabilities = generativeqc::methods::find_capabilities(request->method);
  if (!capabilities) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  bool elements_eligible = true;
  for (size_t index = 0; index < request->atom_count; ++index) {
    const auto atomic_number = request->atomic_numbers[index];
    if (atomic_number < 1 || atomic_number > 118) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    elements_eligible =
        elements_eligible &&
        generativeqc::scf::initial_guess::automatic_minao_element_eligible(atomic_number);
  }
  const bool eligible = generativeqc::scf::initial_guess::automatic_minao_model_eligible(
      request->method, capabilities->family, request->backend, request->precision_mode,
      request->density_fitting_mode, request->restricted != 0, request->has_ecp != 0);
  *resolved_kind = eligible && elements_eligible ? GENERATIVEQC_INITIAL_GUESS_MINAO : 0;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_initial_guess_auto_system_kind_v1(
    const generativeqc_initial_guess_auto_policy* request, int64_t electron_count,
    uint32_t multiplicity, uint32_t* resolved_kind) {
  if (!request || !resolved_kind) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(request)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!request->atom_count || electron_count < 0 || !multiplicity)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  uint32_t kind = 0;
  const auto status = generativeqc_initial_guess_auto_kind_v1(request, &kind);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  *resolved_kind = generativeqc::scf::initial_guess::automatic_minao_population_eligible(
                       electron_count, multiplicity)
                       ? kind
                       : 0;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_initial_guess_auto_kind_v2(
    const generativeqc_initial_guess_auto_policy* request, uint32_t* resolved_kind) {
  if (!resolved_kind) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  uint32_t compatibility_kind = 0;
  const auto status = generativeqc_initial_guess_auto_kind_v1(request, &compatibility_kind);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  const auto* capabilities = generativeqc::methods::find_capabilities(request->method);
  bool elements_eligible = true;
  for (size_t index = 0; index < request->atom_count; ++index)
    elements_eligible =
        elements_eligible && generativeqc::scf::initial_guess::automatic_minao_element_eligible(
                                 request->atomic_numbers[index]);
  const bool eligible = generativeqc::scf::initial_guess::automatic_huckel_model_eligible(
      request->method, capabilities->family, request->backend, request->precision_mode,
      request->density_fitting_mode, request->restricted != 0, request->has_ecp != 0);
  *resolved_kind = eligible && elements_eligible ? GENERATIVEQC_INITIAL_GUESS_HUCKEL : 0;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_initial_guess_auto_system_kind_v2(
    const generativeqc_initial_guess_auto_policy* request, int64_t electron_count,
    uint32_t multiplicity, uint32_t* resolved_kind) {
  if (!request || !resolved_kind) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(request)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!request->atom_count || electron_count < 0 || !multiplicity)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  uint32_t kind = 0;
  const auto status = generativeqc_initial_guess_auto_kind_v2(request, &kind);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  *resolved_kind = generativeqc::scf::initial_guess::automatic_minao_population_eligible(
                       electron_count, multiplicity)
                       ? kind
                       : 0;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_calculation_get_initial_guess_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_initial_guess_diagnostic* out) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  return generativeqc::api::copy_initial_guess_diagnostic(calculation->initial_guess, out);
}

generativeqc_status generativeqc_calculation_prepare(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_method_descriptor* descriptor, generativeqc_calculation** calculation) {
  generativeqc::runtime::host_trace::Region trace("calculation_prepare");
  if (context == nullptr || system == nullptr || descriptor == nullptr || calculation == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  *calculation = nullptr;
  if (!generativeqc::api::valid_descriptor(descriptor) ||
      (descriptor->initial_guess &&
       !generativeqc::api::valid_descriptor(descriptor->initial_guess))) {
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  }
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  if (!generativeqc::api::valid_system_roles(*system, *descriptor,
                                             context->state.executed_backend)) {
    context->last_detail =
        "auxiliary handle role requires its matching method and density-fitting backend";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    auto candidate = std::make_unique<generativeqc_calculation>();
    candidate->context = context;
    candidate->plan = generativeqc::methods::prepare_calculation(
        context->state, system->data, *descriptor,
        generativeqc::api::native_auxiliary_basis(*descriptor));
    *calculation = candidate.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

void generativeqc_calculation_destroy(generativeqc_calculation* calculation) {
  generativeqc::runtime::host_trace::Region trace("calculation_destroy");
  delete calculation;
}

generativeqc_status generativeqc_calculation_get_supported_properties_v1(
    const generativeqc_calculation* calculation, generativeqc_property_flags* properties) {
  if (!calculation || !properties) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  *properties = calculation->plan->supported_properties();
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_calculation_execute(generativeqc_calculation* calculation,
                                                     generativeqc_result_descriptor* output) {
  return generativeqc_calculation_execute_components_v1(calculation, output, nullptr, nullptr);
}

generativeqc_status generativeqc_calculation_execute_components_v1(
    generativeqc_calculation* calculation, generativeqc_result_descriptor* output,
    generativeqc_result_descriptor* hf, generativeqc_result_descriptor* mp2) {
  generativeqc::runtime::host_trace::Region trace("calculation_execute");
  if (calculation == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(calculation->context->mutex);
  generativeqc::api::PreparedPublication publication(*calculation->plan);
  calculation->precision_work.reset();
  calculation->initial_guess.reset();
  calculation->mp2_component_diagnostic.reset();
  calculation->precision = {};
  calculation->incremental_direct_jk = {};
  calculation->precision_available = false;
  calculation->scf_diagnostic.reset();
  calculation->ks_diagnostic.reset();
  // An attempted execution revokes any internal final-state token even if
  // the output descriptor is rejected before the method can run.
  try {
    calculation->plan->invalidate_result();
  } catch (...) {
    return generativeqc::api::map_exception(&calculation->context->last_detail);
  }
  if (output == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(output)) {
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  }
  const bool omit_forces = output->forces == nullptr && output->force_count == 0;
  if (output->forces == nullptr && !omit_forces) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!omit_forces && output->force_count < calculation->plan->atom_count() * 3) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  for (auto* companion : {hf, mp2}) {
    if (!companion) continue;
    if (!generativeqc::api::valid_descriptor(companion)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
    const bool companion_omits = !companion->forces && companion->force_count == 0;
    if (companion == output || (hf && mp2 && hf == mp2) || companion_omits != omit_forces ||
        (!omit_forces &&
         (!companion->forces || companion->force_count < calculation->plan->atom_count() * 3)))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto method = calculation->plan->capabilities().method;
  if (!omit_forces) {
    const auto bytes = calculation->plan->atom_count() * 3 * sizeof(double);
    for (const auto& pair : {std::pair{output, hf}, std::pair{output, mp2}, std::pair{hf, mp2}}) {
      if (!pair.first || !pair.second) continue;
      const auto first = reinterpret_cast<std::uintptr_t>(pair.first->forces);
      const auto second = reinterpret_cast<std::uintptr_t>(pair.second->forces);
      if ((first < second ? second - first : first - second) < bytes)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  const bool ccsdt =
      method == GENERATIVEQC_METHOD_RCCSD_T || method == GENERATIVEQC_METHOD_DF_RCCSD_T;
  if ((hf || mp2) && method != GENERATIVEQC_METHOD_MP2 && !ccsdt)
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (mp2 && !ccsdt) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;

  try {
    // NULL/zero is an execution request, not merely a copy-out choice: the
    // backend must not launch or assemble analytic-force work in this mode.
    generativeqc::methods::Result native = calculation->plan->execute(
        !omit_forces, (hf ? generativeqc::methods::kHfComponent : 0) |
                          (mp2 ? generativeqc::methods::kMp2Component : 0));
    if ((hf || mp2) && !native.convergence.converged) return GENERATIVEQC_STATUS_NOT_CONVERGED;
    if ((hf || mp2) && !omit_forces && native.forces.size() != calculation->plan->atom_count() * 3)
      throw std::runtime_error("native endpoint returned incomplete primary forces");
    for (const auto& pair :
         {std::pair{hf, &native.hf_component}, std::pair{mp2, &native.mp2_component}}) {
      if (pair.first &&
          (!*pair.second ||
           (!omit_forces && (*pair.second)->forces.size() != calculation->plan->atom_count() * 3)))
        throw std::runtime_error("native endpoint returned an incomplete companion");
    }
    // A normal return (converged or not) is a completed run: record what ran.
    calculation->precision = native.precision;
    calculation->incremental_direct_jk = native.incremental_direct_jk;
    calculation->precision_available = true;
    calculation->ks_diagnostic = std::move(native.ks_diagnostic);
    if (native.preliminary_guess.requested_kind)
      calculation->initial_guess = native.preliminary_guess;
    if (native.physical_residual_rms) {
      calculation->scf_diagnostic = generativeqc_scf_diagnostic{
          sizeof(generativeqc_scf_diagnostic), GENERATIVEQC_ABI_VERSION,
          native.convergence.residual_rms, *native.physical_residual_rms};
    }
    output->energy = native.energy;
    output->iterations = native.convergence.iterations;
    output->energy_change = native.convergence.energy_change;
    output->density_rms = native.convergence.residual_rms;
    output->converged = native.convergence.converged ? 1 : 0;
    output->executed_backend = native.executed_backend;
    if (!native.convergence.converged) {
      calculation->precision_work = std::move(native.precision_work);
      // A retained context may still hold an earlier, unrelated failure.
      // This completed nonconverged run is a new failure and replaces it;
      // successful calls continue to preserve borrowed diagnostic strings.
      calculation->context->last_detail = "SCF did not converge";
      publication.commit();
      return GENERATIVEQC_STATUS_NOT_CONVERGED;
    }
    if (!omit_forces) {
      if (native.forces.size() > output->force_count) {
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
      std::copy(native.forces.begin(), native.forces.end(), output->forces);
    }
    for (const auto& pair :
         {std::pair{hf, &native.hf_component}, std::pair{mp2, &native.mp2_component}}) {
      if (!pair.first) continue;
      auto& target = *pair.first;
      auto* force_buffer = target.forces;
      const auto capacity = target.force_count;
      target = *output;
      target.forces = force_buffer;
      target.force_count = capacity;
      target.energy = (*pair.second)->energy;
      if (!omit_forces)
        std::copy((*pair.second)->forces.begin(), (*pair.second)->forces.end(), target.forces);
    }
    if (native.mp2_component)
      calculation->mp2_component_diagnostic = native.mp2_component->correlation;
    calculation->precision_work = std::move(native.precision_work);
    publication.commit();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&calculation->context->last_detail);
  }
}

generativeqc_status generativeqc_calculation_get_mp2_component_diagnostic_v1(
    const generativeqc_calculation* calculation, generativeqc_correlation_diagnostic* out) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (out && !generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!calculation->mp2_component_diagnostic) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (out) *out = *calculation->mp2_component_diagnostic;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_calculation_get_scf_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_scf_diagnostic* out) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (out && !generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!calculation->scf_diagnostic) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (out) *out = *calculation->scf_diagnostic;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_calculation_get_ks_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_ks_diagnostic* out,
    generativeqc_ks_iteration* history, uint32_t history_capacity) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  return generativeqc::api::copy_ks_diagnostic(calculation->ks_diagnostic, out, history,
                                               history_capacity);
}

generativeqc_status generativeqc_calculation_get_ks_coulomb_exchange_work_v1(
    const generativeqc_calculation* calculation, generativeqc_ks_coulomb_exchange_work_v1* out) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  return generativeqc::api::copy_ks_coulomb_exchange_work(calculation->ks_diagnostic, out);
}

generativeqc_status generativeqc_calculation_get_ks_transport_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_ks_transport_diagnostic* out) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (out && !generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  try {
    const auto source = calculation->plan->ks_transport_diagnostic();
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
    return generativeqc::api::map_exception(&calculation->context->last_detail);
  }
}

generativeqc_status generativeqc_calculation_get_precision_provenance(
    const generativeqc_calculation* calculation, generativeqc_precision_provenance* out) {
  if (calculation == nullptr) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  // A completed run (converged or not) populates \p precision and sets
  // \p precision_available; execute() resets it to false before the run so a
  // failed or not-yet-run execution never exposes a stale record. Gate both the
  // availability query (a NULL \p out) and the copy-out on it so callers see
  // an honest non-success result until a run has actually resolved.
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!calculation->precision_available) {
    return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  }
  return generativeqc::api::copy_precision_provenance(calculation->precision, out);
}

generativeqc_status generativeqc_calculation_get_incremental_direct_jk_diagnostic(
    const generativeqc_calculation* calculation,
    generativeqc_incremental_direct_jk_diagnostic* out) {
  if (calculation == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!calculation->precision_available) return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  return generativeqc::api::copy_incremental_direct_jk_diagnostic(
      calculation->incremental_direct_jk, out);
}

generativeqc_status generativeqc_calculation_get_precision_work(
    const generativeqc_calculation* calculation, uint32_t detail_version,
    generativeqc_precision_work_detail* out, generativeqc_precision_work_event* events,
    uint32_t event_capacity, generativeqc_precision_operator_record* operators,
    uint32_t operator_capacity) {
  if (calculation == nullptr) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!calculation->precision_work.has_value()) return GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE;
  return generativeqc::api::copy_precision_work(*calculation->precision_work, detail_version, out,
                                                events, event_capacity, operators,
                                                operator_capacity);
}

generativeqc_status generativeqc_calculation_get_correlation_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_correlation_diagnostic* diagnostic) {
  if (!calculation || !diagnostic) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!generativeqc::api::valid_descriptor(diagnostic)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  const auto value = calculation->plan->correlation_diagnostic();
  if (!value) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  *diagnostic = *value;
  return GENERATIVEQC_STATUS_SUCCESS;
}

// Private v1 telemetry: six copied counters, independent of the frozen public
// performance descriptor. Query while the calculation and its owner are alive.
generativeqc_status generativeqc_calculation_get_cc_precision_v1(
    const generativeqc_calculation* calculation, uint64_t* values, uint32_t count) {
  if (!calculation || !values || count != 6) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  try {
    const auto value = calculation->plan->cc_performance_diagnostic();
    if (!value) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    values[0] = value->ccsd_fp32_requested;
    values[1] = value->ccsd_fp32_executed;
    values[2] = value->ccsd_fp32_refused;
    values[3] = value->ccsd_fp32_evaluations;
    values[4] = value->ccsd_fp32_gemm_calls;
    values[5] = value->ccsd_fp32_cast_elements;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&calculation->context->last_detail);
  }
}

// Private additive DF response extension. Availability is distinct from a zero
// measurement, and both output spans are validated before dereferencing a handle.
generativeqc_status generativeqc_calculation_get_df_response_v1(
    const generativeqc_calculation* calculation, uint64_t* counts, uint32_t count, double* seconds,
    uint32_t phases) {
  if (!calculation || !counts || count != 12 || !seconds || phases != 3)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  try {
    const auto value = calculation->plan->cc_performance_diagnostic();
    if (!value || !value->df_response_available) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    counts[0] = value->lambda_fp32_requested;
    counts[1] = value->lambda_fp32_executed;
    counts[2] = value->lambda_fp32_refused;
    counts[3] = value->lambda_fp32_iterations;
    counts[4] = value->lambda_fp64_iterations;
    counts[5] = value->lambda_fp32_actions;
    counts[6] = value->lambda_fp64_actions;
    counts[7] = value->lambda_fp32_gemm_calls;
    counts[8] = value->lambda_fp32_cast_elements;
    counts[9] = value->lambda_auxiliary_batch_size;
    counts[10] = value->lambda_numeric_capacity_bytes;
    counts[11] = value->lambda_owned_device_bytes;
    seconds[0] = value->lambda_seconds;
    seconds[1] = value->source_response_seconds;
    seconds[2] = value->orbital_seconds;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&calculation->context->last_detail);
  }
}

generativeqc_status generativeqc_calculation_get_cc_performance_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_cc_performance_diagnostic* out) {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (out && !generativeqc::api::valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  try {
    const auto value = calculation->plan->cc_performance_diagnostic();
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
    return generativeqc::api::map_exception(&calculation->context->last_detail);
  }
}

}  // extern "C"

// Cold endpoint adapter: RawSource supplies normalized metadata only. Every
// electronic value, amplitude, adjoint and nuclear contraction is native CUDA.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "methods/df_ccsdt_force.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/execution_context.hpp"

namespace {
generativeqc_method_descriptor force_descriptor(std::size_t budget) {
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = GENERATIVEQC_METHOD_RCCSD;
  descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
  descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  descriptor.energy_tolerance = 1e-12;
  descriptor.density_tolerance = 1e-12;
  descriptor.ccsd_diis_history = 6;
  descriptor.ccsd_max_iterations = 200;
  descriptor.ccsd_energy_tolerance = 1e-12;
  descriptor.ccsd_residual_tolerance = 1e-10;
  descriptor.correlation_memory_budget_bytes = budget;
  return descriptor;
}
}  // namespace

extern "C" int df_complete_force_probe(void* opaque, bool forces, bool triples, std::size_t budget,
                                       double* force_output, double* values, std::size_t* counts,
                                       char* error, std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    core::ContextState context;
    context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
    runtime::ExecutionContext execution(context);
    const auto descriptor = force_descriptor(budget);
    hf::RHFFrameResponseOptions options;
    // Test-only selection runs unchanged independent FD gates through the
    // optional accelerator without changing public method semantics.
    if (const auto* mode = std::getenv("GENERATIVEQC_TEST_Z_PRECONDITIONER"))
      options.df_preconditioning = std::string(mode) == "1";
    if (const auto* resident = std::getenv("GENERATIVEQC_TEST_RHF_RESIDENT_JK_BYTES"))
      options.resident_jk_maximum_bytes = std::stoull(resident);
    const auto* selected = std::getenv("GENERATIVEQC_TEST_PACKED_CC_HISTORY");
    const bool packed = selected && std::string(selected) == "1";
    const auto* gap_selected = std::getenv("GENERATIVEQC_TEST_PARALLEL_GAP_RESPONSE");
    const bool parallel_gap = gap_selected && std::string(gap_selected) == "1";
    const auto* gap_omitted = std::getenv("GENERATIVEQC_TEST_OMIT_GAP_RESPONSE");
    const bool include_gap = !(gap_omitted && std::string(gap_omitted) == "1");
    const auto result = methods::detail::run_df_ccsdt_native(
        execution, raw.orbital(), raw.auxiliary(), descriptor, forces, triples, true, true, true, 8,
        8, options, true, packed, parallel_gap, include_gap);
    if (forces && triples && !include_gap &&
        (result.triples_gap.requested || result.triples_gap.kernels))
      throw std::runtime_error("complete-force qualification did not omit unrequested gap outputs");
    if (forces && options.resident_jk_maximum_bytes.value_or(0) > 0 &&
        !result.orbital.resident_jk_bytes)
      throw std::runtime_error(
          "resident complete-force qualification did not exercise source reuse");
    if (packed && !result.solver.packed_diis)
      throw std::runtime_error("packed history qualification did not exercise packed storage");
    const double scalars[]{result.energy,
                           result.reference_energy,
                           result.correlation_energy,
                           result.triples_energy,
                           result.lambda.lambda_residual_norm,
                           result.lambda.independent_residual_norm,
                           result.orbital.orbital_residual,
                           result.orbital.maximum_stationarity,
                           result.total_seconds,
                           result.primal.reference_seconds,
                           result.primal.problem_seconds,
                           result.primal.solver_seconds,
                           result.triples_seconds,
                           result.lambda_seconds,
                           result.source_response_seconds,
                           result.orbital_seconds};
    const std::size_t work[]{result.numeric_capacity_bytes,
                             result.source_weight_values,
                             result.metric_weight_values,
                             result.lambda.iterations,
                             result.orbital.orbital_response.iterations,
                             result.orbital.jk_actions,
                             result.orbital.explicit_hessian_elements,
                             result.orbital.derivative_passes,
                             result.triples.occupied_tiles,
                             result.triples.contraction_summands,
                             result.triples_fock.contraction_summands,
                             result.lambda.df_contraction_terms};
    std::copy(result.forces.begin(), result.forces.end(), force_output);
    std::copy(std::begin(scalars), std::end(scalars), values);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}

extern "C" int df_gap_same_primal_probe(void* opaque, std::size_t budget, double* force_output,
                                        double* values, std::uint64_t* common_counts,
                                        std::size_t* case_counts, char* error,
                                        std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    core::ContextState context;
    context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
    runtime::ExecutionContext execution(context);
    const auto comparison = methods::detail::diagnose_df_ccsdt_gap_schedules(
        execution, raw.orbital(), raw.auxiliary(), force_descriptor(budget));
    const std::uint64_t shared[]{comparison.retained_primal_host_bytes,
                                 comparison.retained_df_source_bytes,
                                 comparison.retained_exact_source_bytes,
                                 comparison.output_bytes,
                                 comparison.clone_admission_bytes,
                                 comparison.source_identity,
                                 comparison.denominator_identity,
                                 comparison.primal_identity,
                                 comparison.nocc,
                                 comparison.nvir,
                                 comparison.naux,
                                 static_cast<std::uint64_t>(comparison.reference_iterations)};
    std::copy(std::begin(shared), std::end(shared), common_counts);
    std::size_t force_offset = 0;
    for (std::size_t index = 0; index < comparison.cases.size(); ++index) {
      const auto& snapshot = comparison.cases[index];
      const double scalars[]{snapshot.energy,
                             snapshot.triples_energy,
                             snapshot.lambda.independent_residual_norm,
                             snapshot.orbital_residual,
                             snapshot.maximum_stationarity,
                             snapshot.clone_seconds,
                             snapshot.response_seconds};
      const std::size_t counts[]{snapshot.numeric_capacity_bytes,
                                 snapshot.gap.requested,
                                 snapshot.gap.parallel,
                                 snapshot.gap.kernels,
                                 snapshot.gap.workspace_bytes,
                                 snapshot.gap.materialized_elements,
                                 snapshot.source_weight_values,
                                 snapshot.metric_weight_values,
                                 snapshot.lambda.df_contraction_terms,
                                 snapshot.fock_response_work};
      std::copy(snapshot.forces.begin(), snapshot.forces.end(), force_output + force_offset);
      force_offset += snapshot.forces.size();
      std::copy(std::begin(scalars), std::end(scalars), values + index * std::size(scalars));
      std::copy(std::begin(counts), std::end(counts), case_counts + index * std::size(counts));
    }
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}

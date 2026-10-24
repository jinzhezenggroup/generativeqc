#pragma once

#include <fstream>
#include <iomanip>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

#include "methods/df_ccsdt_force.hpp"

namespace generativeqc::benchmarks {

// Reporting inputs only; scientific invocation and lifecycle accumulation stay
// with the benchmark owner. A missing phase retains its native zero/flag pair.
struct DFCCSDTReportOptions {
  std::size_t nbf{}, naux{};
  int repetition{};
  bool forces{}, reduction{}, matrix{};
  double reference_energy_tolerance{}, reference_density_tolerance{};
  std::size_t lambda_true_residual_interval{}, z_true_residual_interval{};
  std::optional<bool> ccsd_mixed_precision;
  double ccsd_mixed_switch_residual{};
  unsigned long long diis_history{};
  double previous_endpoint_seconds{};
  bool triples_w_fp32{}, lambda_core_reuse{}, lambda_audit_matrix{}, lambda_primal_matrix{},
      lambda_serial_scratch_reuse{}, lambda_mixed_precision{};
};

// Serialize the actual native diagnostic. No scientific computation is performed.
inline void write_df_ccsdt_force_report(std::ostream& output,
                                        const methods::detail::DFCCSDTResult& result,
                                        const DFCCSDTReportOptions& options) {
  output << std::setprecision(17) << "{\n";
  const auto field = [&](const char* name, auto value) {
    output << "  " << std::quoted(name) << ": " << value << ",\n";
  };
  field("lifecycle_stage", std::quoted(options.repetition == 2 ? "moved-warm"
                                       : options.repetition    ? "warm"
                                                               : "cold"));
  // A discarded attempt can include any completed phase. Final-attempt
  // counters, phase times and peaks cannot reconstruct total attempted work.
  const auto work_field = [&](const char* name, auto value) {
    if (result.recycling_discarded_primal_attempt || result.orbital.resident_jk_discarded_attempt)
      field(name, "null");
    else
      field(name, value);
  };
  field("nbf", options.nbf);
  field("naux", options.naux);
  // Distinguish phases absent by request from measured zero-cost phases.
  field("forces_requested", options.forces ? 1 : 0);
  field("lambda_reduction_requested", options.reduction ? 1 : 0);
  field("matrix_gemm_requested", options.matrix ? 1 : 0);
  field("total_energy", result.energy);
  field("reference_energy", result.reference_energy);
  field("reference_energy_tolerance", options.reference_energy_tolerance);
  field("reference_density_tolerance", options.reference_density_tolerance);
  field("reference_energy_change", result.reference_energy_change);
  field("reference_density_rms", result.reference_density_rms);
  work_field("reference_iterations", result.reference_iterations);
  field("correlation_energy", result.correlation_energy);
  field("triples_energy", result.triples_energy);
  field("lambda_residual", result.lambda.independent_residual_norm);
  work_field("lambda_true_residual_interval", options.lambda_true_residual_interval);
  field("z_residual", result.orbital.orbital_residual);
  field("stationarity", result.orbital.maximum_stationarity);
  work_field("numeric_capacity_bytes", result.numeric_capacity_bytes);
  field("native_seconds", result.total_seconds);
  field("resident_jk_discarded_attempt", result.orbital.resident_jk_discarded_attempt ? 1 : 0);
  field("resident_jk_retry_seconds", result.orbital.resident_jk_retry_seconds);
  work_field("reference_seconds", result.primal.reference_seconds);
  work_field("source_seconds", result.primal.problem_seconds);
  work_field("df_source_scans", result.primal.source_scans);
  work_field("df_source_reads", result.primal.source_reads);
  work_field("df_source_values", result.primal.source_values);
  work_field("df_transform_fmas", result.primal.transform_fmas);
  work_field("df_transform_stages", result.primal.transform_stages);
  work_field("df_block_gemms", result.primal.mo_blocks);
  work_field("df_cuda_transform_calls", result.primal.cuda_transform_calls);
  work_field("ccsd_seconds", result.primal.solver_seconds);
  field("ccsd_matrix_gemm", result.solver.df_matrix_gemm ? 1 : 0);
  if (options.ccsd_mixed_precision)
    field("ccsd_mixed_precision_requested", *options.ccsd_mixed_precision);
  else
    field("ccsd_mixed_precision_requested", "null");
  field(
      "ccsd_mixed_precision_policy",
      std::quoted(options.ccsd_mixed_precision ? (*options.ccsd_mixed_precision ? "mixed" : "fp64")
                                               : "auto"));
  field("ccsd_mixed_switch_residual", options.ccsd_mixed_switch_residual);
  field("ccsd_mixed_precision", result.solver.df_mixed_precision);
  field("ccsd_mixed_precision_refused", result.solver.df_mixed_precision_refused);
  output << "  \"ccsd_mixed_program_hash\": "
         << std::quoted(result.solver.df_mixed_program_hash ? result.solver.df_mixed_program_hash
                                                            : "")
         << ",\n";
  work_field("ccsd_mixed_evaluations", result.solver.df_mixed_evaluations);
  work_field("ccsd_fp32_gemm_calls", result.solver.df_fp32_gemm_calls);
  work_field("ccsd_fp32_gemm_summands", result.solver.df_fp32_gemm_summands);
  work_field("ccsd_precision_cast_elements", result.solver.df_precision_cast_elements);
  work_field("ccsd_precision_switches", result.solver.df_precision_switches);
  work_field("ccsd_mixed_pair_projection_calls", result.solver.df_mixed_pair_projection_calls);
  work_field("ccsd_mixed_pair_projection_bytes", result.solver.df_mixed_pair_projection_bytes);
  work_field("ccsd_strict_batch_size", result.solver.df_strict_auxiliary_batch_size);
  field("ccsd_replay_matrix_gemm", result.solver.df_replay_matrix_gemm ? 1 : 0);
  work_field("ccsd_gemm_calls", result.solver.df_gemm_calls);
  work_field("ccsd_gemm_summands", result.solver.df_gemm_summands);
  work_field("ccsd_packing_bytes", result.solver.df_packing_bytes);
  work_field("ccsd_provider_capacity", result.solver.df_provider_capacity_bytes);
  field("ccsd_q_batch_size", result.solver.df_auxiliary_batch_size);
  work_field("ccsd_q_tiles", result.solver.df_auxiliary_tiles);
  work_field("ccsd_q_slices", result.solver.df_auxiliary_slices);
  work_field("ccsd_q_operations", result.solver.df_virtual_operations);
  work_field("ccsd_accumulation_calls", result.solver.df_accumulation_calls);
  work_field("ccsd_accumulation_bytes", result.solver.df_accumulation_bytes);
  work_field("ccsd_contraction_terms", result.solver.df_contraction_terms);
  field("ccsd_occupied_pairs", result.solver.df_occupied_pairs ? 1 : 0);
  field("ccsd_pair_resource_refused", result.solver.df_pair_resource_refused ? 1 : 0);
  field("ccsd_pair_initial_symmetry_refused",
        result.solver.df_pair_initial_symmetry_refused ? 1 : 0);
  work_field("ccsd_pair_evaluations", result.solver.df_pair_evaluations);
  work_field("ccsd_pair_refusals", result.solver.df_pair_refusals);
  work_field("ccsd_pair_projection_calls", result.solver.df_pair_projection_calls);
  work_field("ccsd_pair_projection_bytes", result.solver.df_pair_projection_bytes);
  work_field("ccsd_pair_geometry_elements", result.solver.df_pair_geometry_elements);
  work_field("ccsd_pair_capacity_bytes", result.solver.df_pair_capacity_bytes);
  work_field("ccsd_pair_binding_host_bytes", result.solver.df_pair_binding_host_bytes);
  work_field("ccsd_evaluations", result.solver.iteration_graph_calls);
  work_field("ccsd_replay_evaluations", result.solver.replay_graph_calls);
  work_field("ccsd_capacity", result.solver.numeric_capacity_bytes);
  work_field("ccsd_device_bytes", result.solver.owned_device_bytes);
  work_field("ccsd_setup_h2d_bytes", result.solver.setup_h2d_bytes);
  field("denominator_identity", result.solver.denominator_identity);
  work_field("derived_d2_iteration_evaluations", result.solver.derived_d2_iteration_evaluations);
  work_field("ccsd_iterations", result.solver.iterations);
  field("ccsd_replay_r1_max", result.solver.replay_r1_max);
  field("ccsd_replay_r2_max", result.solver.replay_r2_max);
  field("ccsd_diis_history", options.diis_history);
  work_field("ccsd_diis_seconds", result.solver.diis_seconds);
  work_field("ccsd_iteration_seconds", result.solver.iteration_seconds);
  work_field("ccsd_replay_seconds", result.solver.replay_seconds);
  work_field("ccsd_update_seconds", result.solver.update_seconds);
  work_field("ccsd_diis_restarts", result.solver.diis_restarts);
  field("ccsd_packed_diis", result.solver.packed_diis ? 1 : 0);
  field("ccsd_packed_diis_refused", result.solver.packed_diis_refused ? 1 : 0);
  field("ccsd_diis_disabled_after_packing_refusal",
        result.solver.diis_disabled_after_packing_refusal ? 1 : 0);
  work_field("ccsd_diis_history_capacity_bytes", result.solver.diis_history_capacity_bytes);
  work_field("ccsd_diis_conversion_bytes", result.solver.diis_conversion_bytes);
  work_field("ccsd_diis_metric_weight_terms", result.solver.diis_metric_weight_terms);
  work_field("ccsd_diis_pack_calls", result.solver.diis_pack_calls);
  field("ccsd_diis_maximum_pair_asymmetry", result.solver.diis_maximum_pair_asymmetry);
  work_field("ccsd_diis_gram_calls", result.solver.diis_gram_calls);
  work_field("ccsd_diis_coefficient_calls", result.solver.diis_coefficient_calls);
  work_field("ccsd_diis_combine_calls", result.solver.diis_combine_calls);
  work_field("ccsd_diis_insert_bytes", result.solver.diis_history_insert_bytes);
  work_field("ccsd_diis_shift_bytes", result.solver.diis_history_shift_bytes);
  work_field("ccsd_diis_dot_terms", result.solver.diis_residual_dot_terms);
  work_field("ccsd_diis_gram_updates", result.solver.diis_gram_updates);
  work_field("ccsd_diis_combine_terms", result.solver.diis_combine_terms);
  work_field("triples_seconds", result.triples_seconds);
  work_field("lambda_seconds", result.lambda_seconds);
  work_field("source_response_seconds", result.source_response_seconds);
  work_field("orbital_seconds", result.orbital_seconds);
  work_field("jk_actions", result.orbital.jk_actions);
  work_field("resident_jk_bytes", result.orbital.resident_jk_bytes);
  work_field("resident_jk_values", result.orbital.resident_jk_values);
  work_field("resident_jk_actions", result.orbital.resident_jk_actions);
  work_field("resident_jk_value_reads", result.orbital.resident_jk_value_reads);
  work_field("resident_jk_setup_seconds", result.orbital.resident_jk_setup_seconds);
  work_field("resident_jk_seconds", result.orbital.resident_jk_seconds);
  work_field("resident_jk_device_seconds", result.orbital.resident_jk_device_seconds);
  work_field("z_iterations", result.orbital.orbital_response.iterations);
  work_field("z_operator_actions", result.orbital.orbital_response.operator_actions);
  work_field("z_preconditioner_actions", result.orbital.orbital_response.preconditioner_actions);
  field("z_true_residual_interval", options.z_true_residual_interval);
  field("z_df_preconditioned", result.orbital.df_preconditioned ? 1 : 0);
  field("z_preconditioner_fallback", result.orbital.preconditioner_fallback ? 1 : 0);
  work_field("z_preconditioner_setup_seconds", result.orbital.preconditioner_setup_seconds);
  work_field("z_preconditioner_capacity_bytes", result.orbital.preconditioner_capacity_bytes);
  work_field("z_preconditioner_planned_contraction_terms",
             result.orbital.preconditioner_contraction_terms);
  field("z_recycled_guess", result.orbital.recycled_guess ? 1 : 0);
  field("z_recycle_published", result.orbital.recycle_published ? 1 : 0);
  work_field("z_recycle_capacity_bytes", result.orbital.recycle_capacity_bytes);
  field("previous_endpoint_seconds", options.previous_endpoint_seconds);
  field("recycling_discarded_primal_attempt", result.recycling_discarded_primal_attempt ? 1 : 0);
  work_field("orbital_setup_seconds", result.orbital.setup_seconds);
  work_field("orbital_reference_audit_seconds", result.orbital.reference_audit_seconds);
  work_field("orbital_weights_seconds", result.orbital.weights_seconds);
  work_field("orbital_solve_seconds", result.orbital.solve_seconds);
  work_field("orbital_independent_audit_seconds", result.orbital.independent_audit_seconds);
  work_field("orbital_one_electron_seconds", result.orbital.one_electron_seconds);
  work_field("orbital_two_electron_seconds", result.orbital.two_electron_seconds);
  field("orbital_bilinear_derivative_used", result.orbital.bilinear_derivative_used);
  field("orbital_symmetric_polarization_used", result.orbital.symmetric_polarization_used);
  field("orbital_derivative_census_measured", result.orbital.derivative_census_measured);
  work_field("orbital_derivative_quartet_visits", result.orbital.derivative_quartet_visits);
  work_field("orbital_derivative_jet_evaluations", result.orbital.derivative_jet_evaluations);
  work_field("orbital_jk_seconds", result.orbital.jk_seconds);
  work_field("orbital_jk_device_seconds", result.orbital.jk_device_seconds);
  work_field("orbital_h2d_bytes", result.orbital.h2d_bytes);
  work_field("orbital_d2h_bytes", result.orbital.d2h_bytes);
  work_field("orbital_synchronizations", result.orbital.synchronizations);
  work_field("orbital_screened_jk_seconds", result.orbital.screened_jk_seconds);
  field("orbital_screened_residual", result.orbital.screened_residual);
  field("orbital_requested_screening", result.orbital.requested_screening);
  field("orbital_applied_screening", result.orbital.applied_screening);
  work_field("orbital_screened_jk_actions", result.orbital.screened_jk_actions);
  work_field("orbital_jk_census_actions", result.orbital.jk_census_actions);
  work_field("orbital_exact_refinements", result.orbital.exact_refinements);
  work_field("orbital_screened_iterations", result.orbital.screened_iterations);
  work_field("orbital_screened_operator_actions", result.orbital.screened_operator_actions);
  work_field("orbital_jk_quartet_visits", result.orbital.jk_quartet_visits);
  work_field("orbital_jk_eri_evaluations", result.orbital.jk_eri_evaluations);
  field("orbital_jk_timing_measured", result.orbital.jk_timing_measured);
  field("orbital_linear_screening_available", result.orbital.linear_screening_available);
  field("orbital_screened_converged", result.orbital.screened_converged);
  work_field("orbital_derivative_passes", result.orbital.derivative_passes);
  work_field("orbital_shell_derivative_passes", result.orbital.shell_derivative_passes);
  work_field("orbital_generic_derivative_passes", result.orbital.generic_derivative_passes);
  field("hessian_elements", result.orbital.explicit_hessian_elements);
  work_field("source_weight_values", result.source_weight_values);
  work_field("metric_weight_values", result.metric_weight_values);
  work_field("triples_work", result.triples.contraction_summands);
  work_field("triples_occupied_tiles", result.triples.occupied_tiles);
  work_field("triples_panel_capacity", result.triples.panel_capacity);
  work_field("triples_panel_calls", result.triples.panel_gemms);
  // Logical two-product W work; batched vendor calls and device kernels
  // have independent counts and must not be equated to this quantity.
  work_field("triples_moment_products", result.triples.moment_gemms);
  work_field("triples_epilogue_kernels", result.triples.epilogue_kernels);
  work_field("triples_reduction_kernels", result.triples.reduction_kernels);
  work_field("triples_reserved_bytes", result.triples.workspace_bytes);
  work_field("triples_arena_bytes", result.triples.arena_bytes);
  field("triples_w_fp32_requested", options.triples_w_fp32);
  field("triples_w_resource_fallback", result.triples.resource_fallback);
  work_field("triples_w_storage_bits", result.triples.w_contraction_storage_bits);
  work_field("triples_w_compute_bits", result.triples.w_contraction_compute_bits);
  work_field("triples_w_accumulation_bits", result.triples.w_contraction_accumulation_bits);
  work_field("triples_fp64_gemms", result.triples.fp64_gemms);
  work_field("triples_fp32_gemms", result.triples.fp32_gemms);
  work_field("triples_precision_cast_elements", result.triples.precision_cast_elements);
  output << "  \"triples_w_candidate_identity\": "
         << std::quoted(result.triples.w_candidate_identity) << ",\n";
  output << "  \"triples_w_precision_schedule_identity\": "
         << std::quoted(result.triples.w_codegen_precision_schedule_identity) << ",\n";
  field("triples_gap_parallel", result.triples_gap.parallel);
  field("triples_gap_requested", result.triples_gap.requested);
  output << "  \"triples_gap_schedule\": " << std::quoted(result.triples_gap.schedule) << ",\n";
  work_field("triples_gap_kernels", result.triples_gap.kernels);
  work_field("triples_gap_workspace_bytes", result.triples_gap.workspace_bytes);
  work_field("triples_gap_materialized_elements", result.triples_gap.materialized_elements);
  work_field("triples_gap_value_reads", result.triples_gap.value_reads);
  work_field("triples_gap_value_writes", result.triples_gap.value_writes);
  work_field("triples_gap_reduction_summands", result.triples_gap.reduction_summands);
  field("triples_scalar_fusion_requested", result.triples_scalar_fusion.requested);
  field("triples_scalar_fusion_selected", result.triples_scalar_fusion.selected);
  field("triples_scalar_fusion_resource_fallback", result.triples_scalar_fusion.resource_fallback);
  output << "  \"triples_scalar_fusion_schedule\": "
         << std::quoted(result.triples_scalar_fusion.schedule) << ",\n";
  work_field("triples_scalar_fusion_tiles", result.triples_scalar_fusion.tiles);
  work_field("triples_scalar_fusion_kernels", result.triples_scalar_fusion.kernels);
  work_field("triples_scalar_fusion_avoided_kernels", result.triples_scalar_fusion.avoided_kernels);
  work_field("triples_scalar_fusion_workspace_bytes", result.triples_scalar_fusion.workspace_bytes);
  work_field("triples_scalar_fusion_value_reads", result.triples_scalar_fusion.value_reads);
  work_field("triples_scalar_fusion_value_writes", result.triples_scalar_fusion.value_writes);
  work_field("triples_scalar_fusion_arithmetic_ops", result.triples_scalar_fusion.arithmetic_ops);
  work_field("fock_response_work", result.triples_fock.contraction_summands);
  work_field("fock_response_w_gemms", result.triples_fock.w_gemms);
  work_field("fock_response_w_copy_calls", result.triples_fock.w_copy_calls);
  work_field("fock_response_w_copy_bytes", result.triples_fock.w_copy_bytes);
  work_field("fock_response_panel_gemms", result.triples_fock.panel_gemms);
  work_field("fock_response_page_builds", result.triples_fock.page_builds);
  work_field("fock_response_vector_cubes", result.triples_fock.vector_cubes);
  work_field("lambda_work", result.lambda.df_contraction_terms);
  field("lambda_matrix_gemm", result.lambda.df_matrix_gemm);
  field("lambda_batch_size", result.lambda.df_auxiliary_batch_size);
  work_field("lambda_batches", result.lambda.df_auxiliary_batches);
  work_field("lambda_gemm_calls", result.lambda.df_gemm_calls);
  work_field("lambda_gemm_summands", result.lambda.df_gemm_summands);
  work_field("lambda_packing_output_bytes", result.lambda.df_packing_output_bytes);
  work_field("lambda_provider_allowance", result.lambda.df_provider_allowance_bytes);
  work_field("lambda_shared_scratch_bytes", result.lambda.df_shared_scratch_bytes);
  work_field("lambda_scratch_avoided_bytes", result.lambda.df_scratch_avoided_bytes);
  work_field("lambda_available_device_bytes", result.lambda.df_available_device_bytes);
  work_field("lambda_device_limit_bytes", result.lambda.df_device_limit_bytes);
  output << "  \"lambda_shared_program_hash\": "
         << std::quoted(result.lambda.shared_program_hash ? result.lambda.shared_program_hash : "")
         << ",\n  \"lambda_independent_program_hash\": "
         << std::quoted(result.lambda.independent_program_hash
                            ? result.lambda.independent_program_hash
                            : "")
         << ",\n";
  field("lambda_core_reuse_requested", options.lambda_core_reuse);
  field("lambda_core_reuse", result.lambda.df_core_reuse);
  work_field("lambda_core_reuse_bytes", result.lambda.df_core_reuse_bytes);
  work_field("lambda_core_reuse_preparations", result.lambda.df_core_reuse_preparations);
  work_field("lambda_core_reuse_actions", result.lambda.df_core_reuse_actions);
  field("lambda_audit_matrix_requested", options.lambda_audit_matrix);
  field("lambda_audit_matrix", result.lambda.df_audit_matrix_gemm);
  field("lambda_primal_matrix_requested", options.lambda_primal_matrix);
  field("lambda_serial_scratch_reuse_requested", options.lambda_serial_scratch_reuse);
  field("lambda_primal_matrix", result.lambda.df_primal_matrix_gemm);
  field("lambda_mixed_precision_requested", options.lambda_mixed_precision);
  field("lambda_mixed_precision", result.lambda.df_mixed_precision);
  field("lambda_mixed_precision_refused", result.lambda.df_mixed_precision_refused);
  work_field("lambda_mixed_storage_bytes", result.lambda.df_mixed_storage_bytes);
  work_field("lambda_mixed_input_offset_bytes", result.lambda.df_mixed_input_offset_bytes);
  work_field("lambda_mixed_actions", result.lambda.df_mixed_actions);
  work_field("lambda_strict_actions", result.lambda.df_strict_actions);
  work_field("lambda_precision_switches", result.lambda.df_precision_switches);
  work_field("lambda_fp32_gemm_calls", result.lambda.df_fp32_gemm_calls);
  work_field("lambda_fp32_gemm_summands", result.lambda.df_fp32_gemm_summands);
  work_field("lambda_precision_cast_elements", result.lambda.df_precision_cast_elements);
  work_field("lambda_mixed_iterations", result.lambda.df_mixed_iterations);
  work_field("lambda_strict_iterations", result.lambda.df_strict_iterations);
  output << "  \"lambda_mixed_program_hash\": "
         << std::quoted(result.lambda.mixed_program_hash ? result.lambda.mixed_program_hash : "")
         << ",\n";
  work_field("lambda_audit_arena_bytes", result.lambda.df_audit_arena_bytes);
  output << "  \"lambda_audit_schedule_hash\": "
         << std::quoted(result.lambda.audit_schedule_hash ? result.lambda.audit_schedule_hash : "")
         << ",\n";
  output << "  \"lambda_core_reuse_plan_hash\": "
         << std::quoted(result.lambda.core_reuse_plan_hash ? result.lambda.core_reuse_plan_hash
                                                           : "")
         << ",\n";
  field("lambda_reduced", result.lambda.df_auxiliary_reduction ? 1 : 0);
  work_field("lambda_preparations", result.lambda.df_preparation_calls);
  work_field("lambda_reduced_actions", result.lambda.df_reduced_actions);
  work_field("lambda_actions", result.lambda.operator_actions);
  work_field("lambda_iterations", result.lambda.iterations);
  work_field("lambda_auxiliary_visits", result.lambda.df_auxiliary_slices);
  work_field("lambda_kernels", result.lambda.df_generated_kernels);
  work_field("lambda_capacity", result.lambda.numeric_capacity_bytes);
  work_field("lambda_device_bytes", result.lambda.owned_device_bytes);
  work_field("lambda_h2d_bytes", result.lambda.h2d_bytes);
  work_field("lambda_d2h_bytes", result.lambda.d2h_bytes);
  field("global_stability_certified", result.orbital.global_stability_certified ? 1 : 0);
  output << "  \"forces\": [";
  for (std::size_t i = 0; i < result.forces.size(); ++i) {
    if (i) output << ',';
    output << result.forces[i];
  }
  output << "]\n}\n";
}

inline void publish_df_ccsdt_force_report(const std::string& path,
                                          const methods::detail::DFCCSDTResult& result,
                                          const DFCCSDTReportOptions& options) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open completed force output");
  write_df_ccsdt_force_report(output, result, options);
  output.close();
  if (!output) throw std::runtime_error("failed publishing completed force output");
}

}  // namespace generativeqc::benchmarks

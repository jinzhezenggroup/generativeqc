// Complete cold DF-CCSD(T) energy/force benchmark with explicit schedule selectors.
// The input contains no orbitals, Fock matrix, factors or amplitudes from an oracle.
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cc/df_triples.hpp"
#include "methods/df_ccsdt_force.hpp"
#include "molecule/basis.hpp"
#include "runtime/execution_context.hpp"

namespace {

void read_shells(std::istream& input, generativeqc::core::System& system, std::size_t count) {
  system.shells.resize(count);
  for (auto& shell : system.shells) {
    std::size_t primitives = 0;
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    if (!input || !primitives || primitives > 1000)
      throw std::invalid_argument("invalid molecular probe shell");
    shell.primitives.resize(primitives);
    for (auto& p : shell.primitives) input >> p.exponent >> p.coefficient;
  }
  if (!input) throw std::invalid_argument("truncated molecular probe basis");
  std::string detail;
  if (generativeqc::molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 27)
      throw std::invalid_argument(
          "usage: df-force-endpoint INPUT OUTPUT_JSON REDUCTION_0_OR_1 [MATRIX_0_OR_1 "
          "[FORCES_0_OR_1 [LAMBDA_MATRIX_0_OR_1 [Q_BATCH_LIMIT [DIIS_HISTORY [CCSD_Q_BATCH_LIMIT "
          "[ORBITAL_SCHWARZ "
          "[PROFILE_JK_0_OR_1 [NUCLEAR_0_LEGACY_1_CANONICAL_2_SYMMETRIC "
          "[DERIVED_DENOMINATORS_0_OR_1 [Z_TRUE_RESIDUAL_INTERVAL [Z_DF_PRECONDITIONER_0_OR_1 "
          "[Z_RECYCLE_REPEAT_0_OR_1 [PACKED_DIIS_0_OR_1 [RESIDENT_JK_MAXIMUM_BYTES_OR_AUTO "
          "[PARALLEL_GAP_0_OR_1 [REQUEST_GAP_0_OR_1 [REFERENCE_TOLERANCE_OR_AUTO "
          "[FUSED_SCALAR_RESPONSE_0_OR_1 [TRIPLES_W_FP32_0_OR_1 "
          "[LAMBDA_TRUE_RESIDUAL_INTERVAL [LAMBDA_CORE_REUSE_0_OR_1 "
          "[LAMBDA_AUDIT_MATRIX_0_OR_1]]]]]]]]]]]]]]]]]]]]]]]");
    const bool reduction = std::string(argv[3]) == "1";
    if (!reduction && std::string(argv[3]) != "0")
      throw std::invalid_argument("invalid schedule selector");
    const auto selector = [&](int index) {
      if (argc <= index) return true;
      const std::string value(argv[index]);
      if (value != "0" && value != "1") throw std::invalid_argument("invalid endpoint selector");
      return value == "1";
    };
    const bool matrix = selector(4), forces = selector(5), lambda_matrix = selector(6);
    const auto unsigned_argument = [&](int index, unsigned long long fallback) {
      if (argc <= index) return fallback;
      const std::string token = argv[index];
      if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("invalid unsigned endpoint argument");
      return std::stoull(token);
    };
    const std::size_t batch_limit = unsigned_argument(7, 8);
    // Preserve the established DIIS and CCSD batch slots; append response controls.
    const auto diis_history = unsigned_argument(8, 6);
    if (diis_history == 1 || diis_history > 20)
      throw std::invalid_argument("invalid endpoint DIIS history");
    const auto ccsd_batch_limit = unsigned_argument(9, 8);
    generativeqc::hf::RHFFrameResponseOptions frame_options;
    const std::string screening_argument = argc > 10 ? argv[10] : "0";
    std::size_t screening_consumed = 0;
    frame_options.orbital_screening_tolerance = std::stod(screening_argument, &screening_consumed);
    if (screening_consumed != screening_argument.size() ||
        !std::isfinite(frame_options.orbital_screening_tolerance) ||
        frame_options.orbital_screening_tolerance < 0.0)
      throw std::invalid_argument("invalid orbital screening threshold");
    frame_options.profile_jk = argc > 11 && selector(11);
    const std::string nuclear_selector = argc > 12 ? argv[12] : "2";
    if (nuclear_selector != "0" && nuclear_selector != "1" && nuclear_selector != "2")
      throw std::invalid_argument("invalid nuclear response selector");
    const auto nuclear_schedule = nuclear_selector[0] - '0';
    frame_options.bilinear_derivative = nuclear_schedule == 1;
    frame_options.symmetric_polarization = nuclear_schedule == 2;
    const bool derived_denominators = selector(13);
    if (argc > 14) frame_options.gmres.true_residual_every = unsigned_argument(14, 0);
    if (argc > 15) frame_options.df_preconditioning = selector(15);
    generativeqc::hf::RHFFrameResponseRecycle recycling;
    const bool recycle_repeat = argc > 16 && selector(16);
    if (recycle_repeat) frame_options.recycling = &recycling;
    const bool packed_diis = argc > 17 && selector(17);
    if (argc > 18 && std::string(argv[18]) != "auto")
      frame_options.resident_jk_maximum_bytes = unsigned_argument(18, 0);
    const bool parallel_gap_reduction = argc <= 19 || selector(19);
    const bool request_triples_gap_cotangents = argc > 20 && selector(20);
    const bool fused_triples_scalar_response = argc > 22 && selector(22);
    const bool lambda_core_reuse = selector(25);
    const bool lambda_audit_matrix = selector(26);
    const bool triples_w_fp32 = argc > 23 && selector(23);
    // The complete DF force owner defaults to amortized FP64 Lambda checks.
    // Explicit interval 1 retains the historical per-iteration control.
    const auto lambda_true_residual_interval = unsigned_argument(24, 30);
    if (!lambda_true_residual_interval)
      throw std::invalid_argument("Lambda true residual interval must be positive");
    generativeqc::runtime::PrecisionDirective admitted_triples_w;
    if (triples_w_fp32) {
      admitted_triples_w = {
          generativeqc::runtime::PrecisionDtype::Fp32, generativeqc::runtime::PrecisionDtype::Fp32,
          generativeqc::runtime::PrecisionDtype::Fp32, "issue1764/df-triples-w-fp32-candidate-v1"};
    }
    double reference_energy_tolerance = 1e-12, reference_density_tolerance = 1e-11;
    if (argc > 21 && std::string(argv[21]) != "auto") {
      const std::string token(argv[21]);
      std::size_t consumed = 0;
      const double tolerance = std::stod(token, &consumed);
      if (consumed != token.size() || !std::isfinite(tolerance) || tolerance <= 0.0 ||
          tolerance > reference_energy_tolerance)
        throw std::invalid_argument("invalid reference tolerance: require 0 < value <= 1e-12");
      // This benchmark-only control tightens RHF, never the CC/response acceptance gates.
      reference_energy_tolerance = reference_density_tolerance = tolerance;
    }
    std::ifstream input(argv[1]);
    std::size_t atoms = 0, orbital_shells = 0, auxiliary_shells = 0, budget = 0;
    input >> atoms >> orbital_shells >> auxiliary_shells >> budget;
    if (!input || !atoms || atoms > 1000 || !orbital_shells || !auxiliary_shells || !budget)
      throw std::invalid_argument("invalid molecular probe dimensions");
    generativeqc::core::System orbital, auxiliary;
    orbital.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    orbital.atoms.resize(atoms);
    for (auto& atom : orbital.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    auxiliary = orbital;
    read_shells(input, orbital, orbital_shells);
    read_shells(input, auxiliary, auxiliary_shells);

    generativeqc::core::ContextState context;
    context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
    generativeqc::runtime::ExecutionContext execution(context);
    generativeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = GENERATIVEQC_METHOD_RCCSD;
    descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
    descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor.max_iterations = 150;
    descriptor.energy_tolerance = reference_energy_tolerance;
    descriptor.density_tolerance = reference_density_tolerance;
    descriptor.ccsd_max_iterations = 150;
    descriptor.ccsd_diis_history = static_cast<unsigned>(diis_history);
    descriptor.ccsd_energy_tolerance = 1e-12;
    descriptor.ccsd_residual_tolerance = 1e-10;
    descriptor.correlation_memory_budget_bytes = budget;
    std::cerr << "Starting native molecular RHF/source/CCSD: N="
              << generativeqc::molecule::ao_count(orbital)
              << " Q=" << generativeqc::molecule::ao_count(auxiliary) << std::endl;
    double previous_endpoint_seconds = 0;
    for (int repetition = 0; repetition < (recycle_repeat ? 2 : 1); ++repetition) {
      const auto result = generativeqc::methods::detail::run_df_ccsdt_native(
          execution, orbital, auxiliary, descriptor, forces, true, reduction, matrix, lambda_matrix,
          batch_limit, ccsd_batch_limit, frame_options, derived_denominators, packed_diis,
          parallel_gap_reduction, request_triples_gap_cotangents, fused_triples_scalar_response,
          admitted_triples_w, lambda_true_residual_interval, lambda_core_reuse,
          lambda_audit_matrix);
      std::ofstream output(std::string(argv[2]) + (repetition ? ".warm.json" : ""));
      if (!output) throw std::runtime_error("cannot open completed force output");
      output << std::setprecision(17) << "{\n";
      const auto field = [&](const char* name, auto value) {
        output << "  " << std::quoted(name) << ": " << value << ",\n";
      };
      // A discarded attempt can include any completed phase. Final-attempt
      // counters, phase times and peaks cannot reconstruct total attempted work.
      const auto work_field = [&](const char* name, auto value) {
        if (result.recycling_discarded_primal_attempt ||
            result.orbital.resident_jk_discarded_attempt)
          field(name, "null");
        else
          field(name, value);
      };
      field("nbf", generativeqc::molecule::ao_count(orbital));
      field("naux", generativeqc::molecule::ao_count(auxiliary));
      // Distinguish phases absent by request from measured zero-cost phases.
      field("forces_requested", forces ? 1 : 0);
      field("lambda_reduction_requested", reduction ? 1 : 0);
      field("matrix_gemm_requested", matrix ? 1 : 0);
      field("total_energy", result.energy);
      field("reference_energy", result.reference_energy);
      field("reference_energy_tolerance", descriptor.energy_tolerance);
      field("reference_density_tolerance", descriptor.density_tolerance);
      field("reference_energy_change", result.reference_energy_change);
      field("reference_density_rms", result.reference_density_rms);
      work_field("reference_iterations", result.reference_iterations);
      field("correlation_energy", result.correlation_energy);
      field("triples_energy", result.triples_energy);
      field("lambda_residual", result.lambda.independent_residual_norm);
      work_field("lambda_true_residual_interval", lambda_true_residual_interval);
      field("z_residual", result.orbital.orbital_residual);
      field("stationarity", result.orbital.maximum_stationarity);
      work_field("numeric_capacity_bytes", result.numeric_capacity_bytes);
      field("native_seconds", result.total_seconds);
      field("resident_jk_discarded_attempt", result.orbital.resident_jk_discarded_attempt ? 1 : 0);
      field("resident_jk_retry_seconds", result.orbital.resident_jk_retry_seconds);
      work_field("reference_seconds", result.primal.reference_seconds);
      work_field("source_seconds", result.primal.problem_seconds);
      work_field("ccsd_seconds", result.primal.solver_seconds);
      field("ccsd_matrix_gemm", result.solver.df_matrix_gemm ? 1 : 0);
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
      work_field("ccsd_evaluations", result.solver.iteration_graph_calls);
      work_field("ccsd_capacity", result.solver.numeric_capacity_bytes);
      work_field("ccsd_device_bytes", result.solver.owned_device_bytes);
      work_field("ccsd_setup_h2d_bytes", result.solver.setup_h2d_bytes);
      field("denominator_identity", result.solver.denominator_identity);
      work_field("derived_d2_iteration_evaluations",
                 result.solver.derived_d2_iteration_evaluations);
      work_field("ccsd_iterations", result.solver.iterations);
      field("ccsd_replay_r1_max", result.solver.replay_r1_max);
      field("ccsd_replay_r2_max", result.solver.replay_r2_max);
      field("ccsd_diis_history", diis_history);
      work_field("ccsd_diis_seconds", result.solver.diis_seconds);
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
      work_field("z_preconditioner_actions",
                 result.orbital.orbital_response.preconditioner_actions);
      field("z_true_residual_interval", frame_options.gmres.true_residual_every);
      field("z_df_preconditioned", result.orbital.df_preconditioned ? 1 : 0);
      field("z_preconditioner_fallback", result.orbital.preconditioner_fallback ? 1 : 0);
      work_field("z_preconditioner_setup_seconds", result.orbital.preconditioner_setup_seconds);
      work_field("z_preconditioner_capacity_bytes", result.orbital.preconditioner_capacity_bytes);
      work_field("z_preconditioner_planned_contraction_terms",
                 result.orbital.preconditioner_contraction_terms);
      field("z_recycled_guess", result.orbital.recycled_guess ? 1 : 0);
      field("z_recycle_published", result.orbital.recycle_published ? 1 : 0);
      work_field("z_recycle_capacity_bytes", result.orbital.recycle_capacity_bytes);
      field("previous_endpoint_seconds", previous_endpoint_seconds);
      field("recycling_discarded_primal_attempt",
            result.recycling_discarded_primal_attempt ? 1 : 0);
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
      field("triples_w_fp32_requested", triples_w_fp32);
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
      field("triples_scalar_fusion_resource_fallback",
            result.triples_scalar_fusion.resource_fallback);
      output << "  \"triples_scalar_fusion_schedule\": "
             << std::quoted(result.triples_scalar_fusion.schedule) << ",\n";
      work_field("triples_scalar_fusion_tiles", result.triples_scalar_fusion.tiles);
      work_field("triples_scalar_fusion_kernels", result.triples_scalar_fusion.kernels);
      work_field("triples_scalar_fusion_avoided_kernels",
                 result.triples_scalar_fusion.avoided_kernels);
      work_field("triples_scalar_fusion_workspace_bytes",
                 result.triples_scalar_fusion.workspace_bytes);
      work_field("triples_scalar_fusion_value_reads", result.triples_scalar_fusion.value_reads);
      work_field("triples_scalar_fusion_value_writes", result.triples_scalar_fusion.value_writes);
      work_field("triples_scalar_fusion_arithmetic_ops",
                 result.triples_scalar_fusion.arithmetic_ops);
      work_field("fock_response_work", result.triples_fock.contraction_summands);
      work_field("lambda_work", result.lambda.df_contraction_terms);
      field("lambda_matrix_gemm", result.lambda.df_matrix_gemm);
      field("lambda_batch_size", result.lambda.df_auxiliary_batch_size);
      work_field("lambda_batches", result.lambda.df_auxiliary_batches);
      work_field("lambda_gemm_calls", result.lambda.df_gemm_calls);
      work_field("lambda_gemm_summands", result.lambda.df_gemm_summands);
      work_field("lambda_packing_output_bytes", result.lambda.df_packing_output_bytes);
      work_field("lambda_provider_allowance", result.lambda.df_provider_allowance_bytes);
      output << "  \"lambda_shared_program_hash\": "
             << std::quoted(result.lambda.shared_program_hash ? result.lambda.shared_program_hash
                                                              : "")
             << ",\n  \"lambda_independent_program_hash\": "
             << std::quoted(result.lambda.independent_program_hash
                                ? result.lambda.independent_program_hash
                                : "")
             << ",\n";
      field("lambda_core_reuse_requested", lambda_core_reuse);
      field("lambda_core_reuse", result.lambda.df_core_reuse);
      work_field("lambda_core_reuse_bytes", result.lambda.df_core_reuse_bytes);
      work_field("lambda_core_reuse_preparations", result.lambda.df_core_reuse_preparations);
      work_field("lambda_core_reuse_actions", result.lambda.df_core_reuse_actions);
      field("lambda_audit_matrix_requested", lambda_audit_matrix);
      field("lambda_audit_matrix", result.lambda.df_audit_matrix_gemm);
      work_field("lambda_audit_arena_bytes", result.lambda.df_audit_arena_bytes);
      output << "  \"lambda_audit_schedule_hash\": "
             << std::quoted(result.lambda.audit_schedule_hash ? result.lambda.audit_schedule_hash
                                                              : "")
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
      output.close();
      if (!output) throw std::runtime_error("failed publishing completed force output");
      std::cout << "Complete " << (forces ? "force" : "energy") << " endpoint in "
                << result.total_seconds << " seconds\n";
      previous_endpoint_seconds += result.total_seconds;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}

// Complete cold DF-CCSD(T) energy/force benchmark with explicit schedule selectors.
// The input contains no orbitals, Fock matrix, factors or amplitudes from an oracle.
#include <chrono>
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
    if (argc < 4 || argc > 14)
      throw std::invalid_argument(
          "usage: df-force-endpoint INPUT OUTPUT_JSON REDUCTION_0_OR_1 [MATRIX_0_OR_1 "
          "[FORCES_0_OR_1 [LAMBDA_MATRIX_0_OR_1 [Q_BATCH_LIMIT [DIIS_HISTORY "
          "[CCSD_Q_BATCH_LIMIT [DERIVED_DENOMINATORS_0_OR_1 [Z_TRUE_RESIDUAL_INTERVAL "
          "[Z_DF_PRECONDITIONER_0_OR_1 [Z_RECYCLE_REPEAT_0_OR_1]]]]]]]]]]");
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
    const std::size_t batch_limit = argc > 7 ? std::stoull(argv[7]) : 8;
    const auto diis_history = argc > 8 ? std::stoul(argv[8]) : 6;
    const auto ccsd_batch_limit = argc > 9 ? std::stoull(argv[9]) : 8;
    const bool derived_denominators = selector(10);
    generativeqc::hf::RHFFrameResponseOptions response_options;
    if (argc > 11) response_options.gmres.true_residual_every = std::stoull(argv[11]);
    if (argc > 12) response_options.df_preconditioning = selector(12);
    generativeqc::hf::RHFFrameResponseRecycle recycling;
    const bool recycle_repeat = argc > 13 && selector(13);
    if (recycle_repeat) response_options.recycling = &recycling;
    if (diis_history == 1 || diis_history > 20)
      throw std::invalid_argument("invalid endpoint DIIS history");
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
    descriptor.energy_tolerance = 1e-12;
    descriptor.density_tolerance = 1e-11;
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
          batch_limit, ccsd_batch_limit, derived_denominators, &response_options);
      std::ofstream output(std::string(argv[2]) + (repetition ? ".warm.json" : ""));
      if (!output) throw std::runtime_error("cannot open completed force output");
      output << std::setprecision(17) << "{\n";
      const auto field = [&](const char* name, auto value) {
        output << "  " << std::quoted(name) << ": " << value << ",\n";
      };
      // A discarded attempt can include any completed phase. Final-attempt
      // counters, phase times and peaks cannot reconstruct total attempted work.
      const auto work_field = [&](const char* name, auto value) {
        if (result.recycling_discarded_primal_attempt)
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
      field("correlation_energy", result.correlation_energy);
      field("triples_energy", result.triples_energy);
      field("lambda_residual", result.lambda.independent_residual_norm);
      field("z_residual", result.orbital.orbital_residual);
      field("stationarity", result.orbital.maximum_stationarity);
      work_field("numeric_capacity_bytes", result.numeric_capacity_bytes);
      field("native_seconds", result.total_seconds);
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
      work_field("z_iterations", result.orbital.orbital_response.iterations);
      work_field("z_operator_actions", result.orbital.orbital_response.operator_actions);
      work_field("z_preconditioner_actions",
                 result.orbital.orbital_response.preconditioner_actions);
      field("z_true_residual_interval", response_options.gmres.true_residual_every);
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
      field("hessian_elements", result.orbital.explicit_hessian_elements);
      work_field("source_weight_values", result.source_weight_values);
      work_field("metric_weight_values", result.metric_weight_values);
      work_field("triples_work", result.triples.contraction_summands);
      work_field("fock_response_work", result.triples_fock.contraction_summands);
      work_field("lambda_work", result.lambda.df_contraction_terms);
      field("lambda_matrix_gemm", result.lambda.df_matrix_gemm);
      field("lambda_batch_size", result.lambda.df_auxiliary_batch_size);
      work_field("lambda_batches", result.lambda.df_auxiliary_batches);
      work_field("lambda_gemm_calls", result.lambda.df_gemm_calls);
      work_field("lambda_gemm_summands", result.lambda.df_gemm_summands);
      work_field("lambda_packing_output_bytes", result.lambda.df_packing_output_bytes);
      work_field("lambda_provider_allowance", result.lambda.df_provider_allowance_bytes);
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

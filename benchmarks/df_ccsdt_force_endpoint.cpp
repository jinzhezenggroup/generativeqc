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
    if (argc < 4 || argc > 14)
      throw std::invalid_argument(
          "usage: df-force-endpoint INPUT OUTPUT_JSON REDUCTION_0_OR_1 [MATRIX_0_OR_1 "
          "[FORCES_0_OR_1 [LAMBDA_MATRIX_0_OR_1 [Q_BATCH_LIMIT [DIIS_HISTORY [CCSD_Q_BATCH_LIMIT "
          "[ORBITAL_SCHWARZ "
          "[PROFILE_JK_0_OR_1 [NUCLEAR_0_LEGACY_1_CANONICAL_2_SYMMETRIC "
          "[DERIVED_DENOMINATORS_0_OR_1]]]]]]]]]]");
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
    const auto result = generativeqc::methods::detail::run_df_ccsdt_native(
        execution, orbital, auxiliary, descriptor, forces, true, reduction, matrix, lambda_matrix,
        batch_limit, ccsd_batch_limit, frame_options, derived_denominators);
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot open completed force output");
    output << std::setprecision(17) << "{\n";
    const auto field = [&](const char* name, auto value) {
      output << "  " << std::quoted(name) << ": " << value << ",\n";
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
    field("numeric_capacity_bytes", result.numeric_capacity_bytes);
    field("native_seconds", result.total_seconds);
    field("reference_seconds", result.primal.reference_seconds);
    field("source_seconds", result.primal.problem_seconds);
    field("ccsd_seconds", result.primal.solver_seconds);
    field("ccsd_matrix_gemm", result.solver.df_matrix_gemm ? 1 : 0);
    field("ccsd_gemm_calls", result.solver.df_gemm_calls);
    field("ccsd_gemm_summands", result.solver.df_gemm_summands);
    field("ccsd_packing_bytes", result.solver.df_packing_bytes);
    field("ccsd_provider_capacity", result.solver.df_provider_capacity_bytes);
    field("ccsd_q_batch_size", result.solver.df_auxiliary_batch_size);
    field("ccsd_q_tiles", result.solver.df_auxiliary_tiles);
    field("ccsd_q_slices", result.solver.df_auxiliary_slices);
    field("ccsd_q_operations", result.solver.df_virtual_operations);
    field("ccsd_accumulation_calls", result.solver.df_accumulation_calls);
    field("ccsd_accumulation_bytes", result.solver.df_accumulation_bytes);
    field("ccsd_contraction_terms", result.solver.df_contraction_terms);
    field("ccsd_evaluations", result.solver.iteration_graph_calls);
    field("ccsd_capacity", result.solver.numeric_capacity_bytes);
    field("ccsd_device_bytes", result.solver.owned_device_bytes);
    field("ccsd_setup_h2d_bytes", result.solver.setup_h2d_bytes);
    field("denominator_identity", result.solver.denominator_identity);
    field("derived_d2_iteration_evaluations", result.solver.derived_d2_iteration_evaluations);
    field("ccsd_iterations", result.solver.iterations);
    field("ccsd_replay_r1_max", result.solver.replay_r1_max);
    field("ccsd_replay_r2_max", result.solver.replay_r2_max);
    field("ccsd_diis_history", diis_history);
    field("ccsd_diis_seconds", result.solver.diis_seconds);
    field("ccsd_diis_restarts", result.solver.diis_restarts);
    field("ccsd_diis_gram_calls", result.solver.diis_gram_calls);
    field("ccsd_diis_coefficient_calls", result.solver.diis_coefficient_calls);
    field("ccsd_diis_combine_calls", result.solver.diis_combine_calls);
    field("ccsd_diis_insert_bytes", result.solver.diis_history_insert_bytes);
    field("ccsd_diis_shift_bytes", result.solver.diis_history_shift_bytes);
    field("ccsd_diis_dot_terms", result.solver.diis_residual_dot_terms);
    field("ccsd_diis_gram_updates", result.solver.diis_gram_updates);
    field("ccsd_diis_combine_terms", result.solver.diis_combine_terms);
    field("triples_seconds", result.triples_seconds);
    field("lambda_seconds", result.lambda_seconds);
    field("source_response_seconds", result.source_response_seconds);
    field("orbital_seconds", result.orbital_seconds);
    field("jk_actions", result.orbital.jk_actions);
    field("orbital_setup_seconds", result.orbital.setup_seconds);
    field("orbital_reference_audit_seconds", result.orbital.reference_audit_seconds);
    field("orbital_weights_seconds", result.orbital.weights_seconds);
    field("orbital_solve_seconds", result.orbital.solve_seconds);
    field("orbital_independent_audit_seconds", result.orbital.independent_audit_seconds);
    field("orbital_one_electron_seconds", result.orbital.one_electron_seconds);
    field("orbital_two_electron_seconds", result.orbital.two_electron_seconds);
    field("orbital_bilinear_derivative_used", result.orbital.bilinear_derivative_used);
    field("orbital_symmetric_polarization_used", result.orbital.symmetric_polarization_used);
    field("orbital_derivative_census_measured", result.orbital.derivative_census_measured);
    field("orbital_derivative_quartet_visits", result.orbital.derivative_quartet_visits);
    field("orbital_derivative_jet_evaluations", result.orbital.derivative_jet_evaluations);
    field("orbital_jk_seconds", result.orbital.jk_seconds);
    field("orbital_screened_jk_seconds", result.orbital.screened_jk_seconds);
    field("orbital_screened_residual", result.orbital.screened_residual);
    field("orbital_requested_screening", result.orbital.requested_screening);
    field("orbital_applied_screening", result.orbital.applied_screening);
    field("orbital_screened_jk_actions", result.orbital.screened_jk_actions);
    field("orbital_jk_census_actions", result.orbital.jk_census_actions);
    field("orbital_exact_refinements", result.orbital.exact_refinements);
    field("orbital_screened_iterations", result.orbital.screened_iterations);
    field("orbital_screened_operator_actions", result.orbital.screened_operator_actions);
    field("orbital_jk_quartet_visits", result.orbital.jk_quartet_visits);
    field("orbital_jk_eri_evaluations", result.orbital.jk_eri_evaluations);
    field("orbital_jk_timing_measured", result.orbital.jk_timing_measured);
    field("orbital_linear_screening_available", result.orbital.linear_screening_available);
    field("orbital_screened_converged", result.orbital.screened_converged);
    field("orbital_derivative_passes", result.orbital.derivative_passes);
    field("orbital_shell_derivative_passes", result.orbital.shell_derivative_passes);
    field("orbital_generic_derivative_passes", result.orbital.generic_derivative_passes);
    field("hessian_elements", result.orbital.explicit_hessian_elements);
    field("source_weight_values", result.source_weight_values);
    field("metric_weight_values", result.metric_weight_values);
    field("triples_work", result.triples.contraction_summands);
    field("fock_response_work", result.triples_fock.contraction_summands);
    field("lambda_work", result.lambda.df_contraction_terms);
    field("lambda_matrix_gemm", result.lambda.df_matrix_gemm);
    field("lambda_batch_size", result.lambda.df_auxiliary_batch_size);
    field("lambda_batches", result.lambda.df_auxiliary_batches);
    field("lambda_gemm_calls", result.lambda.df_gemm_calls);
    field("lambda_gemm_summands", result.lambda.df_gemm_summands);
    field("lambda_packing_output_bytes", result.lambda.df_packing_output_bytes);
    field("lambda_provider_allowance", result.lambda.df_provider_allowance_bytes);
    field("lambda_reduced", result.lambda.df_auxiliary_reduction ? 1 : 0);
    field("lambda_preparations", result.lambda.df_preparation_calls);
    field("lambda_reduced_actions", result.lambda.df_reduced_actions);
    field("lambda_actions", result.lambda.operator_actions);
    field("lambda_iterations", result.lambda.iterations);
    field("lambda_auxiliary_visits", result.lambda.df_auxiliary_slices);
    field("lambda_kernels", result.lambda.df_generated_kernels);
    field("lambda_capacity", result.lambda.numeric_capacity_bytes);
    field("lambda_device_bytes", result.lambda.owned_device_bytes);
    field("lambda_h2d_bytes", result.lambda.h2d_bytes);
    field("lambda_d2h_bytes", result.lambda.d2h_bytes);
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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}

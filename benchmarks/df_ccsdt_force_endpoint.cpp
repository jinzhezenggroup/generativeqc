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
    if (argc < 4 || argc > 8)
      throw std::invalid_argument(
          "usage: df-force-endpoint INPUT OUTPUT_JSON REDUCTION_0_OR_1 [MATRIX_0_OR_1 "
          "[FORCES_0_OR_1 [LAMBDA_MATRIX_0_OR_1 [Q_BATCH_LIMIT]]]]");
    const bool reduction = std::string(argv[3]) == "1";
    if (!reduction && std::string(argv[3]) != "0")
      throw std::invalid_argument("invalid schedule selector");
    const auto selector = [&](int index, bool fallback = true) {
      if (argc <= index) return fallback;
      const std::string value(argv[index]);
      if (value != "0" && value != "1") throw std::invalid_argument("invalid endpoint selector");
      return value == "1";
    };
    const bool matrix = selector(4), forces = selector(5), lambda_matrix = selector(6, false);
    const std::size_t batch_limit = argc > 7 ? std::stoull(argv[7]) : 8;
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
    descriptor.ccsd_diis_history = 6;
    descriptor.ccsd_energy_tolerance = 1e-12;
    descriptor.ccsd_residual_tolerance = 1e-10;
    descriptor.correlation_memory_budget_bytes = budget;
    std::cerr << "Starting native molecular RHF/source/CCSD: N="
              << generativeqc::molecule::ao_count(orbital)
              << " Q=" << generativeqc::molecule::ao_count(auxiliary) << std::endl;
    const auto result = generativeqc::methods::detail::run_df_ccsdt_native(
        execution, orbital, auxiliary, descriptor, forces, true, reduction, matrix, lambda_matrix,
        batch_limit);
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
    field("ccsd_contraction_terms", result.solver.df_contraction_terms);
    field("ccsd_evaluations", result.solver.iteration_graph_calls);
    field("ccsd_capacity", result.solver.numeric_capacity_bytes);
    field("triples_seconds", result.triples_seconds);
    field("lambda_seconds", result.lambda_seconds);
    field("source_response_seconds", result.source_response_seconds);
    field("orbital_seconds", result.orbital_seconds);
    field("jk_actions", result.orbital.jk_actions);
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

// Diagnostic only: one native cold primal, then matched complete response schedules.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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
      throw std::invalid_argument("invalid shared-primal shell");
    shell.primitives.resize(primitives);
    for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
  }
  if (!input) throw std::invalid_argument("truncated shared-primal basis");
  std::string detail;
  if (generativeqc::molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: df-gap-same-primal INPUT OUTPUT_JSON");
    std::ifstream input(argv[1]);
    std::size_t atoms = 0, orbital_shells = 0, auxiliary_shells = 0, budget = 0;
    input >> atoms >> orbital_shells >> auxiliary_shells >> budget;
    if (!input || !atoms || atoms > 1000 || !orbital_shells || !auxiliary_shells || !budget)
      throw std::invalid_argument("invalid shared-primal dimensions");
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
    descriptor.ccsd_diis_history = 8;
    descriptor.ccsd_energy_tolerance = 1e-12;
    descriptor.ccsd_residual_tolerance = 1e-10;
    descriptor.correlation_memory_budget_bytes = budget;
    generativeqc::hf::RHFFrameResponseOptions frame_options;
    frame_options.profile_jk = true;
    frame_options.gmres.true_residual_every = 30;
    const auto comparison = generativeqc::methods::detail::diagnose_df_ccsdt_gap_schedules(
        execution, orbital, auxiliary, descriptor, frame_options, 8, 8, true, true);
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot open completed shared-primal output");
    output << std::setprecision(17) << "{\n";
    const auto field = [&](const char* name, auto value) {
      output << "  " << std::quoted(name) << ": " << value << ",\n";
    };
    field("shared_primal", "true");
    field("native_primal_calls", 1);
    field("cold_endpoint_timing", "false");
    field("complete_comparison_seconds", comparison.total_seconds);
    field("cold_primal_seconds", comparison.primal_seconds);
    field("nbf", comparison.nocc + comparison.nvir);
    field("nocc", comparison.nocc);
    field("nvir", comparison.nvir);
    field("naux", comparison.naux);
    field("source_identity", comparison.source_identity);
    field("denominator_identity", comparison.denominator_identity);
    field("primal_identity", comparison.primal_identity);
    field("reference_energy", comparison.reference_energy);
    field("reference_energy_tolerance", descriptor.energy_tolerance);
    field("reference_density_tolerance", descriptor.density_tolerance);
    field("reference_energy_change", comparison.reference_energy_change);
    field("reference_density_rms", comparison.reference_density_rms);
    field("reference_iterations", comparison.reference_iterations);
    field("ccsd_iterations", comparison.solver.iterations);
    field("ccsd_evaluations", comparison.solver.iteration_graph_calls);
    field("reference_seconds", comparison.primal.reference_seconds);
    field("source_seconds", comparison.primal.problem_seconds);
    field("ccsd_seconds", comparison.primal.solver_seconds);
    field("numeric_budget_bytes", descriptor.correlation_memory_budget_bytes);
    field("retained_primal_host_bytes", comparison.retained_primal_host_bytes);
    field("retained_df_source_bytes", comparison.retained_df_source_bytes);
    field("retained_exact_source_bytes", comparison.retained_exact_source_bytes);
    field("output_bytes", comparison.output_bytes);
    field("clone_admission_bytes", comparison.clone_admission_bytes);
    output << "  \"cases\": [\n";
    const char* modes[]{"serial", "parallel", "omitted", "serial-repeat"};
    for (std::size_t index = 0; index < comparison.cases.size(); ++index) {
      const auto& snapshot = comparison.cases[index];
      output << " {\n  \"mode\": " << std::quoted(modes[index]) << ",\n";
      field("total_energy", snapshot.energy);
      field("triples_energy", snapshot.triples_energy);
      field("lambda_residual", snapshot.lambda.independent_residual_norm);
      field("z_residual", snapshot.orbital_residual);
      field("stationarity", snapshot.maximum_stationarity);
      field("clone_seconds", snapshot.clone_seconds);
      field("response_seconds", snapshot.response_seconds);
      field("triples_seconds", snapshot.triples_seconds);
      field("lambda_seconds", snapshot.lambda_seconds);
      field("source_response_seconds", snapshot.source_response_seconds);
      field("orbital_seconds", snapshot.orbital_seconds);
      field("fingerprint_seconds", snapshot.fingerprints.seconds);
      field("fingerprint_value_reads", snapshot.fingerprints.value_reads);
      field("numeric_capacity_bytes", snapshot.numeric_capacity_bytes);
      field("source_weight_values", snapshot.source_weight_values);
      field("metric_weight_values", snapshot.metric_weight_values);
      field("lambda_work", snapshot.lambda.df_contraction_terms);
      field("fock_response_work", snapshot.fock_response_work);
      field("z_iterations", snapshot.orbital_iterations);
      field("z_operator_actions", snapshot.orbital_actions);
      field("triples_gap_requested", snapshot.gap.requested ? "true" : "false");
      field("triples_gap_parallel", snapshot.gap.parallel ? "true" : "false");
      output << "  \"triples_gap_schedule\": " << std::quoted(snapshot.gap.schedule) << ",\n";
      field("triples_gap_kernels", snapshot.gap.kernels);
      field("triples_gap_workspace_bytes", snapshot.gap.workspace_bytes);
      field("triples_gap_materialized_elements", snapshot.gap.materialized_elements);
      field("triples_gap_value_reads", snapshot.gap.value_reads);
      field("triples_gap_value_writes", snapshot.gap.value_writes);
      field("triples_gap_reduction_summands", snapshot.gap.reduction_summands);
      output << "  \"response_fingerprints\": {";
      for (std::size_t stage = 0; stage < snapshot.fingerprints.identities.size(); ++stage) {
        if (stage) output << ',';
        output << std::quoted(generativeqc::methods::detail::df_gap_fingerprint_names[stage])
               << ": {\"identity\": " << snapshot.fingerprints.identities[stage]
               << ", \"elements\": " << snapshot.fingerprints.elements[stage] << '}';
      }
      output << "},\n";
      output << "  \"forces\": [";
      for (std::size_t coordinate = 0; coordinate < snapshot.forces.size(); ++coordinate) {
        if (coordinate) output << ',';
        output << snapshot.forces[coordinate];
      }
      output << "]\n }" << (index + 1 == comparison.cases.size() ? "\n" : ",\n");
    }
    output << " ]\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}

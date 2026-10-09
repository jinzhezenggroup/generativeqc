// Geometry-only DF guess -> fully converged unscreened FP64 Direct RHF experiment.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "methods/df_ccsdt_force.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/execution_context.hpp"
#include "scf/mean_field.hpp"
#include "scf/warm_state.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using generativeqc::core::System;

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/** Read the existing geometry/basis-only force-probe format, never oracle state. */
void read_shells(std::istream& input, System& system, std::size_t count) {
  system.shells.resize(count);
  for (auto& shell : system.shells) {
    std::size_t primitives{};
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    if (!input || !primitives || primitives > 1000)
      throw std::invalid_argument("invalid shell dimensions");
    shell.primitives.resize(primitives);
    for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
  }
  if (!input) throw std::invalid_argument("truncated shell data");
  std::string detail;
  if (generativeqc::molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
}

/** Strict selectors fail before CUDA setup, including malformed numeric suffixes. */
double tolerance(const char* text) {
  std::size_t consumed{};
  const std::string token(text);
  const double value = std::stod(token, &consumed);
  if (consumed != token.size() || !std::isfinite(value) || value <= 0 || value > 1e-3)
    throw std::invalid_argument("invalid preconvergence tolerance");
  return value;
}

unsigned iteration_limit(const char* text) {
  const std::string token(text);
  if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid preconvergence iteration limit");
  const auto value = std::stoul(token);
  if (!value || value > 150) throw std::invalid_argument("invalid preconvergence iteration limit");
  return static_cast<unsigned>(value);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 6 || argc > 8)
      throw std::invalid_argument(
          "usage: df-hf-preconvergence INPUT JK_SHELLS OUTPUT direct|df-direct|auto-direct "
          "hf|energy|forces "
          "[PRE_TOLERANCE [PRE_MAX_ITERATIONS]]");
    const std::string mode(argv[4]), endpoint(argv[5]);
    if (mode != "direct" && mode != "df-direct" && mode != "auto-direct")
      throw std::invalid_argument("invalid mode");
    if (mode == "auto-direct" && argc > 6)
      throw std::invalid_argument("auto-direct uses fixed policy controls");
    if (endpoint != "hf" && endpoint != "energy" && endpoint != "forces")
      throw std::invalid_argument("invalid endpoint");
    const double pre_tolerance = mode == "auto-direct" ? 1e-4
                                 : argc > 6            ? tolerance(argv[6])
                                                       : 1e-6;
    const unsigned pre_iterations = argc > 7 ? iteration_limit(argv[7]) : 50;
    std::ifstream input(argv[1]);
    std::size_t atoms{}, orbital_shells{}, correlation_shells{}, budget{};
    input >> atoms >> orbital_shells >> correlation_shells >> budget;
    if (!input || !atoms || atoms > 1000 || !orbital_shells || orbital_shells > 10000 ||
        !correlation_shells || correlation_shells > 10000 || !budget)
      throw std::invalid_argument("invalid molecular probe dimensions");
    System orbital;
    orbital.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    orbital.atoms.resize(atoms);
    for (auto& atom : orbital.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    System correlation = orbital;
    read_shells(input, orbital, orbital_shells);
    read_shells(input, correlation, correlation_shells);
    System jk = orbital;
    if (mode == "df-direct") {
      std::ifstream jk_input(argv[2]);
      std::size_t jk_shells{};
      jk_input >> jk_shells;
      if (!jk_input || !jk_shells || jk_shells > 10000)
        throw std::invalid_argument("invalid JK shell dimensions");
      read_shells(jk_input, jk, jk_shells);
    }
    std::ofstream output(argv[3]);
    if (!output) throw std::invalid_argument("cannot open output");

    generativeqc::core::ContextState context;
    context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
    const auto start = Clock::now();
    generativeqc::runtime::ExecutionContext execution(context);
    generativeqc::scf::ScfOptions exact_options;
    exact_options.max_iterations = 150;
    exact_options.energy_tolerance = 1e-12;
    exact_options.density_tolerance = 1e-11;
    exact_options.screening_tolerance = 0;
    exact_options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    exact_options.compute_forces = false;
    exact_options.export_physical_reference = true;
    exact_options.reference_memory_budget_bytes = budget;
    std::vector<double> seed;
    double pre_seconds{}, pre_energy{}, pre_density_rms{}, pre_energy_change{};
    std::size_t native_jk_functions{};
    unsigned pre_cycles{};
    bool pre_converged{}, pre_fallback{};
    bool pre_work_counters_complete{true};
    generativeqc::methods::detail::DFHFGuess automatic_guess;
    if (mode == "auto-direct" && endpoint == "hf") {
      generativeqc_method_descriptor descriptor{};
      descriptor.correlation_memory_budget_bytes = budget;
      automatic_guess = generativeqc::methods::detail::prepare_df_hf_guess(
          orbital, correlation, descriptor, 0, execution.device_id());
      seed = std::move(automatic_guess.density);
      pre_seconds = automatic_guess.seconds;
      pre_cycles = automatic_guess.iterations;
      pre_work_counters_complete = automatic_guess.work_counters_complete;
      native_jk_functions = automatic_guess.auxiliary_functions;
      pre_converged =
          automatic_guess.outcome == generativeqc::methods::detail::DFHFGuessOutcome::Used;
      pre_fallback =
          automatic_guess.outcome == generativeqc::methods::detail::DFHFGuessOutcome::Failed;
    }
    if (mode == "df-direct") {
      auto pre_options = exact_options;
      pre_options.export_physical_reference = false;
      pre_options.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
      pre_options.density_fitting_host_retry = false;
      pre_options.density_fitting_memory_budget_bytes = budget;
      pre_options.energy_tolerance = pre_tolerance;
      pre_options.density_tolerance = pre_tolerance;
      pre_options.max_iterations = pre_iterations;
      const auto pre_start = Clock::now();
      try {
        auto pre = generativeqc::scf::run_rhf_density_fitting_cuda(orbital, jk, pre_options,
                                                                   execution.device_id());
        pre_cycles = pre.iterations;
        pre_energy = pre.energy;
        pre_energy_change = pre.energy_change;
        pre_density_rms = pre.density_rms;
        pre_converged = pre.converged;
        if (pre.converged) {
          generativeqc::scf::validate_hf_warm_density(orbital, GENERATIVEQC_METHOD_RHF,
                                                      pre.density);
          seed = std::move(pre.density);
        } else {
          pre_fallback = true;
        }
      } catch (const std::exception& error) {
        pre_fallback = true;
        pre_work_counters_complete = false;
        std::cerr << "DF preconvergence refused: " << error.what() << '\n';
      }
      pre_seconds = seconds(pre_start);
    }

    generativeqc::methods::detail::DFCCSDTReferenceExperiment experiment;
    experiment.disable_preconvergence = mode != "auto-direct";
    const auto seed_bytes = generativeqc::posthf::checked_mul(seed.capacity(), sizeof(double));
    if (seed_bytes >= budget) {
      seed = std::vector<double>{};
      pre_fallback = true;
    } else {
      exact_options.reference_memory_budget_bytes = budget - seed_bytes;
    }
    experiment.initial_density = seed.empty() ? nullptr : &seed;
    double direct_seconds{}, native_seconds{}, energy{}, energy_change{}, density_rms{};
    double ccsd_seconds{}, triples_seconds{}, lambda_seconds{}, orbital_seconds{};
    double lambda_residual{}, z_residual{}, stationarity{};
    bool discarded_primal_attempt{}, discarded_response_attempt{};
    unsigned direct_cycles{};
    std::vector<double> forces;
    if (endpoint == "hf") {
      const auto direct_start = Clock::now();
      auto direct = [&] {
        try {
          return generativeqc::scf::run_rhf_cuda(orbital, exact_options, execution.device_id(),
                                                 experiment.initial_density);
        } catch (...) {
          if (!experiment.initial_density) throw;
          experiment.seed_fallback = true;
          return generativeqc::scf::run_rhf_cuda(orbital, exact_options, execution.device_id());
        }
      }();
      if ((!direct.converged || !direct.reference) && experiment.initial_density) {
        experiment.seed_fallback = true;
        direct = generativeqc::scf::run_rhf_cuda(orbital, exact_options, execution.device_id());
      }
      direct_seconds = seconds(direct_start);
      if (!direct.converged || !direct.reference)
        throw std::runtime_error("Direct RHF not converged");
      experiment.reference = direct.reference;
      experiment.work = direct.precision;
      direct_cycles = direct.iterations;
      energy = direct.energy;
      energy_change = direct.energy_change;
      density_rms = direct.density_rms;
    } else {
      generativeqc_method_descriptor descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
      descriptor.method = GENERATIVEQC_METHOD_RCCSD;
      descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
      descriptor.max_iterations = exact_options.max_iterations;
      descriptor.energy_tolerance = exact_options.energy_tolerance;
      descriptor.density_tolerance = exact_options.density_tolerance;
      descriptor.ccsd_max_iterations = 150;
      descriptor.ccsd_diis_history = 8;
      descriptor.ccsd_energy_tolerance = 1e-12;
      descriptor.ccsd_residual_tolerance = 1e-10;
      descriptor.correlation_memory_budget_bytes = budget;
      auto result = generativeqc::methods::detail::run_df_ccsdt_native(
          execution, orbital, correlation, descriptor, endpoint == "forces", true, true, true, true,
          8, 8, {}, true, true, true, false, false, {}, 30, true, true, &experiment);
      direct_seconds = result.primal.reference_seconds;
      if (mode == "auto-direct") {
        automatic_guess = std::move(result.reference_guess);
        pre_seconds = automatic_guess.seconds;
        direct_seconds -= pre_seconds;
        pre_cycles = automatic_guess.iterations;
        pre_work_counters_complete = automatic_guess.work_counters_complete;
        native_jk_functions = automatic_guess.auxiliary_functions;
        pre_converged =
            automatic_guess.outcome == generativeqc::methods::detail::DFHFGuessOutcome::Used;
        pre_fallback =
            automatic_guess.outcome == generativeqc::methods::detail::DFHFGuessOutcome::Failed;
      }
      native_seconds = result.total_seconds;
      direct_cycles = result.reference_iterations;
      energy = result.energy;
      energy_change = result.reference_energy_change;
      density_rms = result.reference_density_rms;
      forces = result.forces;
      ccsd_seconds = result.primal.solver_seconds;
      triples_seconds = result.triples_seconds;
      lambda_seconds = result.lambda_seconds;
      orbital_seconds = result.orbital_seconds;
      lambda_residual = result.lambda.independent_residual_norm;
      z_residual = result.orbital.orbital_residual;
      stationarity = result.orbital.maximum_stationarity;
      discarded_primal_attempt = result.recycling_discarded_primal_attempt;
      discarded_response_attempt = result.orbital.resident_jk_discarded_attempt;
    }
    const double endpoint_seconds = seconds(start);
    const auto& reference = *experiment.reference;
    double orthogonality_max{};
    const auto nbf = reference.nbf;
    std::vector<double> overlap_coefficients(nbf * nbf);
    for (std::size_t row = 0; row < nbf; ++row)
      for (std::size_t col = 0; col < nbf; ++col)
        for (std::size_t inner = 0; inner < nbf; ++inner)
          overlap_coefficients[row * nbf + col] +=
              reference.overlap[row * nbf + inner] * reference.coefficients[inner * nbf + col];
    for (std::size_t row = 0; row < nbf; ++row)
      for (std::size_t col = 0; col < nbf; ++col) {
        double value = row == col ? -1 : 0;
        for (std::size_t inner = 0; inner < nbf; ++inner)
          value +=
              reference.coefficients[inner * nbf + row] * overlap_coefficients[inner * nbf + col];
        orthogonality_max = std::max(orthogonality_max, std::abs(value));
      }
    output << std::setprecision(17) << "{\n";
    const auto field = [&](const char* name, auto value) {
      output << std::quoted(name) << ": ";
      // A refused one-cycle DF solve can report an infinite initial energy change.
      if constexpr (std::is_floating_point_v<decltype(value)>) {
        if (std::isfinite(value))
          output << value;
        else
          output << "null";
      } else {
        output << value;
      }
      output << ",\n";
    };
    const auto array = [&](const char* name, const std::vector<double>& values) {
      output << std::quoted(name) << ": [";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
      }
      output << "],\n";
    };
    field("mode", std::quoted(mode == "auto-direct" ? "df-direct" : mode));
    field("reference_policy", std::quoted(mode));
    field("endpoint", std::quoted(endpoint));
    field("nbf", nbf);
    field("naux_correlation", generativeqc::molecule::ao_count(correlation));
    field("naux_jk", mode == "auto-direct" ? native_jk_functions
                     : mode == "df-direct" ? generativeqc::molecule::ao_count(jk)
                                           : 0);
    field("pre_tolerance", pre_tolerance);
    field("pre_iterations", pre_work_counters_complete ? std::to_string(pre_cycles) : "null");
    field("pre_work_counters_complete", pre_work_counters_complete ? "true" : "false");
    field("pre_policy_outcome", static_cast<unsigned>(automatic_guess.outcome));
    field("pre_cartesian_functions", automatic_guess.cartesian_functions);
    field("pre_work_amortization_ratio", automatic_guess.work_amortization_ratio);
    field("pre_preparation_peak_bytes", automatic_guess.preparation_peak_bytes);
    field("pre_value_budget_bytes", automatic_guess.value_budget_bytes);
    field("df_fock_builds", "null");
    field("pre_converged", pre_converged ? "true" : "false");
    field("pre_fallback", pre_fallback ? "true" : "false");
    field("seed_fallback", experiment.seed_fallback ? "true" : "false");
    field("pre_energy", pre_energy);
    field("pre_energy_change", pre_energy_change);
    field("pre_density_rms", pre_density_rms);
    field("pre_seconds", pre_seconds);
    field("direct_seconds", direct_seconds);
    field("rhf_seconds", pre_seconds + direct_seconds);
    field("direct_iterations", direct_cycles);
    field("direct_fock_builds",
          experiment.seed_fallback || !experiment.work.operator_work_counters_valid ||
                  experiment.work.execution_retries || discarded_primal_attempt
              ? "null"
              : std::to_string(experiment.work.strict_stage_fock_builds +
                               experiment.work.post_scf_fock_builds));
    field("direct_work_counters_valid", experiment.work.operator_work_counters_valid);
    field("direct_execution_retries", experiment.work.execution_retries);
    field("recycling_discarded_primal_attempt", discarded_primal_attempt ? "true" : "false");
    field("resident_jk_discarded_attempt", discarded_response_attempt ? "true" : "false");
    field("direct_post_scf_fock_builds", experiment.work.post_scf_fock_builds);
    field("energy_change", energy_change);
    field("density_rms", density_rms);
    field("reference_energy", reference.energy);
    field("commutator_residual", reference.commutator_residual);
    field("canonical_density_drift", reference.canonical_density_drift);
    field("eigen_residual", reference.eigen_residual);
    field("orthogonality_max", orthogonality_max);
    field("energy", energy);
    field("endpoint_seconds", endpoint_seconds);
    field("native_seconds", native_seconds);
    field("ccsd_seconds", ccsd_seconds);
    field("triples_seconds", triples_seconds);
    field("lambda_seconds", lambda_seconds);
    field("orbital_seconds", orbital_seconds);
    field("lambda_residual", lambda_residual);
    field("z_residual", z_residual);
    field("stationarity", stationarity);
    array("density", reference.density);
    array("orbital_energies", reference.orbital_energies);
    array("forces", forces);
    output << "\"converged\": true\n}\n";
    std::cout << mode << ' ' << endpoint << " RHF " << pre_seconds + direct_seconds
              << " s; Direct cycles " << direct_cycles << "; endpoint " << endpoint_seconds
              << " s\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

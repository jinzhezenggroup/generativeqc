// Complete cold DF-CCSD(T) energy/force benchmark with explicit schedule selectors.
// The input contains no orbitals, Fock matrix, factors or amplitudes from an oracle.
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cc/df_triples.hpp"
#include "cc_source_session.hpp"
#include "df_ccsdt_force_report.hpp"
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

static int run_scientific_endpoint(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 33)
      throw std::invalid_argument(
          "usage: df-force-endpoint INPUT OUTPUT_JSON REDUCTION_0_OR_1 [MATRIX_0_OR_1 "
          "[FORCES_0_OR_1 [LAMBDA_MATRIX_0_OR_1 [Q_BATCH_LIMIT [DIIS_HISTORY "
          "[CCSD_Q_BATCH_LIMIT_OR_AUTO_0 "
          "[ORBITAL_SCHWARZ "
          "[PROFILE_JK_0_OR_1 [NUCLEAR_AUTO_OR_0_LEGACY_1_BILINEAR_2_SYMMETRIC "
          "[DERIVED_DENOMINATORS_0_OR_1 [Z_TRUE_RESIDUAL_INTERVAL [Z_DF_PRECONDITIONER_0_OR_1 "
          "[Z_RECYCLE_REPEAT_0_OR_1 [PACKED_DIIS_0_OR_1 [RESIDENT_JK_MAXIMUM_BYTES_OR_AUTO "
          "[PARALLEL_GAP_0_OR_1 [REQUEST_GAP_0_OR_1 [REFERENCE_TOLERANCE_OR_AUTO "
          "[FUSED_SCALAR_RESPONSE_0_OR_1 [TRIPLES_W_FP32_0_OR_1 "
          "[LAMBDA_TRUE_RESIDUAL_INTERVAL [LAMBDA_CORE_REUSE_0_OR_1 "
          "[LAMBDA_AUDIT_MATRIX_0_OR_1 [LAMBDA_PRIMAL_MATRIX_0_OR_1 "
          "[LAMBDA_SERIAL_SCRATCH_REUSE_0_OR_1 [LIFECYCLE_0_OR_1 "
          "[CCSD_MIXED_PRECISION_AUTO_OR_0_OR_1 "
          "[LAMBDA_MIXED_PRECISION_0_OR_1 "
          "[CCSD_MIXED_SWITCH_RESIDUAL]]]]]]]]]]]]]]]]]]]]]]]]]]]]]");
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
    const std::size_t batch_limit = unsigned_argument(7, 32);
    // Preserve the established DIIS and CCSD batch slots; append response controls.
    const auto diis_history = unsigned_argument(8, 6);
    if (diis_history == 1 || diis_history > 20)
      throw std::invalid_argument("invalid endpoint DIIS history");
    const auto ccsd_batch_limit = unsigned_argument(9, 0);
    generativeqc::hf::RHFFrameResponseOptions frame_options;
    const std::string screening_argument = argc > 10 ? argv[10] : "0";
    std::size_t screening_consumed = 0;
    frame_options.orbital_screening_tolerance = std::stod(screening_argument, &screening_consumed);
    if (screening_consumed != screening_argument.size() ||
        !std::isfinite(frame_options.orbital_screening_tolerance) ||
        frame_options.orbital_screening_tolerance < 0.0)
      throw std::invalid_argument("invalid orbital screening threshold");
    frame_options.profile_jk = argc > 11 && selector(11);
    const std::string nuclear_selector = argc > 12 ? argv[12] : "auto";
    if (nuclear_selector != "auto" && nuclear_selector != "0" && nuclear_selector != "1" &&
        nuclear_selector != "2")
      throw std::invalid_argument("invalid nuclear response selector");
    if (nuclear_selector != "auto") {
      const auto nuclear_schedule = nuclear_selector[0] - '0';
      frame_options.bilinear_derivative = nuclear_schedule == 1;
      frame_options.symmetric_polarization = nuclear_schedule == 2;
    }
    const bool derived_denominators = selector(13);
    if (argc > 14) frame_options.gmres.true_residual_every = unsigned_argument(14, 0);
    if (argc > 15) frame_options.df_preconditioning = selector(15);
    generativeqc::hf::RHFFrameResponseRecycle recycling;
    const bool recycle_repeat = argc > 16 && selector(16);
    const bool lifecycle = argc > 29 && selector(29);
    if (recycle_repeat) frame_options.recycling = &recycling;
    const bool packed_diis = argc > 17 && selector(17);
    if (argc > 18 && std::string(argv[18]) != "auto")
      frame_options.resident_jk_maximum_bytes = unsigned_argument(18, 0);
    const bool parallel_gap_reduction = argc <= 19 || selector(19);
    const bool request_triples_gap_cotangents = argc > 20 && selector(20);
    const bool fused_triples_scalar_response =
        argc <= 22 || std::string(argv[22]) == "auto" || selector(22);
    const bool lambda_core_reuse = selector(25);
    const bool lambda_audit_matrix = selector(26);
    const bool lambda_primal_matrix = argc <= 27 || selector(27);
    const bool lambda_serial_scratch_reuse =
        argc <= 28 || std::string(argv[28]) == "auto" || selector(28);
    const bool triples_w_fp32 = argc > 23 && selector(23);
    std::optional<bool> ccsd_mixed_precision;
    if (argc > 30 && std::string(argv[30]) != "auto") ccsd_mixed_precision = selector(30);
    const bool lambda_mixed_precision = argc > 31 && selector(31);
    double ccsd_mixed_switch_residual =
        generativeqc::cc::SolverOptions{}.df_mixed_precision_switch_residual;
    if (argc > 32) {
      const std::string token(argv[32]);
      std::size_t consumed = 0;
      ccsd_mixed_switch_residual = std::stod(token, &consumed);
      if (consumed != token.size() || !std::isfinite(ccsd_mixed_switch_residual))
        throw std::invalid_argument("invalid CCSD mixed switch residual");
    }
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
    for (int repetition = 0; repetition < (lifecycle ? 3 : recycle_repeat ? 2 : 1); ++repetition) {
      // Warm calls reuse this process/context, not imported scientific state.
      // The moved call rebuilds both bases at one explicit displaced geometry.
      if (lifecycle && repetition == 2) {
        orbital.atoms.front().position[2] += 1e-4;
        auxiliary.atoms = orbital.atoms;
      }
      const auto result = generativeqc::methods::detail::run_df_ccsdt_native(
          execution, orbital, auxiliary, descriptor, forces, true, reduction, matrix, lambda_matrix,
          batch_limit, ccsd_batch_limit, frame_options, derived_denominators, packed_diis,
          parallel_gap_reduction, request_triples_gap_cotangents, fused_triples_scalar_response,
          admitted_triples_w, lambda_true_residual_interval, lambda_core_reuse, lambda_audit_matrix,
          lambda_primal_matrix, nullptr, lambda_serial_scratch_reuse, ccsd_mixed_precision,
          lambda_mixed_precision, false, ccsd_mixed_switch_residual);
      const char* suffix = repetition == 2 ? ".moved-warm.json" : repetition ? ".warm.json" : "";
      generativeqc::benchmarks::publish_df_ccsdt_force_report(
          std::string(argv[2]) + suffix, result,
          {generativeqc::molecule::ao_count(orbital),
           generativeqc::molecule::ao_count(auxiliary),
           repetition,
           forces,
           reduction,
           matrix,
           descriptor.energy_tolerance,
           descriptor.density_tolerance,
           lambda_true_residual_interval,
           frame_options.gmres.true_residual_every,
           ccsd_mixed_precision,
           ccsd_mixed_switch_residual,
           diis_history,
           previous_endpoint_seconds,
           triples_w_fp32,
           lambda_core_reuse,
           lambda_audit_matrix,
           lambda_primal_matrix,
           lambda_serial_scratch_reuse,
           lambda_mixed_precision});
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

int main(int argc, char** argv) {
  const char* backend = nullptr;
  const char* output = nullptr;
  if (argc >= 7 && std::strcmp(argv[argc - 3], "--cc-cupti-v1") == 0) {
    backend = argv[argc - 2];
    output = argv[argc - 1];
    argc -= 3;
  }
  try {
    generativeqc::benchmarks::cc_trace::Session observation(backend, output);
    // All scientific locals/resource owners die on return before the explicit
    // observer-only fence/export. Original scientific timing stays inside.
    const int scientific_status = run_scientific_endpoint(argc, argv);
    const bool captured = observation.finish(static_cast<std::uint64_t>(scientific_status));
    return scientific_status ? scientific_status : captured ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}

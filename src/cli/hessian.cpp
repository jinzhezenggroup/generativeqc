#include "cli/hessian.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <string_view>

#if GENERATIVEQC_CLI_NATIVE_HESSIAN
#include <cuda_runtime_api.h>

#include "cli/native_basis.hpp"
#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "molecule/basis.hpp"
#include "response/wb97mv_stationary_model.hpp"
#endif

namespace generativeqc::cli {
namespace {
std::uint64_t positive_integer(std::string_view text, std::string_view option,
                               std::uint64_t maximum = INT64_MAX) {
  std::uint64_t parsed{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !parsed ||
      parsed > maximum)
    throw UsageError(std::string(option) + " requires a positive bounded integer");
  return parsed;
}
double nonnegative_number(std::string_view text, std::string_view option) {
  std::size_t consumed{};
  double parsed{};
  try {
    parsed = std::stod(std::string(text), &consumed);
  } catch (...) {
    throw UsageError(std::string(option) + " requires a finite nonnegative number");
  }
  if (consumed != text.size() || !std::isfinite(parsed) || parsed < 0)
    throw UsageError(std::string(option) + " requires a finite nonnegative number");
  return parsed;
}
std::vector<double> read_direction(const std::string& path, std::size_t atoms) {
  const auto panel = read_json(path);
  if (!panel.is_array() || panel.size() != atoms)
    throw UsageError("direction must be a finite natoms-by-3 array");
  std::vector<double> result;
  for (const auto& row : panel) {
    if (!row.is_array() || row.size() != 3)
      throw UsageError("direction must be a finite natoms-by-3 array");
    for (const auto& value : row) {
      if (!value.is_number() || !std::isfinite(value.get<double>()))
        throw UsageError("direction must be a finite natoms-by-3 array");
      result.push_back(value.get<double>());
    }
  }
  return result;
}
#if GENERATIVEQC_CLI_NATIVE_HESSIAN
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(generativeqc_status status, const std::string& detail) {
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
/** Decode input only. The common native validator owns normalization, effective
 * charges, ECP/channel legality and orbital/auxiliary representation admission. */
core::System native_system(const NativeBasisData& basis,
                           const std::vector<generativeqc_atom>& atoms, const RunOptions& options) {
  core::System system;
  system.charge = options.charge;
  system.multiplicity = options.multiplicity;
  system.basis_representation = basis.representation;
  for (std::size_t index = 0; index < atoms.size(); ++index) {
    const auto& atom = atoms[index];
    system.atoms.push_back({atom.atomic_number,
                            {atom.x, atom.y, atom.z},
                            basis.core_electrons.empty() ? 0 : basis.core_electrons.at(index)});
  }
  for (const auto& shell : basis.shells) {
    core::Shell output{shell.atom_index, shell.angular_momentum, {}};
    for (std::size_t index = 0; index < shell.primitive_count; ++index) {
      const auto& primitive = basis.primitives.at(shell.primitive_offset + index);
      output.primitives.push_back({primitive.exponent, primitive.coefficient});
    }
    system.shells.push_back(std::move(output));
  }
  for (const auto& term : basis.ecp_terms)
    system.ecp_terms.push_back(
        {term.atom_index, term.channel, term.power, term.exponent, term.coefficient});
  std::string detail;
  require(molecule::validate_and_normalize(system, detail), detail);
  return system;
}
struct DeviceStorage {
  double *direction{}, *output{};
  cudaStream_t stream{};
  ~DeviceStorage() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (direction) (void)cudaFree(direction);
    if (output) (void)cudaFree(output);
  }
};
Json execute_native(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                    const HessianInvocation& invocation, const std::vector<double>& motion) {
  const auto started = std::chrono::steady_clock::now();
  // Initial device selection belongs to the CLI context, never to an action.
  check(cudaSetDevice(options.device_id));
  core::ContextState context;
  context.requested_backend = context.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  context.device_id = options.device_id;
  const auto representation =
      options.representation_explicit ? std::optional(options.representation) : std::nullopt;
  const auto basis = expand_basis_input(options.basis_name, atoms, representation);
  auto system = native_system(basis, atoms, options);
  std::optional<core::System> auxiliary;
  if (options.auxiliary_basis_explicit) {
    const auto input = expand_basis_input(
        options.auxiliary_basis_name, atoms,
        is_local_basis(options.auxiliary_basis_name) ? std::nullopt : representation);
    if (!input.ecp_terms.empty())
      throw UsageError("an auxiliary basis cannot contain ECP potentials");
    auxiliary = native_system(input, atoms, options);
  }
  const unsigned spins = options.method_name == "wb97m-v-uks" ? 2 : 1;
  generativeqc_ks_options ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = GENERATIVEQC_ABI_VERSION;
  ks.grid_version = options.grid.version;
  ks.radial_points = options.grid.radial_points;
  ks.angular_polar = options.grid.angular_polar;
  ks.angular_azimuth = options.grid.angular_azimuth;
  ks.partition_iterations = options.grid.partition_iterations;
  ks.coincident_tolerance = options.grid.coincident_tolerance;
  ks.tile_points = options.grid.tile_points;
  ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  ks.nonlocal_maximum_bytes = invocation.grid_bytes;
  if (spins == 1)
    dft::wb97mv_stationary_model::configure_1(ks);
  else
    dft::wb97mv_stationary_model::configure_2(ks);
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = spins == 1 ? GENERATIVEQC_METHOD_PBE_RKS : GENERATIVEQC_METHOD_PBE_UKS;
  descriptor.max_iterations = options.max_iterations.value_or(250);
  descriptor.diis_history = options.diis_history;
  descriptor.energy_tolerance = options.energy_tolerance.value_or(1e-12);
  descriptor.density_tolerance = options.density_tolerance.value_or(1e-10);
  descriptor.screening_tolerance = options.screening_tolerance.value_or(1e-14);
  descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
  descriptor.density_fitting_mode = options.density_fitting;
  descriptor.density_fitting_relative_threshold = options.density_fitting_threshold;
  descriptor.density_fitting_memory_budget_bytes = options.density_fitting_memory_bytes;
  descriptor.ks_options = &ks;
  auto batch = methods::prepare_batch(context, {std::move(system)}, descriptor,
                                      GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
                                      auxiliary ? &*auxiliary : nullptr);
  const auto solved = batch->execute(methods::Coordinates(1), false);
  if (solved.size() != 1 || solved[0].status != GENERATIVEQC_STATUS_SUCCESS ||
      !solved[0].calculation.convergence.converged)
    throw std::runtime_error("native WB97M-V SCF did not converge; no Hessian published");
  std::string detail;
  dft::CudaKsFinalStateToken token;
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail), detail);
  dft::CudaKsGridReference reference(*batch, 0, token);
  require(reference.prepare(detail), detail);
  DeviceStorage storage;
  storage.stream = reference.reference().stream;
  const auto coordinates = 3 * atoms.size(),
             count = invocation.operation == "hvp" ? coordinates : coordinates * coordinates;
  const auto external_bytes = (motion.size() + count) * sizeof(double);
  if (external_bytes >= invocation.device_bytes)
    throw UsageError("endpoint output exceeds --device-budget-bytes");
  check(cudaMalloc(reinterpret_cast<void**>(&storage.output), count * sizeof(double)));
  if (!motion.empty()) {
    check(cudaMalloc(reinterpret_cast<void**>(&storage.direction), motion.size() * sizeof(double)));
    check(cudaMemcpyAsync(storage.direction, motion.data(), motion.size() * sizeof(double),
                          cudaMemcpyHostToDevice, storage.stream));
  }
  dft::CudaKsWb97mvHessianOptions policy;
  policy.max_device_bytes = invocation.device_bytes - external_bytes;
  policy.grid.max_workspace_bytes = policy.geometry.max_workspace_bytes = invocation.grid_bytes;
  policy.induced.grid.max_workspace_bytes = invocation.grid_bytes;
  policy.grid.max_pair_evaluations = policy.induced.grid.max_pair_evaluations =
      invocation.max_nonlocal_pairs;
  policy.one_electron.max_workspace_bytes = policy.exact.max_workspace_bytes =
      invocation.integral_bytes;
  policy.raw_df.max_workspace_bytes = policy.fitted.max_workspace_bytes =
      policy.ecp.max_workspace_bytes = invocation.integral_bytes;
  policy.cpks.max_workspace_bytes = policy.cpks.gmres.max_workspace_bytes =
      invocation.response_bytes;
  policy.cpks.gmres.relative_tolerance = invocation.response_rtol;
  policy.cpks.gmres.absolute_tolerance = invocation.response_atol;
  policy.cpks.gmres.restart = invocation.response_restart;
  policy.cpks.gmres.max_iterations = invocation.response_iterations;
  dft::CudaKsWb97mvHessian endpoint;
  require(endpoint.prepare(*batch, 0, token, policy, detail), detail);
  dft::CudaKsWb97mvHessianResult result;
  if (motion.empty())
    check(endpoint.hessian(storage.output, storage.stream, result));
  else
    check(endpoint.apply(storage.direction, storage.output, storage.stream, result));
  if (!result.converged())
    throw std::runtime_error("native stationary response did not converge; no Hessian published");
  std::vector<double> values(count);
  check(cudaMemcpyAsync(values.data(), storage.output, count * sizeof(double),
                        cudaMemcpyDeviceToHost, storage.stream));
  check(cudaStreamSynchronize(storage.stream));
  Json panel = Json::array();
  const auto columns = motion.empty() ? coordinates : 3;
  for (std::size_t row = 0; row < count / columns; ++row) {
    Json line = Json::array();
    for (std::size_t column = 0; column < columns; ++column)
      line.push_back(values[row * columns + column]);
    panel.push_back(std::move(line));
  }
  const auto work = endpoint.work();
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Json diagnostics{
      {"derivative_authority", "compiler-ad"},
      {"execution_residency", "device-resident"},
      {"runtime_python", false},
      {"runtime_compilation", false},
      {"displaced_scf_evaluations", 0},
      {"numerical_host_fallbacks", 0},
      {"intermediate_panel_downloads", 0},
      {"posthoc_symmetrization", false},
      {"first_integral_backend", "cuda"},
      {"second_integral_backend", "cuda"},
      {"grid_ad_backend", "cuda"},
      {"response_backend", "cuda"},
      {"device_bytes", work.device_bytes + external_bytes},
      {"completed_directions", work.completed_directions},
      {"induced_fock_actions", work.induced.actions},
      {"exact_primitive_components", work.exact.component_evaluations},
      {"one_electron_primitive_components", work.one_electron.integrals.component_evaluations},
      {"df_value_passes", work.raw_df.value_passes},
      {"df_contraction_passes", work.raw_df.contraction_passes},
      {"grid_right_directions", work.grid.nuclear_directions},
      {"grid_mixed_contractions", work.grid.nuclear_contractions},
      {"vv10_reference_pairs", work.induced.grid.reference_pairs + work.grid.reference_pairs},
      {"native_execution_seconds", seconds},
      {"scf_iterations", solved[0].calculation.convergence.iterations},
      {"response_iterations", result.response.gmres.iterations},
      {"ecp_ad_backend", work.ecp.workspace_bytes ? Json("cuda") : Json(nullptr)},
      {"df_integral_ad_backend", work.raw_df.workspace_bytes ? Json("cuda") : Json(nullptr)}};
  return {{"schema", "generativeqc.native-hessian-result/v1"},
          {"operation", invocation.operation},
          {"units", "hartree/bohr^2"},
          {"backend", "cuda"},
          {"method", options.method_name},
          {"method_ir_identity", spins == 1 ? dft::wb97mv_stationary_model::identity_1
                                            : dft::wb97mv_stationary_model::identity_2},
          {"native_source_identity", generativeqc_get_source_identity()},
          {"charge", options.charge},
          {"multiplicity", options.multiplicity},
          {"energy_hartree", solved[0].calculation.energy},
          {"diagnostics", std::move(diagnostics)},
          {motion.empty() ? "matrix" : "value", std::move(panel)}};
}
#endif
}  // namespace

HessianInvocation parse_hessian_invocation(int argc, char** argv) {
  if (argc < 3) throw UsageError("hessian/hvp requires an XYZ input path");
  HessianInvocation invocation;
  invocation.operation = argv[1];
  invocation.run_arguments = {argv[0], invocation.operation, argv[2], "--method", "wb97m-v-rks"};
  const std::set<std::string_view> run_options{"--method",
                                               "--basis",
                                               "--representation",
                                               "--density-fitting",
                                               "--auxiliary-basis",
                                               "--backend",
                                               "--device-id",
                                               "--charge",
                                               "--multiplicity",
                                               "--diis-history",
                                               "--precision",
                                               "--density-fitting-threshold",
                                               "--density-fitting-memory-bytes",
                                               "--max-iterations",
                                               "--energy-tolerance",
                                               "--density-tolerance",
                                               "--screening-tolerance",
                                               "--units",
                                               "--grid-radial-points",
                                               "--grid-polar-points",
                                               "--grid-azimuth-points"};
  for (int index = 3; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--json") {
      invocation.run_arguments.emplace_back(option);
      continue;
    }
    if (option == "--compiler-python" || option == "--compiler-library" || option == "--cache" ||
        option == "--timeout-seconds")
      throw UsageError(
          "native hessian/hvp has no Python worker or runtime compilation; obsolete option: " +
          std::string(option));
    if (++index >= argc) throw UsageError(std::string(option) + " requires a value");
    const std::string_view value = argv[index];
    if (run_options.contains(option)) {
      invocation.run_arguments.emplace_back(option);
      invocation.run_arguments.emplace_back(value);
    } else if (option == "--direction")
      invocation.direction_file = value;
    else if (option == "--device-budget-bytes")
      invocation.device_bytes = positive_integer(value, option);
    else if (option == "--integral-budget-bytes")
      invocation.integral_bytes = positive_integer(value, option);
    else if (option == "--grid-budget-bytes")
      invocation.grid_bytes = positive_integer(value, option);
    else if (option == "--output-budget-bytes")
      invocation.output_bytes = positive_integer(value, option);
    else if (option == "--response-budget-bytes")
      invocation.response_bytes = positive_integer(value, option);
    else if (option == "--max-nonlocal-pairs")
      invocation.max_nonlocal_pairs = positive_integer(value, option);
    else if (option == "--response-atol")
      invocation.response_atol = nonnegative_number(value, option);
    else if (option == "--response-rtol") {
      invocation.response_rtol = nonnegative_number(value, option);
      if (invocation.response_rtol >= 1) throw UsageError("--response-rtol must be less than 1");
    } else if (option == "--response-restart")
      invocation.response_restart = positive_integer(value, option, INT32_MAX);
    else if (option == "--response-max-iterations")
      invocation.response_iterations = positive_integer(value, option, INT32_MAX);
    else
      throw UsageError("unsupported hessian/hvp option: " + std::string(option));
  }
  if ((invocation.operation == "hvp") != !invocation.direction_file.empty())
    throw UsageError("only hvp requires --direction FILE.json (finite natoms-by-3 array)");
  return invocation;
}
int hessian_command(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                    const HessianInvocation& invocation) {
  if (options.method_name != "wb97m-v" && options.method_name != "wb97m-v-rks" &&
      options.method_name != "wb97m-v-uks")
    throw UsageError("native hessian/hvp currently admits only WB97M-V RKS/UKS");
  if (options.backend != GENERATIVEQC_BACKEND_CUDA ||
      options.precision != GENERATIVEQC_PRECISION_FP64)
    throw UsageError(
        "native hessian/hvp requires --backend cuda --precision fp64; no Python/CPU fallback");
  if (options.density_fitting == GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE)
    throw UsageError("CUDA hessian/hvp cannot select a CPU density-fitting provider");
  const auto coordinates = 3ULL * atoms.size();
  if (!coordinates || coordinates > SIZE_MAX / coordinates / sizeof(double))
    throw UsageError("Hessian dimensions overflow");
  const auto count = invocation.operation == "hvp" ? coordinates : coordinates * coordinates;
  if (count > invocation.output_bytes / (2 * sizeof(double)))
    throw UsageError("result and publication exceed --output-budget-bytes");
  const auto motion = invocation.operation == "hvp"
                          ? read_direction(invocation.direction_file, atoms.size())
                          : std::vector<double>{};
#if GENERATIVEQC_CLI_NATIVE_HESSIAN
  const auto result = execute_native(options, atoms, invocation, motion);
  if (options.json)
    std::cout << result.dump() << '\n';
  else {
    std::cout << "WB97M-V " << invocation.operation
              << " (hartree/bohr^2; native compiler AD, CUDA)\n";
    for (const auto& row : result.at(invocation.operation == "hvp" ? "value" : "matrix"))
      std::cout << row.dump() << '\n';
  }
  return 0;
#else
  throw UsageError(
      "native Hessian AOT closure is not linked; rebuild with "
      "GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON and an explicit complete inventory; no runtime "
      "Python/JIT fallback");
#endif
}
}  // namespace generativeqc::cli

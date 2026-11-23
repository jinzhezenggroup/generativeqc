#include "cli/hessian.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string_view>

#include "api/ks_host_hessian.hpp"
#include "cli/molecular_system.hpp"
#include "cli/native_basis.hpp"
#include "cli/resources.hpp"

#if GENERATIVEQC_CLI_NATIVE_HESSIAN
#include <cuda_runtime_api.h>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "dft/split_hybrid_cuda_domain.hpp"
#include "molecule/basis.hpp"
#include "molecule/basis_template.hpp"
#include "r2scan_cuda_domain.hpp"
#include "response/wb97mv_stationary_model.hpp"
#endif

namespace generativeqc::cli {
namespace {
bool is_wb97mv(std::string_view method) {
  return method == "wb97m-v" || method == "wb97m-v-rks" || method == "wb97m-v-uks";
}

bool is_split_hybrid(std::string_view method) {
  return method == "m06-2x-rks" || method == "m06-2x-uks" || method == "mn15-rks" ||
         method == "mn15-uks";
}

/** AUTO is an SCF proposal schedule, not a lower-precision Hessian. Only the
 * admitted electronic models with an unchanged point domain may select it. */
bool auto_hessian_scf_model(std::string_view method) {
  return is_wb97mv(method) || method == "lda-rks" || method == "lda-uks" || method == "pbe-rks" ||
         method == "pbe-uks" || method == "pbe0-rks" || method == "pbe0-uks" ||
         method == "b3lyp-rks" || method == "b3lyp-uks";
}

bool is_r2scan_rvv10(std::string_view method) {
#if GENERATIVEQC_CLI_NATIVE_HESSIAN
  const auto* entry = method_generated::find_method(method);
  if (!dft::kCudaR2scanVariational || !entry || !entry->cuda) return false;
  const auto* canonical =
      method_generated::find_method(entry->spin == 2 ? "r2scan+rvv10-uks" : "r2scan+rvv10-rks");
  return canonical && entry->identity == canonical->identity;
#else
  return false;
#endif
}

/** Compare catalog semantics so aliases cannot change second-order admission. */
bool qualified_standard_d4(const method_generated::Method& entry) {
  const auto matches = [&](std::string_view restricted, std::string_view unrestricted) {
    const auto selector = entry.spin == 2 ? unrestricted : restricted;
    const auto* canonical = method_generated::find_method(selector);
    return canonical && canonical->cuda && entry.identity == canonical->identity;
  };
  if (matches("pbe-d4(bj-eeq-atm)-rks", "pbe-d4(bj-eeq-atm)-uks") ||
      matches("pbe0-d4(bj-eeq-atm)-rks", "pbe0-d4(bj-eeq-atm)-uks") ||
      matches("b3lyp-d4(bj-eeq-atm)-rks", "b3lyp-d4(bj-eeq-atm)-uks"))
    return true;
#if GENERATIVEQC_CLI_NATIVE_HESSIAN
  return dft::kCudaR2scanVariational &&
         matches("r2scan-d4(bj-eeq-atm)-rks", "r2scan-d4(bj-eeq-atm)-uks");
#else
  return false;
#endif
}
std::string_view semilocal_catalog_selector(std::string_view method) {
  if (method == "lda-rks") return "lda_xc_pw-rks";
  if (method == "lda-uks") return "lda_xc_pw-uks";
  if (method == "pbe-d4-rks") return "pbe-d4(bj-eeq-atm)-rks";
  if (is_r2scan_rvv10(method)) return method;
  // A catalog correction cannot grant curvature to an unqualified electronic
  // functional. Refuse unsupported D4 rows before running their SCF.
  if (const auto* entry = method_generated::find_method(method); entry && entry->cuda) {
    if (entry->dispersion.kind == 1 || entry->dispersion.kind == 2) return method;
    if (entry->dispersion.kind == 3 && entry->dispersion.composite_profile == 0 &&
        qualified_standard_d4(*entry))
      return method;
  }
  if (method == "pbe-rks" || method == "pbe-uks" || method == "pbe0-rks" || method == "pbe0-uks" ||
      method == "b3lyp-rks" || method == "b3lyp-uks")
    return method;
#if GENERATIVEQC_CLI_NATIVE_HESSIAN
  if ((dft::kCudaM062XVariational && (method == "m06-2x-rks" || method == "m06-2x-uks")) ||
      (dft::kCudaMN15Variational && (method == "mn15-rks" || method == "mn15-uks")))
    return method;
  if (dft::kCudaR2scanVariational && (method == "r2scan-rks" || method == "r2scan-uks" ||
                                      method == "r2scan-3c-rks" || method == "r2scan-3c-uks"))
    return method;
#endif
  return {};
}
std::uint64_t positive_integer(std::string_view text, std::string_view option,
                               std::uint64_t maximum = INT64_MAX) {
  std::uint64_t parsed{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !parsed ||
      parsed > maximum)
    throw UsageError(std::string(option) + " requires a positive bounded integer");
  return parsed;
}

/** MethodIR serializes reduced fractions as strings, not JSON floats. Bound
 * each integer to the exactly representable FP64 range so the only rounding
 * is the final division, as in Python's Fraction-to-double lowering. */
double method_ir_fraction(const std::string& text, std::string_view field) {
  constexpr std::uint64_t exact_integer_limit = 1ULL << 53;
  const auto separator = text.find('/');
  const std::string_view value(text);
  const auto numerator = positive_integer(value.substr(0, separator), field, exact_integer_limit);
  const auto denominator = separator == text.npos ? 1
                                                  : positive_integer(value.substr(separator + 1),
                                                                     field, exact_integer_limit);
  auto canonical = std::to_string(numerator);
  if (denominator != 1) canonical += '/' + std::to_string(denominator);
  if (std::gcd(numerator, denominator) != 1 || text != canonical)
    throw UsageError(std::string(field) + " must be a canonical positive MethodIR fraction");
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

RangePbeMethodIR read_range_pbe_method_ir(const std::string& path) {
  try {
    const auto payload = read_json(path);
    RangePbeMethodIR result;
    result.identifier = payload.at("identifier").get<std::string>();
    if (result.identifier.find_first_not_of(" \t\r\n\v\f") == std::string::npos)
      throw UsageError("MethodIR requires a non-empty identifier");
    const auto& spin = payload.at("spin");
    if (spin == "unpolarized")
      result.spins = 1;
    else if (spin == "polarized")
      result.spins = 2;
    else
      throw UsageError("MethodIR spin must be unpolarized or polarized");
    const auto& primitives = payload.at("primitives");
    if (!primitives.is_array() || primitives.size() != 3 || !primitives[0].is_object())
      throw UsageError("expected unscaled PBE and exactly two range-exchange primitives");

    // Authenticate the complete semilocal graph, including its source/domain
    // provenance, against the same catalog row used to lower ordinary PBE.
    auto semantic = payload;
    semantic.erase("identifier");
    semantic.erase("requirements");
    semantic.at("primitives").at(0).erase("functional_identifier");
    auto base = semantic;
    base.at("primitives").erase(2);
    base.at("primitives").erase(1);
    const auto* pbe = method_generated::find_method(result.spins == 1 ? "pbe-rks" : "pbe-uks");
    if (!pbe || canonical_hash(base) != pbe->identity)
      throw UsageError("MethodIR semilocal graph does not match the linked unscaled PBE catalog");

    const auto& range = semantic.at("primitives");
    const double divisor = result.spins == 1 ? 2.0 : 1.0;
    for (std::size_t index = 0; index != result.exchange.size(); ++index) {
      const auto& term = range[index + 1];
      const auto op = index == 0 ? "short-range" : "long-range";
      if (!term.is_object() || term.size() != 6 || term.at("kind") != "range_separated_exchange" ||
          term.at("operator") != op || term.at("omega_units") != "bohr^-1" ||
          term.at("derivative_capabilities") != Json::array({"energy", "fock", "nuclear-gradient"}))
        throw UsageError("MethodIR requires canonical short-range and long-range HF primitives");
      const auto coefficient =
          method_ir_fraction(term.at("coefficient").get<std::string>(), "MethodIR HF coefficient");
      const auto omega = method_ir_fraction(term.at("omega").get<std::string>(), "MethodIR omega");
      if (coefficient > 1) throw UsageError("MethodIR HF fractions must not exceed one");
      result.exchange[index] = {
          index == 0 ? GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE : GENERATIVEQC_KS_EXCHANGE_LONG_RANGE,
          coefficient, omega, -coefficient / divisor};
    }
    if (range[1].at("omega") != range[2].at("omega") ||
        result.exchange[0].coefficient == result.exchange[1].coefficient)
      throw UsageError("MethodIR requires one shared omega and distinct FP64 SR/LR fractions");
    result.identity = canonical_hash(semantic);
    return result;
  } catch (const std::exception& error) {
    throw UsageError("--method-ir: " + std::string(error.what()));
  }
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
void append_direction(const Json& panel, std::size_t atoms, std::vector<double>& result) {
  if (!panel.is_array() || panel.size() != atoms)
    throw UsageError("direction must be a finite natoms-by-3 array");
  for (const auto& row : panel) {
    if (!row.is_array() || row.size() != 3)
      throw UsageError("direction must be a finite natoms-by-3 array");
    for (const auto& value : row) {
      if (!value.is_number() || !std::isfinite(value.get<double>()))
        throw UsageError("direction must be a finite natoms-by-3 array");
      result.push_back(value.get<double>());
    }
  }
}

std::vector<double> read_direction(const std::string& path, std::size_t atoms) {
  std::vector<double> values;
  append_direction(read_json(path), atoms, values);
  return values;
}

std::vector<double> read_directions(const std::string& path, std::size_t atoms,
                                    std::uint64_t output_bytes) {
  const auto panel = read_json(path);
  if (!panel.is_array() || panel.empty())
    throw UsageError("directions must be a nonempty array of finite natoms-by-3 arrays");
  const auto coordinates = 3 * atoms;
  if (panel.size() > SIZE_MAX / coordinates ||
      panel.size() > output_bytes / (2 * sizeof(double)) / coordinates)
    throw UsageError("action panel and publication exceed --output-budget-bytes");
  std::vector<double> values;
  values.reserve(panel.size() * coordinates);
  for (const auto& direction : panel) append_direction(direction, atoms, values);
  return values;
}

/** The SDK, Python and CPU CLI share this synchronous prepared-batch bridge. */
Json execute_cpu(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                 const HessianInvocation& invocation, const std::vector<double>& motion) {
  const auto* program = linked_cpu_hessian_program(options);
  if (!program)
    throw UsageError(
        "native CPU Hessian has no exact linked total MethodIR; no Python/JIT fallback");
  if (invocation.response_sequential && program->carrier != GENERATIVEQC_METHOD_LDA_RKS &&
      program->carrier != GENERATIVEQC_METHOD_PBE_RKS)
    throw UsageError("--response-sequential requires linked LDA/PBE RKS");
  if (options.density_fitting != GENERATIVEQC_DENSITY_FITTING_NONE ||
      options.auxiliary_basis_explicit || options.precision != GENERATIVEQC_PRECISION_FP64)
    throw UsageError("native CPU Hessian requires direct FP64 without an auxiliary basis");
  if (invocation.cuda_controls_explicit || invocation.nonlocal_level ||
      invocation.nonlocal_radial || invocation.response_fp32_proposal)
    throw UsageError("CPU Hessian does not use CUDA device/nonlocal budgets");
  const auto started = std::chrono::steady_clock::now();
  const generativeqc_context_descriptor context_options{
      sizeof(context_options), GENERATIVEQC_ABI_VERSION, -1, GENERATIVEQC_BACKEND_CPU_REFERENCE};
  Context context(context_options);
  const auto basis = expand_basis_input(
      options.basis_name, atoms,
      options.representation_explicit ? std::optional(options.representation) : std::nullopt);
  const auto system = make_cpu_hessian_system(context, options, atoms, basis);
  const auto check_status = [&](generativeqc_status status) {
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw Error(status, generativeqc_context_get_last_detail(context.get()));
  };
  auto controls = cpu_hessian_controls(options, atoms, invocation);
  std::size_t minimum_bytes{};
  check_status(cpu_hessian_minimum_bytes(context.get(), system.get(), program->identity.data(),
                                         invocation, controls, &minimum_bytes));
  if (minimum_bytes > controls.numeric_bytes)
    throw Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                "CPU Hessian aggregate numerical reservation exceeded (required=" +
                    std::to_string(minimum_bytes) +
                    ", cap=" + std::to_string(controls.numeric_bytes) + ')');
  std::optional<CpuHessianHostAdmission> host_admission;
  if (options.host_bytes) {
    host_admission =
        admit_cpu_hessian_host_budget(options, atoms, invocation, basis, minimum_bytes);
    controls.numeric_bytes = host_admission->derivative_cap_bytes;
  }
  generativeqc_ks_options grid{};
  grid.struct_size = sizeof(grid);
  grid.abi_version = GENERATIVEQC_ABI_VERSION;
  grid.grid_version = options.grid.version;
  grid.radial_points = options.grid.radial_points;
  grid.angular_polar = options.grid.angular_polar;
  grid.angular_azimuth = options.grid.angular_azimuth;
  grid.partition_iterations = options.grid.partition_iterations;
  grid.coincident_tolerance = options.grid.coincident_tolerance;
  grid.tile_points = options.grid.tile_points;
  methods::named_ks::electronic_options(*program, grid);
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = program->carrier;
  method.max_iterations = options.max_iterations.value_or(250);
  method.diis_history = options.diis_history;
  method.energy_tolerance = options.energy_tolerance.value_or(1e-12);
  method.density_tolerance = options.density_tolerance.value_or(1e-10);
  method.screening_tolerance = options.screening_tolerance.value_or(1e-14);
  method.ks_options = &grid;
  const generativeqc_named_ks_descriptor named{sizeof(named),
                                               GENERATIVEQC_ABI_VERSION,
                                               program->name.data(),
                                               program->identity.data(),
                                               invocation.dispersion_bytes,
                                               invocation.cpu_packed_eri_bytes};
  const generativeqc_system* systems[]{system.get()};
  generativeqc_batch* prepared{};
  check_status(generativeqc_batch_prepare_named_ks_v1(context.get(), systems, 1, &method, &named, 0,
                                                      &prepared));
  std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)> batch(
      prepared, generativeqc_batch_destroy);
  generativeqc_batch_item_result_descriptor primal{};
  primal.struct_size = sizeof(primal);
  primal.abi_version = GENERATIVEQC_ABI_VERSION;
  const auto scf_status = generativeqc_batch_execute(batch.get(), nullptr, 0, &primal, 1);
  if (scf_status == GENERATIVEQC_STATUS_NOT_CONVERGED ||
      (scf_status == GENERATIVEQC_STATUS_SUCCESS &&
       (primal.status == GENERATIVEQC_STATUS_NOT_CONVERGED ||
        (primal.status == GENERATIVEQC_STATUS_SUCCESS && !primal.converged))))
    throw Error(GENERATIVEQC_STATUS_NOT_CONVERGED,
                "native CPU SCF did not converge; no Hessian published");
  check_status(scf_status);
  check_status(primal.status);
  std::uint64_t scf_fock_builds{};
  check_status(generativeqc_batch_get_last_fock_builds(batch.get(), 0, &scf_fock_builds));
  const auto coordinates = 3 * atoms.size();
  std::vector<double> output(motion.empty() ? coordinates * coordinates : motion.size());
  auto receipt = api::empty_host_hessian_receipt();
  if (invocation.hvp_directions)
    check_status(generativeqc_ks_hvp_cpu_batch_v1(batch.get(), 0, &controls, motion.data(),
                                                  motion.size(), output.data(), output.size(),
                                                  &receipt));
  else if (invocation.response_sequential)
    check_status(generativeqc_ks_hessian_cpu_sequential_v1(batch.get(), 0, &controls, output.data(),
                                                           output.size(), &receipt));
  else if (invocation.response_recycle_block_size)
    check_status(generativeqc_ks_hessian_cpu_recycled_v1(batch.get(), 0, &controls,
                                                         invocation.response_recycle_block_size,
                                                         output.data(), output.size(), &receipt));
  else
    check_status(generativeqc_ks_hessian_cpu_v1(
        batch.get(), 0, &controls, motion.empty() ? nullptr : motion.data(), motion.size(),
        output.data(), output.size(), &receipt));
  Json panel = Json::array();
  const auto row_size = motion.empty() ? coordinates : 3;
  const auto append_rows = [&](Json& destination, std::size_t begin, std::size_t end) {
    for (; begin < end; begin += row_size)
      destination.push_back(
          std::vector<double>(output.begin() + begin, output.begin() + begin + row_size));
  };
  if (invocation.hvp_directions) {
    for (std::size_t begin = 0; begin < output.size(); begin += coordinates) {
      auto action = Json::array();
      append_rows(action, begin, begin + coordinates);
      panel.push_back(std::move(action));
    }
  } else {
    append_rows(panel, 0, output.size());
  }
  Json diagnostics{
      {"numerical_runtime", "native-cpp-cpu"},
      {"runtime_compilation", false},
      {"displaced_scf_evaluations", 0},
      {"posthoc_symmetrization", false},
      {"completed_directions", receipt.completed_directions},
      {"response_strategy", invocation.response_sequential           ? "sequential"
                            : invocation.response_recycle_block_size ? "recycled"
                                                                     : "scalar"},
      {"response_iterations", receipt.response_iterations},
      {"response_residual_norm", receipt.response_residual_norm},
      {"response_absolute_tolerance", controls.absolute_tolerance},
      {"response_relative_tolerance", controls.relative_tolerance},
      {"induced_fock_actions", receipt.induced_fock_actions},
      {"first_eri_primitives", receipt.first_eri_primitives},
      {"curvature_eri_primitives", receipt.curvature_eri_primitives},
      {"first_one_electron_primitives", receipt.first_one_electron_primitives},
      {"curvature_one_electron_primitives", receipt.curvature_one_electron_primitives},
      {"source_xc_point_visits", receipt.source_xc_point_visits},
      {"curvature_xc_point_visits", receipt.curvature_xc_point_visits},
      {"response_xc_point_visits", receipt.response_xc_point_visits},
      {"ecp_kernel_blocks", receipt.ecp_kernel_blocks},
      {"ecp_radial_layers", receipt.ecp_radial_layers},
      {"ecp_ao_primitive_points", receipt.ecp_ao_primitive_points},
      {"ecp_contraction_terms", receipt.ecp_contraction_terms},
      {"d3_pair_visits", receipt.d3_pair_visits},
      {"d3_reference_terms", receipt.d3_reference_terms},
      {"d3_energy", receipt.d3_energy},
      {"raw_symmetry_error", receipt.raw_symmetry_error},
      {"numeric_reservation_bytes", receipt.numeric_reservation_bytes},
      {"minimum_numeric_bytes", minimum_bytes},
      {"max_eri_primitives", controls.max_eri_primitives},
      {"max_map_terms", controls.max_map_terms},
      {"max_d3_pairs", controls.max_d3_pairs},
      {"reservation_scope", "derivative-owned; excludes primal/export-scratch/BLAS/RSS"},
      {"scf_iterations", primal.iterations},
      {"scf_fock_builds", scf_fock_builds},
      {"native_execution_seconds", receipt.execution_seconds},
      {"endpoint_seconds",
       std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}};
  if (invocation.response_recycle_block_size)
    diagnostics["block_size"] = invocation.response_recycle_block_size;
  if (invocation.response_sequential) diagnostics["response_concurrent_rhs"] = 1;
  if (invocation.hvp_directions) {
    diagnostics["response_strategy"] = "independent";
    diagnostics["direction_count"] = invocation.hvp_directions;
    diagnostics["output_bytes"] = receipt.output_bytes;
  }
  if (host_admission)
    diagnostics["resource_admission"] = {
        {"effective_host_limit_bytes", host_admission->effective_limit_bytes},
        {"minimum_host_peak_bytes", host_admission->minimum_peak_bytes},
        {"selected_host_peak_bytes", host_admission->selected_peak_bytes},
        {"derivative_numeric_cap_bytes", host_admission->derivative_cap_bytes}};
  return {{"schema", "generativeqc.native-hessian-result/v1"},
          {"operation", invocation.operation},
          {"units", "hartree/bohr^2"},
          {"backend", "cpu"},
          {"method", options.method_name},
          {"method_ir_identity", program->identity},
          {"native_source_identity", generativeqc_get_source_identity()},
          {"charge", options.charge},
          {"multiplicity", options.multiplicity},
          {"energy_hartree", receipt.total_energy},
          {"diagnostics", std::move(diagnostics)},
          {motion.empty() ? "matrix" : "value", std::move(panel)}};
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
  const auto basis = options.basis_program
                         ? molecule::expand_basis_template(*options.basis_program->bound_basis,
                                                           atoms, GENERATIVEQC_BASIS_SPHERICAL)
                         : expand_basis_input(options.basis_name, atoms, representation);
  const bool wb97mv = is_wb97mv(options.method_name);
  const bool split_hybrid = is_split_hybrid(options.method_name);
  const bool nonlocal = wb97mv || is_r2scan_rvv10(options.method_name);
  auto system = native_system(basis, atoms, options);
  if (options.precision == GENERATIVEQC_PRECISION_AUTO &&
      (!system.ecp_terms.empty() ||
       std::any_of(system.atoms.begin(), system.atoms.end(),
                   [](const auto& atom) { return atom.ecp_core != 0; })))
    throw UsageError("native hessian/hvp AUTO is only qualified for all-electron execution");
  std::optional<core::System> auxiliary;
  if (options.auxiliary_basis_explicit) {
    const auto input = expand_basis_input(
        options.auxiliary_basis_name, atoms,
        is_local_basis(options.auxiliary_basis_name) ? std::nullopt : representation);
    if (!input.ecp_terms.empty())
      throw UsageError("an auxiliary basis cannot contain ECP potentials");
    auxiliary = native_system(input, atoms, options);
  }
  if (split_hybrid) {
    const auto within_domain = [](const core::System& input, unsigned spherical_limit) {
      const auto limit =
          input.basis_representation == GENERATIVEQC_BASIS_SPHERICAL ? spherical_limit : 1U;
      return std::all_of(input.shells.begin(), input.shells.end(),
                         [limit](const auto& shell) { return shell.angular_momentum <= limit; });
    };
    if (!within_domain(system, 3) || (auxiliary && !within_domain(*auxiliary, 4)))
      throw UsageError(
          "CUDA split-hybrid Hessian/HVP requires orbital shells through f and "
          "auxiliary shells through g (spherical for d and higher)");
  }
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
  auto method = options.method;
  std::string_view identity;
  const method_generated::Dispersion* corrections{};
  if (wb97mv) {
    if (options.method_name == "wb97m-v-uks") {
      dft::wb97mv_stationary_model::configure_2(ks);
      identity = dft::wb97mv_stationary_model::identity_2;
      method = GENERATIVEQC_METHOD_PBE_UKS;
    } else {
      dft::wb97mv_stationary_model::configure_1(ks);
      identity = dft::wb97mv_stationary_model::identity_1;
      method = GENERATIVEQC_METHOD_PBE_RKS;
    }
  } else {
    // Read the electronic model and identity from the same generated catalog
    // as the named first-order method. Keep LDA's native integral carrier.
    const auto* entry =
        method_generated::find_method(semilocal_catalog_selector(options.method_name));
    if (!entry || !entry->cuda)
      throw UsageError("native semilocal Hessian method is absent from the CUDA catalog");
    method_generated::electronic_options(*entry, ks);
    identity = entry->identity;
    if (invocation.range_pbe) {
      const auto& supplied = *invocation.range_pbe;
      ks.exchange_terms = supplied.exchange.data();
      ks.exchange_term_count = supplied.exchange.size();
      identity = supplied.identity;
    }
    method_generated::validate_bound_basis(*entry, system);
    if (split_hybrid) method = entry->carrier;
    if (entry->dispersion.kind) {
      // Use the catalog's bare electronic carrier; the legacy PBE-D4 enum
      // would otherwise create a second D4 owner alongside the descriptor.
      method = entry->carrier;
      corrections = &entry->dispersion;
    }
  }
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = method;
  descriptor.max_iterations = options.max_iterations.value_or(250);
  descriptor.diis_history = options.diis_history;
  descriptor.energy_tolerance = options.energy_tolerance.value_or(1e-12);
  descriptor.density_tolerance = options.density_tolerance.value_or(1e-10);
  descriptor.screening_tolerance = options.screening_tolerance.value_or(1e-14);
  descriptor.precision_mode = options.precision;
  descriptor.density_fitting_mode = options.density_fitting;
  descriptor.density_fitting_relative_threshold = options.density_fitting_threshold;
  descriptor.density_fitting_memory_budget_bytes = options.density_fitting_memory_bytes;
  descriptor.ks_options = &ks;
  generativeqc_ks_options vv10{};
  generativeqc_ks_nonlocal_grid_options split{};
  // Match the Python KS policy: an XC Treutler override does not also change
  // the default VV10 level, while an explicit legacy tensor grid stays shared.
  const bool default_split = nonlocal && ks.grid_version == 4 && ks.radial_points != 3;
  if (invocation.nonlocal_radial || invocation.nonlocal_level || default_split) {
    vv10 = ks;
    if (invocation.nonlocal_radial) {
      vv10.grid_version = 1;
      vv10.radial_points = invocation.nonlocal_radial;
      vv10.angular_polar = invocation.nonlocal_polar;
      vv10.angular_azimuth = invocation.nonlocal_azimuth;
    } else {
      vv10.grid_version = 4;
      vv10.radial_points = invocation.nonlocal_level.value_or(3);
      vv10.angular_polar = vv10.angular_azimuth = 0;
    }
    split.base.base = ks;
    split.base.base.struct_size = sizeof(split);
    split.version = 1;
    split.grid = &vv10;
    descriptor.ks_options = &split.base.base;
  }
  const auto scf_started = std::chrono::steady_clock::now();
  auto batch = methods::prepare_batch(context, {std::move(system)}, descriptor,
                                      GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
                                      auxiliary ? &*auxiliary : nullptr, corrections,
                                      corrections ? options.dispersion_memory_bytes : 0);
  const auto solved = batch->execute(methods::Coordinates(1), false);
  const auto scf_completed = std::chrono::steady_clock::now();
  if (solved.size() != 1 || solved[0].status != GENERATIVEQC_STATUS_SUCCESS ||
      !solved[0].calculation.convergence.converged)
    throw std::runtime_error("native KS SCF did not converge; no Hessian published");
  std::string detail;
  dft::CudaKsFinalStateToken token;
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail), detail);
  if (split_hybrid) {
    const auto domain = options.method_name.starts_with("mn15-") ? dft::kCudaMN15VariationalDomain
                                                                 : dft::kCudaM062XVariationalDomain;
    if (token.identity.model.scf_domain_version != domain)
      throw std::runtime_error(
          "CUDA split-hybrid Hessian requires the matching variational SCF domain");
  }
  dft::CudaKsGridReference reference(*batch, 0, token);
  require(reference.prepare(detail), detail);
  DeviceStorage storage;
  storage.stream = reference.reference().stream;
  const bool hvp = invocation.operation == "hvp";
  const auto coordinates = 3 * atoms.size(), count = hvp ? coordinates : coordinates * coordinates;
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
  if (invocation.response_fp32_proposal)
    policy.cpks.resident_precision = response::CudaCpksResidentPrecision::fp32_proposal;
  dft::CudaKsWb97mvHessian endpoint;
  using Mode = dft::CudaKsWb97mvHessianMode;
  const auto prepared = endpoint.prepare(
      *batch, 0, token, hvp ? Mode::action_only : Mode::dense_and_action, policy, detail);
  if (prepared != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error("native Hessian preparation failed: " + detail);
  const auto hessian_prepared = std::chrono::steady_clock::now();
  dft::CudaKsWb97mvHessianResult result;
  const auto error = hvp ? endpoint.apply(storage.direction, storage.output, storage.stream, result)
                         : endpoint.hessian(storage.output, storage.stream, result);
  if (error == cudaErrorInvalidValue && result.failure_check)
    throw std::runtime_error(result.failure_check);
  check(error);
  if (!result.converged())
    throw std::runtime_error("native stationary response did not converge; no Hessian published");
  std::vector<double> values(count);
  check(cudaMemcpyAsync(values.data(), storage.output, count * sizeof(double),
                        cudaMemcpyDeviceToHost, storage.stream));
  check(cudaStreamSynchronize(storage.stream));
  const auto hessian_completed = std::chrono::steady_clock::now();
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
      {"scf_requested_precision",
       options.precision == GENERATIVEQC_PRECISION_AUTO ? "auto" : "fp64"},
      {"scf_strict_physical_fock", token.strict_physical_fock},
      {"scf_mixed_fock_builds", solved[0].calculation.precision.mixed_stage_fock_builds},
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
      {"cpks_operator_bytes", work.cpks_operator_bytes},
      {"cpks_operator_columns", work.cpks_operator_columns},
      {"cpks_cached_actions", work.cpks_cached_actions},
      {"cpks_physical_checks", work.cpks_physical_checks},
      {"cpks_proposal_iterations", work.cpks_proposal_iterations},
      {"cpks_proposal_seeds", work.cpks_proposal_seeds},
      {"cpks_strict_iterations", work.cpks_strict_iterations},
      {"cpks_cache_precision",
       work.cpks_operator_bytes ? Json(invocation.response_fp32_proposal ? "fp32-proposal" : "fp64")
                                : Json(nullptr)},
      {"induced_resident_integral_bytes", work.induced.resident_integral_bytes},
      {"induced_resident_source_evaluations", work.induced.resident_source_evaluations},
      {"induced_response_eris", work.induced.response_eris},
      {"induced_response_quartets", work.induced.response_quartets},
      {"exact_primitive_components", work.exact.component_evaluations},
      {"exact_integral_source_builds", work.exact.integral_source_builds},
      {"exact_resident_integral_hits", work.exact.resident_integral_hits},
      {"exact_resident_integral_bytes", work.exact.resident_integral_bytes},
      {"exact_batched_workspace_bytes", work.exact.batched_workspace_bytes},
      {"exact_retained_window_source_bytes", work.exact.retained_window_source_bytes},
      {"exact_streaming_tiles", work.exact.streaming_tiles},
      {"exact_source_groups", work.exact.source_groups},
      {"exact_frozen_cache_bytes", work.frozen_exact_bytes},
      {"exact_frozen_columns", work.frozen_exact_columns},
      {"exact_frozen_contractions", work.frozen_exact_contractions},
      {"exact_integral_launches", work.exact.integral_launches},
      {"exact_projection_launches", work.exact.projection_launches},
      {"exact_reduction_launches", work.exact.reduction_launches},
      {"one_electron_primitive_components", work.one_electron.integrals.component_evaluations},
      {"one_electron_frozen_cache_bytes", work.frozen_core_bytes},
      {"one_electron_frozen_columns", work.frozen_core_columns},
      {"one_electron_frozen_contractions", work.frozen_core_contractions},
      {"df_value_passes", work.raw_df.value_passes + work.raw_df_range.value_passes},
      {"df_contraction_passes",
       work.raw_df.contraction_passes + work.raw_df_range.contraction_passes},
      {"df_range_value_passes", work.raw_df_range.value_passes},
      {"df_range_contraction_passes", work.raw_df_range.contraction_passes},
      {"df_range_workspace_bytes", work.raw_df_range.workspace_bytes},
      {"grid_right_directions", work.grid.nuclear_directions},
      {"grid_mixed_contractions", work.grid.nuclear_contractions},
      {"grid_ao_adjoint_builds", work.grid.nuclear_adjoint_builds},
      {"grid_ao_adjoint_contractions", work.grid.nuclear_adjoint_contractions},
      {"grid_nuclear_affine_terms", work.grid.nuclear_affine_terms},
      {"grid_geometry_workspace_bytes",
       work.geometry.workspace_bytes + work.vv10_geometry.workspace_bytes},
      {"grid_materialized_ao_jet_values",
       work.geometry.action_ao_jet_values + work.vv10_geometry.action_ao_jet_values},
      {"grid_ao_prepared_primitive_term_bound", work.geometry.prepared_ao_primitive_term_bound},
      {"grid_ao_replayed_primitive_term_bound", work.geometry.replayed_ao_primitive_term_bound},
      {"scf_domain_version", token.identity.model.scf_domain_version},
      {"correction_directions", work.correction_directions},
      {"grid_nuclear_graph_replays", work.grid.nuclear_graph_replays},
      {"separate_vv10_grid", token.identity.model.nonlocal_grid.has_value()},
      {"vv10_reference_pairs", work.induced.grid.reference_pairs + work.grid.reference_pairs +
                                   work.induced.separate_vv10.reference_pairs +
                                   work.separate_vv10.reference_pairs},
      {"separate_vv10_right_directions", work.separate_vv10.nuclear_directions},
      {"native_execution_seconds", seconds},
      {"scf_seconds", std::chrono::duration<double>(scf_completed - scf_started).count()},
      {"response_preparation_seconds",
       std::chrono::duration<double>(hessian_prepared - scf_completed).count()},
      {"hessian_seconds",
       std::chrono::duration<double>(hessian_completed - hessian_prepared).count()},
      {"scf_iterations", solved[0].calculation.convergence.iterations},
      {"response_iterations", result.response.gmres.iterations},
      {"ecp_ad_backend", work.ecp.workspace_bytes ? Json("cuda") : Json(nullptr)},
      {"ecp_radial_layers", work.ecp.radial_layers},
      {"ecp_ao_evaluations", work.ecp.ao_evaluations},
      {"ecp_projection_terms", work.ecp.projection_terms},
      {"ecp_affine_terms", work.ecp.affine_terms},
      {"ecp_validation_local_attempts", work.ecp.validation.local_attempts},
      {"ecp_validation_nonlocal_attempts", work.ecp.validation.nonlocal_attempts},
      {"ecp_validation_passed_actions", work.ecp.validation.passed_actions},
      {"ecp_validation_radial_layers", work.ecp.validation.radial_layers},
      {"ecp_validation_ao_evaluations", work.ecp.validation.ao_evaluations},
      {"ecp_validation_projection_terms", work.ecp.validation.projection_terms},
      {"ecp_validation_affine_terms", work.ecp.validation.affine_terms},
      {"ecp_validation_reduction_terms_bound", work.ecp.validation.reduction_terms_bound},
      {"df_integral_ad_backend", work.raw_df.workspace_bytes ? Json("cuda") : Json(nullptr)}};
  return {{"schema", "generativeqc.native-hessian-result/v1"},
          {"operation", invocation.operation},
          {"units", "hartree/bohr^2"},
          {"backend", "cuda"},
          {"method", invocation.range_pbe ? invocation.range_pbe->identifier : options.method_name},
          {"method_ir_identity", identity},
          {"native_source_identity", generativeqc_get_source_identity()},
          {"charge", options.charge},
          {"multiplicity", options.multiplicity},
          {"energy_hartree", solved[0].calculation.energy},
          {"diagnostics", std::move(diagnostics)},
          {motion.empty() ? "matrix" : "value", std::move(panel)}};
}
#endif
}  // namespace

bool parse_cpu_hessian_byte_limit(HessianInvocation& invocation, std::string_view option,
                                  std::string_view value) {
  constexpr std::pair<std::string_view, std::uint64_t HessianInvocation::*> limits[]{
      {"--integral-budget-bytes", &HessianInvocation::integral_bytes},
      {"--grid-budget-bytes", &HessianInvocation::grid_bytes},
      {"--output-budget-bytes", &HessianInvocation::output_bytes},
      {"--response-budget-bytes", &HessianInvocation::response_bytes},
      {"--numeric-budget-bytes", &HessianInvocation::numeric_bytes},
      {"--ecp-budget-bytes", &HessianInvocation::ecp_bytes},
      {"--dispersion-budget-bytes", &HessianInvocation::dispersion_bytes},
      {"--cpu-packed-eri-budget-bytes", &HessianInvocation::cpu_packed_eri_bytes}};
  for (const auto& [flag, member] : limits) {
    if (flag != option) continue;
    invocation.*member = positive_integer(value, option);
    // The four generic source caps are also accepted by CUDA Hessian.
    if (member == &HessianInvocation::numeric_bytes || member == &HessianInvocation::ecp_bytes ||
        member == &HessianInvocation::dispersion_bytes ||
        member == &HessianInvocation::cpu_packed_eri_bytes)
      invocation.cpu_controls_explicit = true;
    return true;
  }
  return false;
}

const method_generated::Method* linked_cpu_hessian_program(const RunOptions& options) {
  const auto* program = options.composition ? options.composition
                                            : methods::named_ks::find_method(options.method_name);
  if (!program)
    program = methods::named_ks::find_method(semilocal_catalog_selector(options.method_name));
  return generativeqc_ks_hessian_cpu_version_v1() == 1 && program &&
                 generativeqc_ks_hessian_cpu_method_v1(program->identity.data())
             ? program
             : nullptr;
}

System make_cpu_hessian_system(Context& context, const RunOptions& options,
                               std::span<const generativeqc_atom> atoms,
                               const NativeBasisData& basis) {
  const generativeqc_system_descriptor geometry{sizeof(geometry),
                                                GENERATIVEQC_ABI_VERSION,
                                                atoms.data(),
                                                static_cast<std::uint32_t>(atoms.size()),
                                                basis.shells.data(),
                                                static_cast<std::uint32_t>(basis.shells.size()),
                                                basis.primitives.data(),
                                                static_cast<std::uint32_t>(basis.primitives.size()),
                                                options.charge,
                                                options.multiplicity,
                                                basis.representation};
  return make_molecular_system(context, geometry, basis);
}

generativeqc_ks_host_hessian_options_v1 cpu_hessian_controls(
    const RunOptions& options, std::span<const generativeqc_atom> atoms,
    const HessianInvocation& invocation) {
  auto controls = api::default_host_hessian_options();
  controls.expected_atoms = atoms.data();
  controls.expected_atom_count = atoms.size();
  controls.expected_charge = options.charge;
  controls.expected_multiplicity = options.multiplicity;
  controls.numeric_bytes = invocation.numeric_bytes;
  controls.integral_bytes = invocation.integral_bytes;
  controls.grid_bytes = invocation.grid_bytes;
  controls.response_bytes = invocation.response_bytes;
  controls.gmres_bytes = invocation.response_bytes;
  controls.ecp_bytes = invocation.ecp_bytes;
  controls.dispersion_bytes = invocation.dispersion_bytes;
  controls.output_bytes = invocation.output_bytes;
  controls.max_eri_primitives = invocation.max_eri_primitives;
  controls.max_map_terms = invocation.max_map_terms;
  controls.max_d3_pairs = invocation.max_d3_pairs;
  controls.relative_tolerance = invocation.response_rtol;
  controls.absolute_tolerance =
      invocation.response_atol_explicit ? invocation.response_atol : 1e-12;
  controls.restart = invocation.response_restart;
  controls.max_iterations = invocation.response_iterations;
  return controls;
}

generativeqc_status cpu_hessian_minimum_bytes(
    generativeqc_context* context, const generativeqc_system* system, const char* method,
    const HessianInvocation& invocation, const generativeqc_ks_host_hessian_options_v1& controls,
    std::size_t* minimum) {
  if (invocation.hvp_directions)
    return generativeqc_ks_hvp_cpu_batch_minimum_bytes_v1(
        context, system, method, invocation.hvp_directions, &controls, minimum);
  return generativeqc_ks_hessian_cpu_minimum_bytes_v1(
      context, system, method, invocation.operation == "hessian", &controls, minimum);
}

HessianInvocation parse_hessian_invocation(int argc, char** argv) {
  if (argc < 3) throw UsageError("hessian/hvp requires an XYZ input path");
  HessianInvocation invocation;
  invocation.operation = argv[1];
  invocation.run_arguments = {argv[0], invocation.operation, argv[2], "--method", "wb97m-v-rks"};
  std::optional<std::string> method_ir_file;
  bool named_method = false;
  const std::set<std::string_view> run_options{"--method",
                                               "--basis",
                                               "--representation",
                                               "--density-fitting",
                                               "--auxiliary-basis",
                                               "--dispersion-memory-bytes",
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
                                               "--grid-level",
                                               "--grid-radial-points",
                                               "--grid-polar-points",
                                               "--grid-azimuth-points",
                                               "--host-bytes",
                                               "--host-reserve-bytes",
                                               "--headroom-fraction"};
  for (int index = 3; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--json") {
      invocation.run_arguments.emplace_back(option);
      continue;
    }
    if (option == "--response-sequential") {
      invocation.response_sequential = true;
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
      if (option == "--method") named_method = true;
      invocation.run_arguments.emplace_back(option);
      invocation.run_arguments.emplace_back(value);
    } else if (option == "--method-ir") {
      if (method_ir_file) throw UsageError("--method-ir can only be specified once");
      method_ir_file = value;
    } else if (option == "--direction")
      invocation.direction_file = value;
    else if (option == "--directions")
      invocation.directions_file = value;
    else if (option == "--device-budget-bytes") {
      invocation.device_bytes = positive_integer(value, option);
      invocation.cuda_controls_explicit = true;
    } else if (parse_cpu_hessian_byte_limit(invocation, option, value)) {
      continue;
    } else if (option == "--max-eri-primitives" || option == "--max-d3-pairs" ||
               option == "--max-map-terms") {
      invocation.cpu_controls_explicit = true;
      const auto parsed = positive_integer(value, option);
      if (option == "--max-eri-primitives")
        invocation.max_eri_primitives = parsed;
      else if (option == "--max-map-terms")
        invocation.max_map_terms = parsed;
      else
        invocation.max_d3_pairs = parsed;
    } else if (option == "--max-nonlocal-pairs") {
      invocation.max_nonlocal_pairs = positive_integer(value, option);
      invocation.cuda_controls_explicit = true;
    } else if (option == "--nonlocal-grid-radial-points")
      invocation.nonlocal_radial = positive_integer(value, option, UINT32_MAX);
    else if (option == "--nonlocal-grid-polar-points")
      invocation.nonlocal_polar = positive_integer(value, option, UINT32_MAX);
    else if (option == "--nonlocal-grid-azimuth-points")
      invocation.nonlocal_azimuth = positive_integer(value, option, UINT32_MAX);
    else if (option == "--nonlocal-grid-level") {
      if (value.size() != 1 || value.front() < '0' || value.front() > '9')
        throw UsageError("nonlocal grid level must be in 0..9");
      invocation.nonlocal_level = static_cast<std::uint32_t>(value.front() - '0');
    } else if (option == "--response-atol") {
      invocation.response_atol = nonnegative_number(value, option);
      invocation.response_atol_explicit = true;
    } else if (option == "--response-rtol") {
      invocation.response_rtol = nonnegative_number(value, option);
      if (invocation.response_rtol >= 1) throw UsageError("--response-rtol must be less than 1");
    } else if (option == "--response-restart")
      invocation.response_restart = positive_integer(value, option, INT32_MAX);
    else if (option == "--response-max-iterations")
      invocation.response_iterations = positive_integer(value, option, INT32_MAX);
    else if (option == "--response-recycle-block-size") {
      invocation.response_recycle_block_size = positive_integer(value, option, 4);
      if (invocation.response_recycle_block_size < 2)
        throw UsageError("--response-recycle-block-size requires a value from two to four");
    } else if (option == "--response-cache-precision") {
      if (value != "fp64" && value != "fp32-proposal")
        throw UsageError("--response-cache-precision requires fp64 or fp32-proposal");
      invocation.response_fp32_proposal = value == "fp32-proposal";
    } else
      throw UsageError("unsupported hessian/hvp option: " + std::string(option));
  }
  if (!invocation.direction_file.empty() && !invocation.directions_file.empty())
    throw UsageError("--direction and --directions are mutually exclusive");
  if ((invocation.operation == "hvp") !=
      (!invocation.direction_file.empty() || !invocation.directions_file.empty()))
    throw UsageError("only hvp requires --direction FILE.json or --directions FILE.json");
  if (invocation.operation == "hvp" && invocation.response_fp32_proposal)
    throw UsageError("fp32-proposal response cache requires a dense hessian");
  if (invocation.nonlocal_level &&
      (invocation.nonlocal_radial || invocation.nonlocal_polar || invocation.nonlocal_azimuth))
    throw UsageError("--nonlocal-grid-level cannot be combined with --nonlocal-grid-*-points");
  if ((invocation.nonlocal_radial || invocation.nonlocal_polar || invocation.nonlocal_azimuth) &&
      !(invocation.nonlocal_radial && invocation.nonlocal_polar && invocation.nonlocal_azimuth))
    throw UsageError(
        "an independent nonlocal grid requires all three --nonlocal-grid-*-points flags");
  if (method_ir_file) {
    if (named_method) throw UsageError("--method-ir and --method are mutually exclusive");
    invocation.range_pbe = read_range_pbe_method_ir(*method_ir_file);
    invocation.run_arguments[4] = invocation.range_pbe->spins == 1 ? "pbe-rks" : "pbe-uks";
  }
  return invocation;
}
int hessian_command(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                    HessianInvocation invocation) {
  const bool cpu = options.backend == GENERATIVEQC_BACKEND_CPU_REFERENCE;
  if (!invocation.directions_file.empty() && !cpu)
    throw UsageError("--directions requires linked CPU hvp");
  if (options.host_bytes && !cpu)
    throw UsageError("--host-bytes for Hessian/HVP requires --backend cpu");
  if (!options.host_bytes && (options.host_reserve_bytes || options.headroom_fraction))
    throw UsageError("Hessian/HVP host reserve and headroom require --host-bytes");
  if (invocation.response_recycle_block_size && (!cpu || invocation.operation != "hessian"))
    throw UsageError("--response-recycle-block-size requires a dense CPU hessian");
  if (invocation.response_sequential && (!cpu || invocation.operation != "hessian"))
    throw UsageError("--response-sequential requires a dense CPU hessian");
  if (invocation.response_sequential && invocation.response_recycle_block_size)
    throw UsageError(
        "--response-sequential and --response-recycle-block-size are mutually exclusive");
  // Parsing uses PBE as a carrier for range-PBE, not as a CPU method substitute.
  if (cpu && invocation.range_pbe)
    throw UsageError("range-PBE MethodIR is not supported by native CPU hessian/hvp");
  if (!cpu) {
    if (!is_wb97mv(options.method_name) && semilocal_catalog_selector(options.method_name).empty())
      throw UsageError(
          "method has no qualified native CUDA hessian/hvp; see docs/user/hessians.md");
    if ((invocation.nonlocal_radial || invocation.nonlocal_level) &&
        !is_wb97mv(options.method_name) && !is_r2scan_rvv10(options.method_name))
      throw UsageError("independent nonlocal Hessian grids require WB97M-V or r2SCAN+rVV10");
    if (options.backend != GENERATIVEQC_BACKEND_CUDA)
      throw UsageError("native hessian/hvp requires --backend cuda; no Python/CPU fallback");
    if (invocation.range_pbe && (options.precision != GENERATIVEQC_PRECISION_FP64 ||
                                 options.grid.partition_iterations != 3 ||
                                 (options.density_fitting != GENERATIVEQC_DENSITY_FITTING_NONE &&
                                  options.density_fitting != GENERATIVEQC_DENSITY_FITTING_CUDA)))
      throw UsageError("range-PBE MethodIR requires FP64, Becke-3 and Direct or CUDA DF");
    if (is_split_hybrid(options.method_name) &&
        (options.precision != GENERATIVEQC_PRECISION_FP64 ||
         options.grid.partition_iterations != 3 ||
         (options.density_fitting != GENERATIVEQC_DENSITY_FITTING_NONE &&
          (options.density_fitting != GENERATIVEQC_DENSITY_FITTING_CUDA ||
           !options.auxiliary_basis_explicit))))
      throw UsageError(
          "CUDA split-hybrid Hessian/HVP requires FP64, Becke-3 and Direct or explicit CUDA DF");
    const bool auto_fitted_pbe = options.method_name == "pbe-rks" &&
                                 (options.density_fitting == GENERATIVEQC_DENSITY_FITTING_CUDA ||
                                  options.density_fitting == GENERATIVEQC_DENSITY_FITTING_AUTO);
    if (options.precision != GENERATIVEQC_PRECISION_FP64 &&
        !(options.precision == GENERATIVEQC_PRECISION_AUTO &&
          (auto_fitted_pbe || (options.density_fitting == GENERATIVEQC_DENSITY_FITTING_NONE &&
                               auto_hessian_scf_model(options.method_name)))))
      throw UsageError(
          "native hessian/hvp requires FP64 or an AUTO-qualified electronic method and provider");
  }
  if (!cpu && options.density_fitting == GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE)
    throw UsageError("CUDA hessian/hvp cannot select a CPU density-fitting provider");
  const auto coordinates = 3ULL * atoms.size();
  if (invocation.response_recycle_block_size > coordinates)
    throw UsageError("--response-recycle-block-size exceeds the Cartesian coordinate count");
  if (!coordinates || coordinates > SIZE_MAX / coordinates / sizeof(double))
    throw UsageError("Hessian dimensions overflow");
  const auto count = invocation.operation == "hvp" ? coordinates : coordinates * coordinates;
  if (count > invocation.output_bytes / (2 * sizeof(double)))
    throw UsageError("result and publication exceed --output-budget-bytes");
  std::vector<double> motion;
  if (!invocation.directions_file.empty()) {
    motion = read_directions(invocation.directions_file, atoms.size(), invocation.output_bytes);
    invocation.hvp_directions = motion.size() / coordinates;
  } else if (invocation.operation == "hvp") {
    motion = read_direction(invocation.direction_file, atoms.size());
  }
  if (cpu) {
    const auto result = execute_cpu(options, atoms, invocation, motion);
    if (options.json)
      std::cout << result.dump() << '\n';
    else {
      std::cout << options.method_name << ' ' << invocation.operation
                << " (hartree/bohr^2; native compiler AD, CPU)\n";
      for (const auto& row : result.at(invocation.operation == "hvp" ? "value" : "matrix"))
        std::cout << row.dump() << '\n';
    }
    return 0;
  }
  if (invocation.cpu_controls_explicit)
    throw UsageError("CUDA Hessian does not use CPU numeric/ECP/D3 work budgets");
#if GENERATIVEQC_CLI_NATIVE_HESSIAN
  const auto result = execute_native(options, atoms, invocation, motion);
  if (options.json)
    std::cout << result.dump() << '\n';
  else {
    std::cout << options.method_name << ' ' << invocation.operation
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

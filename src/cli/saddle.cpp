#include "cli/saddle.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>

#include "api/ks_batch_capabilities.hpp"
#include "api/ks_host_hessian.hpp"
#include "cli/molecular_evaluator.hpp"
#include "cli/native_basis.hpp"
#include "geometry/harmonic_modes.hpp"
#include "geometry/saddle_optimizer.hpp"

namespace generativeqc::cli {
namespace {
using Clock = std::chrono::steady_clock;
using molecular_detail::rows;

std::uint64_t integer(std::string_view text, std::string_view option,
                      std::uint64_t maximum = INT64_MAX) {
  std::uint64_t result{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !result ||
      result > maximum)
    throw UsageError(std::string(option) + " requires a positive bounded integer");
  return result;
}

double positive(std::string_view text, std::string_view option) {
  std::size_t consumed{};
  double result{};
  try {
    result = std::stod(std::string(text), &consumed);
  } catch (...) {
    throw UsageError(std::string(option) + " requires a finite positive number");
  }
  if (consumed != text.size() || !std::isfinite(result) || result <= 0)
    throw UsageError(std::string(option) + " requires a finite positive number");
  return result;
}

std::vector<double> coordinates_of(const std::vector<generativeqc_atom>& atoms) {
  std::vector<double> result;
  for (const auto& atom : atoms) result.insert(result.end(), {atom.x, atom.y, atom.z});
  return result;
}

std::vector<double> read_masses(const std::string& path, std::size_t atoms) {
  const auto input = read_json(path);
  if (!input.is_array() || input.size() != atoms)
    throw UsageError("--masses requires one positive finite dalton mass per ordered atom");
  std::vector<double> result;
  for (const auto& value : input) {
    if (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() <= 0)
      throw UsageError("--masses requires one positive finite dalton mass per ordered atom");
    result.push_back(value.get<double>());
  }
  return result;
}

std::vector<double> read_mode(const std::string& path, std::size_t atoms) {
  if (path.empty()) return {};
  const auto input = read_json(path);
  if (!input.is_array() || input.size() != atoms)
    throw UsageError("--initial-mode requires a finite natoms-by-3 array");
  std::vector<double> result;
  for (const auto& row : input) {
    if (!row.is_array() || row.size() != 3)
      throw UsageError("--initial-mode requires a finite natoms-by-3 array");
    for (const auto& value : row) {
      if (!value.is_number() || !std::isfinite(value.get<double>()))
        throw UsageError("--initial-mode requires a finite natoms-by-3 array");
      result.push_back(value.get<double>());
    }
  }
  if (std::all_of(result.begin(), result.end(), [](double value) { return value == 0; }))
    throw UsageError("--initial-mode must be nonzero");
  return result;
}

std::string_view status_name(std::uint32_t status) {
  switch (status) {
    case GENERATIVEQC_SADDLE_FIRST_ORDER:
      return "first_order_saddle";
    case GENERATIVEQC_SADDLE_OTHER_INDEX:
      return "other_index";
    case GENERATIVEQC_SADDLE_AMBIGUOUS_CURVATURE:
      return "ambiguous_curvature";
    case GENERATIVEQC_SADDLE_ATTEMPT_LIMIT:
      return "attempt_limit";
    case GENERATIVEQC_SADDLE_TARGET_NOT_NEGATIVE:
      return "target_not_negative";
    case GENERATIVEQC_SADDLE_MODE_TRACKING_LOST:
      return "mode_tracking_lost";
    case GENERATIVEQC_SADDLE_TRUST_EXHAUSTED:
      return "trust_radius_exhausted";
    case GENERATIVEQC_SADDLE_EVALUATION_FAILED:
      return "evaluation_failed";
    case GENERATIVEQC_SADDLE_HESSIAN_FAILED:
      return "hessian_failed";
    case GENERATIVEQC_SADDLE_CANCELLED:
      return "cancelled";
    default:
      throw std::logic_error("unknown saddle termination");
  }
}

Json optimizer_receipt(const generativeqc_saddle_receipt_v1& receipt) {
  return {{"status", status_name(receipt.status)},
          {"has_valid_state", receipt.has_valid_state != 0},
          {"stationary_verified", receipt.stationary_verified != 0},
          {"attempts", receipt.attempts},
          {"accepted_steps", receipt.accepted_steps},
          {"energy_gradient_calls", receipt.energy_gradient_calls},
          {"hessian_calls", receipt.hessian_calls},
          {"negative_curvatures", receipt.negative_curvatures},
          {"near_zero_curvatures", receipt.near_zero_curvatures},
          {"rigid_modes", receipt.rigid_modes},
          {"active_modes", receipt.active_modes},
          {"numeric_capacity_bytes", receipt.numeric_capacity_bytes},
          {"energy_change", receipt.energy_change},
          {"gradient_rms", receipt.gradient_rms},
          {"gradient_max", receipt.gradient_max},
          {"displacement_rms_angstrom", receipt.displacement_rms_angstrom},
          {"displacement_max_angstrom", receipt.displacement_max_angstrom},
          {"trust_radius_bohr", receipt.trust_radius},
          {"target_curvature", receipt.target_curvature},
          {"target_overlap", receipt.target_overlap}};
}

volatile std::sig_atomic_t interrupted = 0;
void interrupt_handler(int) { interrupted = 1; }

/** Convert ordinary process cancellation into the corrector's failure receipt.
 * Native derivative stages finish synchronously before the next cancellation
 * check; a signal never bypasses source lifetimes or publishes a partial stage.
 */
struct SignalScope {
  using Handler = void (*)(int);
  Handler previous_int, previous_term;
  SignalScope()
      : previous_int(std::signal(SIGINT, interrupt_handler)),
        previous_term(std::signal(SIGTERM, interrupt_handler)) {
    if (previous_int == SIG_ERR || previous_term == SIG_ERR) {
      if (previous_int != SIG_ERR) std::signal(SIGINT, previous_int);
      if (previous_term != SIG_ERR) std::signal(SIGTERM, previous_term);
      throw std::runtime_error("cannot install native saddle cancellation handlers");
    }
    interrupted = 0;
  }
  ~SignalScope() {
    std::signal(SIGINT, previous_int);
    std::signal(SIGTERM, previous_term);
  }
};

/** Public CLI policy and optimizer callbacks; numerical ownership is shared. */
class MolecularEvaluator : public NativeMolecularEvaluator {
 public:
  MolecularEvaluator(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                     std::span<const double> masses, const SaddleInvocation& invocation,
                     const method_generated::Method& program)
      : NativeMolecularEvaluator(options, atoms, masses, invocation, program,
                                 expand_basis_input(options.basis_name, atoms,
                                                    options.representation_explicit
                                                        ? std::optional(options.representation)
                                                        : std::nullopt)),
        invocation_(invocation) {
    if (uses_ecp())
      throw UsageError("native molecular saddle ECP requires separate real-element qualification");
    failure.reserve(4096);
  }

  static int evaluate(void* user, std::uint32_t stage, const double* coordinates, std::size_t count,
                      double* energy, double* gradient, double* hessian) noexcept {
    auto& self = *static_cast<MolecularEvaluator*>(user);
    const std::span<const double> positions(coordinates, count);
    const std::string_view name = stage == 0 ? "energy_gradient" : "hessian";
    try {
      self.record(name, "start", positions);
      if (stage == 0)
        self.energy_gradient(positions, *energy, {gradient, count});
      else if (stage == 1)
        self.hessian(positions, {hessian, count * count});
      else
        throw std::invalid_argument("invalid derivative stage");
      return 0;
    } catch (const std::exception& error) {
      self.failure.assign(error.what(), std::min(std::strlen(error.what()), std::size_t{4095}));
    } catch (...) {
      self.failure = "unknown native derivative failure";
    }
    self.discard_current();
    try {
      self.record(name, "failed", positions, {{"detail", self.failure}});
    } catch (...) {
    }
    return 1;
  }

  static int progress(void* user, const generativeqc_saddle_receipt_v1* receipt,
                      const double* coordinates, std::size_t count) noexcept {
    auto& self = *static_cast<MolecularEvaluator*>(user);
    try {
      if (receipt->has_valid_state) self.accept_geometry({coordinates, count});
      auto event = optimizer_receipt(*receipt);
      event["coordinates_bohr"] = rows({coordinates, count}, 3);
      event["elapsed_seconds"] = self.elapsed();
      self.observations.push_back(event);
      if (self.invocation_.progress_json) {
        event["stage"] = "optimizer";
        std::cerr << event.dump() << '\n';
      }
      return interrupted != 0;
    } catch (...) {
      return 1;
    }
  }

  static int cancelled(void*) noexcept { return interrupted != 0; }

  Json observations = Json::array();
  std::string failure;

 private:
  const SaddleInvocation& invocation_;
};
}  // namespace

SaddleInvocation parse_saddle_invocation(int argc, char** argv) {
  if (argc < 3) throw UsageError("saddle requires INPUT.xyz and --masses FILE.json");
  SaddleInvocation result;
  const geometry::saddle::Options defaults;
  result.controls = {sizeof(result.controls),
                     GENERATIVEQC_ABI_VERSION,
                     1,
                     defaults.max_attempts,
                     defaults.max_coordinates,
                     128ULL << 20,
                     defaults.energy_change,
                     defaults.gradient_rms,
                     defaults.gradient_max,
                     defaults.displacement_rms_angstrom,
                     defaults.displacement_max_angstrom,
                     defaults.initial_radius,
                     defaults.minimum_radius,
                     defaults.maximum_radius,
                     defaults.curvature_tolerance,
                     defaults.raw_symmetry_tolerance,
                     defaults.minimum_mode_overlap,
                     defaults.acceptance_ratio};
  std::vector<std::string> forwarded{argv[0], "hessian", argv[2], "--method", "b3lyp-rks"};
  for (int index = 3; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--progress-json") {
      result.progress_json = true;
      continue;
    }
    if (option == "--json") {
      forwarded.emplace_back(option);
      continue;
    }
    if (++index >= argc) throw UsageError(std::string(option) + " requires a value");
    const std::string_view value = argv[index];
    auto& controls = result.controls;
    if (option == "--masses")
      result.masses_file = value;
    else if (option == "--initial-mode")
      result.initial_mode_file = value;
    else if (option == "--max-attempts")
      controls.max_attempts = integer(value, option, 100);
    else if (option == "--max-coordinates")
      controls.max_coordinates = integer(value, option, 300);
    else if (option == "--saddle-budget-bytes")
      controls.maximum_numeric_bytes = integer(value, option);
    else if (option == "--root-budget-bytes")
      result.root_bytes = integer(value, option);
    else if (option == "--energy-change")
      controls.energy_change = positive(value, option);
    else if (option == "--gradient-rms")
      controls.gradient_rms = positive(value, option);
    else if (option == "--gradient-max")
      controls.gradient_max = positive(value, option);
    else if (option == "--displacement-rms")
      controls.displacement_rms_angstrom = positive(value, option);
    else if (option == "--displacement-max")
      controls.displacement_max_angstrom = positive(value, option);
    else if (option == "--initial-radius")
      controls.initial_radius = positive(value, option);
    else if (option == "--minimum-radius")
      controls.minimum_radius = positive(value, option);
    else if (option == "--maximum-radius")
      controls.maximum_radius = positive(value, option);
    else if (option == "--curvature-tolerance")
      controls.curvature_tolerance = positive(value, option);
    else if (option == "--raw-symmetry-tolerance")
      controls.raw_symmetry_tolerance = positive(value, option);
    else if (option == "--minimum-mode-overlap")
      controls.minimum_mode_overlap = positive(value, option);
    else if (option == "--acceptance-ratio")
      controls.acceptance_ratio = positive(value, option);
    else if (option == "--minimum-occupied-overlap")
      result.minimum_occupied_overlap = positive(value, option);
    else {
      forwarded.emplace_back(option);
      forwarded.emplace_back(value);
    }
  }
  if (result.masses_file.empty()) throw UsageError("saddle requires --masses FILE.json (dalton)");
  const auto& controls = result.controls;
  if (controls.minimum_radius > controls.initial_radius ||
      controls.initial_radius > controls.maximum_radius || controls.minimum_mode_overlap > 1 ||
      controls.acceptance_ratio >= 1 || result.minimum_occupied_overlap > 1)
    throw UsageError("invalid saddle trust radii, acceptance ratio or overlap controls");
  std::vector<char*> arguments;
  for (auto& argument : forwarded) arguments.push_back(argument.data());
  result.derivatives = parse_hessian_invocation(arguments.size(), arguments.data());
  result.derivatives.run_arguments[1] = "saddle";
  return result;
}

int saddle_command(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                   const SaddleInvocation& invocation) {
  const auto started = Clock::now();
  const auto* program = options.composition ? options.composition
                                            : methods::named_ks::find_method(options.method_name);
  if (options.backend != GENERATIVEQC_BACKEND_CPU_REFERENCE || !program ||
      generativeqc_ks_hessian_cpu_version_v1() != 1 ||
      !generativeqc_ks_hessian_cpu_method_v1(program->identity.data()))
    throw UsageError(
        "molecular saddle requires a linked native CPU B3LYP RKS/UKS MethodIR; no fallback");
  if (options.precision != GENERATIVEQC_PRECISION_FP64 ||
      options.density_fitting != GENERATIVEQC_DENSITY_FITTING_NONE ||
      options.auxiliary_basis_explicit)
    throw UsageError("molecular saddle requires direct FP64 without an auxiliary basis");
  if (invocation.derivatives.cuda_controls_explicit)
    throw UsageError("molecular saddle does not use CUDA device/nonlocal budgets");
  const auto count = 3 * atoms.size();
  if (!count || count > invocation.controls.max_coordinates)
    throw UsageError("saddle geometry exceeds --max-coordinates");
  if (sizeof(double) * (8 * count * count + 16 * count) > invocation.derivatives.output_bytes)
    throw UsageError("saddle result panels exceed --output-budget-bytes");
  const auto masses = read_masses(invocation.masses_file, atoms.size());
  const auto initial_mode = read_mode(invocation.initial_mode_file, atoms.size());
  const auto initial = coordinates_of(atoms);
  const auto physical_subspace = geometry::rigid_subspace(initial, 1e-8, masses);
  if (!physical_subspace.active_modes)
    throw UsageError("molecular saddle requires at least one vibrational coordinate");
  std::vector<double> coordinates(count), gradient(count), hessian(count * count), mode(count),
      curvatures(count);
  generativeqc_saddle_receipt_v1 receipt{};
  receipt.struct_size = sizeof(receipt);
  receipt.abi_version = GENERATIVEQC_ABI_VERSION;
  char detail[4096]{};
  MolecularEvaluator evaluator(options, atoms, masses, invocation, *program);
  SignalScope signals;
  const auto status = generativeqc_saddle_correct_v1(
      &invocation.controls, initial.data(), count,
      initial_mode.empty() ? nullptr : initial_mode.data(), MolecularEvaluator::evaluate,
      MolecularEvaluator::progress, MolecularEvaluator::cancelled, &evaluator, &receipt,
      coordinates.data(), gradient.data(), hessian.data(), mode.data(), curvatures.data(), detail,
      sizeof(detail));
  if (status != GENERATIVEQC_STATUS_SUCCESS)
    throw Error(status, std::string("native saddle argument/resource refusal: ") + detail);
  bool converged = receipt.status == GENERATIVEQC_SADDLE_FIRST_ORDER && receipt.stationary_verified;
  Json harmonics = nullptr, harmonic_failure = nullptr;
  if (receipt.stationary_verified) {
    try {
      const auto analyzed = geometry::harmonic_modes(hessian, coordinates, masses,
                                                     invocation.controls.maximum_numeric_bytes,
                                                     invocation.controls.raw_symmetry_tolerance);
      if (analyzed.rigid_modes != receipt.rigid_modes ||
          (converged && analyzed.negative_modes != receipt.negative_curvatures))
        throw std::runtime_error(
            "harmonic rigid rank or negative index differs from verified saddle");
      harmonics = {
          {"frequencies_cm1", analyzed.frequencies_cm1},
          {"scaled", false},
          {"negative_mode_count", analyzed.negative_modes},
          {"rigid_mode_count", analyzed.rigid_modes},
          {"raw_symmetry_error", analyzed.raw_symmetry_error},
          {"numeric_reservation_bytes", analyzed.numeric_reservation_bytes},
          {"mass_weighted_modes", rows(analyzed.mass_weighted_modes, analyzed.active_modes)},
          {"cartesian_modes", rows(analyzed.cartesian_modes, analyzed.active_modes)}};
    } catch (const std::exception& error) {
      harmonic_failure = error.what();
      converged = false;
    }
  }
  Json result{
      {"schema", "generativeqc.native-molecular-saddle-result/v1"},
      {"converged", converged},
      {"backend", "cpu"},
      {"method", options.method_name},
      {"method_ir_identity", program->identity},
      {"native_source_identity", generativeqc_get_source_identity()},
      {"basis", options.basis_name},
      {"basis_representation",
       evaluator.representation() == GENERATIVEQC_BASIS_SPHERICAL ? "spherical" : "cartesian"},
      {"charge", options.charge},
      {"multiplicity", options.multiplicity},
      {"masses_dalton", masses},
      {"controls",
       {{"max_attempts", invocation.controls.max_attempts},
        {"energy_change_hartree", invocation.controls.energy_change},
        {"gradient_rms_hartree_per_bohr", invocation.controls.gradient_rms},
        {"gradient_max_hartree_per_bohr", invocation.controls.gradient_max},
        {"displacement_rms_angstrom", invocation.controls.displacement_rms_angstrom},
        {"displacement_max_angstrom", invocation.controls.displacement_max_angstrom},
        {"initial_radius_bohr", invocation.controls.initial_radius},
        {"minimum_radius_bohr", invocation.controls.minimum_radius},
        {"maximum_radius_bohr", invocation.controls.maximum_radius},
        {"raw_symmetry_tolerance", invocation.controls.raw_symmetry_tolerance},
        {"curvature_tolerance", invocation.controls.curvature_tolerance},
        {"minimum_mode_overlap", invocation.controls.minimum_mode_overlap},
        {"minimum_occupied_overlap", invocation.minimum_occupied_overlap},
        {"acceptance_ratio", invocation.controls.acceptance_ratio},
        {"scf_max_iterations", options.max_iterations.value_or(240)},
        {"scf_energy_tolerance", options.energy_tolerance.value_or(1e-13)},
        {"scf_density_tolerance", options.density_tolerance.value_or(1e-11)},
        {"screening_tolerance", options.screening_tolerance.value_or(1e-14)},
        {"grid",
         {{"version", options.grid.version},
          {"radial_points", options.grid.radial_points},
          {"angular_polar", options.grid.angular_polar},
          {"angular_azimuth", options.grid.angular_azimuth},
          {"partition_iterations", options.grid.partition_iterations},
          {"coincident_tolerance", options.grid.coincident_tolerance},
          {"tile_points", options.grid.tile_points}}}}},
      {"receipt", optimizer_receipt(receipt)},
      {"detail", detail},
      {"derivative_failure", evaluator.failure},
      {"harmonics", std::move(harmonics)},
      {"harmonic_failure", std::move(harmonic_failure)},
      {"work", std::move(evaluator.work)},
      {"evaluation_records", std::move(evaluator.records)},
      {"optimizer_observations", std::move(evaluator.observations)},
      {"diagnostics",
       {{"numerical_runtime", "native-cpp-cpu"},
        {"runtime_compilation", false},
        {"displaced_scf_evaluations", 0},
        {"preoptimization_stages", 0},
        {"partial_failed_derivative_work_known", false},
        {"reservation_scope",
         "owner-specific numeric panels; excludes primal, provider scratch, JSON and RSS"},
        {"endpoint_seconds", std::chrono::duration<double>(Clock::now() - started).count()}}}};
  result["atomic_numbers"] = Json::array();
  for (const auto& atom : atoms) result["atomic_numbers"].push_back(atom.atomic_number);
  result["coordinates_bohr"] = receipt.has_valid_state ? rows(coordinates, 3) : rows(initial, 3);
  result["energy_hartree"] = receipt.has_valid_state ? Json(receipt.energy) : Json(nullptr);
  result["gradient_hartree_per_bohr"] = receipt.has_valid_state ? rows(gradient, 3) : Json(nullptr);
  result["raw_hessian_hartree_per_bohr2"] =
      receipt.has_valid_state ? rows(hessian, count) : Json(nullptr);
  result["target_mode"] = receipt.has_valid_state ? rows(mode, 3) : Json(nullptr);
  result["cartesian_curvatures_hartree_per_bohr2"] =
      receipt.has_valid_state
          ? Json(std::vector<double>(curvatures.begin(), curvatures.begin() + receipt.active_modes))
          : Json(nullptr);
  if (options.json)
    std::cout << result.dump() << '\n';
  else {
    std::cout << options.method_name << " saddle: " << status_name(receipt.status) << " (attempts "
              << receipt.attempts << ", accepted " << receipt.accepted_steps << ")\n";
    std::cout << result.dump(2) << '\n';
  }
  return converged ? 0 : 1;
}

}  // namespace generativeqc::cli

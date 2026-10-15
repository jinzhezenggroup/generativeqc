#include "cli/hessian.hpp"

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
extern char** environ;
#endif

namespace generativeqc::cli {
namespace {

constexpr std::size_t kProtocolBytes = 64ULL << 20;

std::uint64_t positive_integer(std::string_view text, std::string_view option,
                               std::uint64_t maximum = INT64_MAX) {
  std::uint64_t parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed == 0 ||
      parsed > maximum)
    throw UsageError(std::string(option) + " requires a positive bounded integer");
  return parsed;
}

double nonnegative_number(std::string_view text, std::string_view option) {
  std::size_t consumed = 0;
  double parsed = 0;
  try {
    parsed = std::stod(std::string(text), &consumed);
  } catch (const std::exception&) {
    throw UsageError(std::string(option) + " requires a finite nonnegative number");
  }
  if (consumed != text.size() || !std::isfinite(parsed) || parsed < 0)
    throw UsageError(std::string(option) + " requires a finite nonnegative number");
  return parsed;
}

void check_panel(const Json& panel, std::size_t rows, std::size_t columns) {
  if (!panel.is_array() || panel.size() != rows)
    throw std::runtime_error("compiler Hessian result has the wrong row count");
  for (const auto& row : panel) {
    if (!row.is_array() || row.size() != columns)
      throw std::runtime_error("compiler Hessian result has the wrong column count");
    for (const auto& value : row)
      if (!value.is_number() || !std::isfinite(value.get<double>()))
        throw std::runtime_error(
            "compiler Hessian result contains a nonfinite or nonnumeric value");
  }
}

#if defined(__unix__) || defined(__APPLE__)
using File = std::unique_ptr<std::FILE, decltype(&std::fclose)>;

// Unnamed, private files avoid pipe deadlocks and leave no geometry/result files
// behind on failure. The publication protocol has a separate 64 MiB hard cap.
File temporary_file() {
  File file(std::tmpfile(), &std::fclose);
  if (!file) throw std::runtime_error("cannot allocate private compiler protocol storage");
  return file;
}

std::string linked_library(const HessianInvocation& invocation) {
#if GENERATIVEQC_MONOLITHIC_CLI
  if (invocation.library.empty())
    throw UsageError(
        "a monolithic CLI requires --compiler-library /absolute/libgenerativeqc.so "
        "from the same source build for the explicit compiler-runtime extension");
  return std::filesystem::canonical(invocation.library).string();
#else
  Dl_info loaded{};
  if (!::dladdr(reinterpret_cast<void*>(generativeqc_get_source_identity), &loaded) ||
      !loaded.dli_fname)
    throw std::runtime_error("cannot identify the CLI's loaded native library");
  const auto library = std::filesystem::canonical(loaded.dli_fname);
  if (!invocation.library.empty() && !std::filesystem::equivalent(library, invocation.library))
    throw UsageError("--compiler-library must match the CLI's already-linked native library");
  return library.string();
#endif
}

class Child {
 public:
  explicit Child(pid_t process) : process_(process) {}
  ~Child() {
    if (process_ > 0) {
      // The worker's process group also owns any finite-lifetime JIT children.
      ::kill(-process_, SIGKILL);
      while (::waitpid(process_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
  }
  void wait(std::uint32_t seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    for (;;) {
      int status = 0;
      const auto result = ::waitpid(process_, &status, WNOHANG);
      if (result == process_) {
        const bool succeeded = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (!succeeded) ::kill(-process_, SIGKILL);
        process_ = -1;
        if (!succeeded)
          throw std::runtime_error("compiler Hessian worker failed; no result published");
        return;
      }
      if (result < 0 && errno != EINTR)
        throw std::runtime_error("cannot wait for compiler Hessian worker");
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("compiler Hessian worker timed out; no result published");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

 private:
  pid_t process_;
};

Json execute_worker(const HessianInvocation& invocation, const Json& request) {
  auto input = temporary_file();
  auto output = temporary_file();
  const std::string encoded = request.dump();
  if (encoded.size() > kProtocolBytes ||
      std::fwrite(encoded.data(), 1, encoded.size(), input.get()) != encoded.size() ||
      std::fflush(input.get()) || std::fseek(input.get(), 0, SEEK_SET))
    throw std::runtime_error("cannot stage bounded compiler Hessian request");

  std::vector<std::string> arguments{invocation.interpreter, "-m", "generativeqc.hessian_cli",
                                     "--library", linked_library(invocation)};
  std::vector<char*> pointers;
  for (auto& argument : arguments) pointers.push_back(argument.data());
  pointers.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  if (::posix_spawn_file_actions_init(&actions) != 0)
    throw std::runtime_error("cannot initialize compiler worker file actions");
  if (::posix_spawnattr_init(&attributes) != 0) {
    ::posix_spawn_file_actions_destroy(&actions);
    throw std::runtime_error("cannot initialize compiler worker attributes");
  }
  int status = ::posix_spawn_file_actions_adddup2(&actions, ::fileno(input.get()), STDIN_FILENO);
  status |= ::posix_spawn_file_actions_adddup2(&actions, ::fileno(output.get()), STDOUT_FILENO);
  status |= ::posix_spawn_file_actions_addclose(&actions, ::fileno(input.get()));
  status |= ::posix_spawn_file_actions_addclose(&actions, ::fileno(output.get()));
  status |= ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
  status |= ::posix_spawnattr_setpgroup(&attributes, 0);
  pid_t process = -1;
  if (status == 0)
    status = ::posix_spawn(&process, invocation.interpreter.c_str(), &actions, &attributes,
                           pointers.data(), environ);
  ::posix_spawn_file_actions_destroy(&actions);
  ::posix_spawnattr_destroy(&attributes);
  if (status != 0) throw std::runtime_error("cannot start the explicit compiler Python executable");
  Child child(process);
  child.wait(invocation.timeout_seconds);

  if (std::fseek(output.get(), 0, SEEK_END))
    throw std::runtime_error("cannot inspect compiler Hessian publication");
  const auto size = std::ftell(output.get());
  if (size < 0 || static_cast<unsigned long>(size) > kProtocolBytes)
    throw std::runtime_error("compiler Hessian JSON exceeds the 64 MiB publication limit");
  std::rewind(output.get());
  std::string content(static_cast<std::size_t>(size), '\0');
  if (std::fread(content.data(), 1, content.size(), output.get()) != content.size())
    throw std::runtime_error("cannot read compiler Hessian publication");
  return parse_json(content);
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
    if (++index >= argc) throw UsageError(std::string(option) + " requires a value");
    const std::string_view value = argv[index];
    if (run_options.contains(option)) {
      invocation.run_arguments.emplace_back(option);
      invocation.run_arguments.emplace_back(value);
    } else if (option == "--compiler-python") {
      invocation.interpreter = value;
    } else if (option == "--compiler-library") {
      invocation.library = value;
    } else if (option == "--direction") {
      invocation.direction_file = value;
    } else if (option == "--cache") {
      invocation.cache = value;
    } else if (option == "--timeout-seconds") {
      invocation.timeout_seconds = positive_integer(value, option, UINT32_MAX);
    } else if (option == "--integral-budget-bytes") {
      invocation.integral_bytes = positive_integer(value, option);
    } else if (option == "--grid-budget-bytes") {
      invocation.grid_bytes = positive_integer(value, option);
    } else if (option == "--output-budget-bytes") {
      invocation.output_bytes = positive_integer(value, option);
    } else if (option == "--response-budget-bytes") {
      invocation.response_bytes = positive_integer(value, option);
    } else if (option == "--max-nonlocal-pairs") {
      invocation.max_nonlocal_pairs = positive_integer(value, option);
    } else if (option == "--response-rtol") {
      invocation.response_rtol = nonnegative_number(value, option);
      if (invocation.response_rtol >= 1) throw UsageError("--response-rtol must be less than 1");
    } else if (option == "--response-atol") {
      invocation.response_atol = nonnegative_number(value, option);
    } else if (option == "--response-restart") {
      invocation.response_restart = positive_integer(value, option, INT32_MAX);
    } else if (option == "--response-max-iterations") {
      invocation.response_iterations = positive_integer(value, option, INT32_MAX);
    } else {
      throw UsageError("unsupported hessian/hvp option: " + std::string(option));
    }
  }
  if (invocation.interpreter.empty() ||
      !std::filesystem::path(invocation.interpreter).is_absolute())
    throw UsageError(
        "hessian/hvp requires --compiler-python with an absolute executable path; "
        "this explicit compiler-runtime endpoint requires Python (and CuPy for CUDA)");
  if ((invocation.operation == "hvp") != !invocation.direction_file.empty())
    throw UsageError("only hvp requires --direction FILE.json (finite natoms-by-3 array)");
  if (!invocation.library.empty() && !std::filesystem::path(invocation.library).is_absolute())
    throw UsageError("--compiler-library requires an absolute native shared-library path");
  return invocation;
}

int hessian_command(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                    const HessianInvocation& invocation) {
#if defined(__unix__) || defined(__APPLE__)
  if (options.method_name != "wb97m-v" && options.method_name != "wb97m-v-rks" &&
      options.method_name != "wb97m-v-uks")
    throw UsageError("the compiler hessian/hvp CLI currently admits only WB97M-V RKS/UKS");
  if (options.precision != GENERATIVEQC_PRECISION_FP64)
    throw UsageError("compiler hessian/hvp requires --precision fp64");
  if (options.backend == GENERATIVEQC_BACKEND_CUDA &&
      options.density_fitting == GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE)
    throw UsageError("CUDA hessian/hvp cannot select a CPU density-fitting provider");
  if (::access(invocation.interpreter.c_str(), X_OK) != 0 ||
      !std::filesystem::is_regular_file(invocation.interpreter))
    throw UsageError("--compiler-python is not an executable file");

  Json positions = Json::array();
  for (const auto& atom : atoms)
    positions.push_back({atom.atomic_number, {atom.x, atom.y, atom.z}});
  Json calculator{
      {"method", options.method_name},
      {"basis", options.basis_name},
      {"basis_representation",
       options.representation_explicit
           ? Json(options.representation == GENERATIVEQC_BASIS_SPHERICAL ? "spherical"
                                                                         : "cartesian")
           : Json(nullptr)},
      {"device", options.backend == GENERATIVEQC_BACKEND_CUDA ? "cuda" : "cpu"},
      {"device_id", options.device_id},
      {"density_fitting", options.density_fitting == GENERATIVEQC_DENSITY_FITTING_NONE   ? "none"
                          : options.density_fitting == GENERATIVEQC_DENSITY_FITTING_CUDA ? "cuda"
                          : options.density_fitting == GENERATIVEQC_DENSITY_FITTING_AUTO ? "auto"
                                                                                         : "cpu"},
      {"auxiliary_basis",
       options.auxiliary_basis_explicit ? Json(options.auxiliary_basis_name) : Json(nullptr)},
      {"density_fitting_relative_threshold", options.density_fitting_threshold},
      {"density_fitting_memory_budget_bytes", options.density_fitting_memory_bytes},
      {"max_iterations", options.max_iterations.value_or(100)},
      {"energy_tolerance", options.energy_tolerance.value_or(1e-12)},
      {"density_tolerance", options.density_tolerance.value_or(1e-10)},
      {"screening_tolerance", options.screening_tolerance.value_or(1e-14)},
      {"diis_history", options.diis_history},
      {"precision", "fp64"},
      {"initial_guess", nullptr}};
  Json request{{"schema", "generativeqc.compiler-hessian-request/v1"},
               {"operation", invocation.operation},
               {"atoms_bohr", positions},
               {"calculator", calculator},
               {"charge", options.charge},
               {"multiplicity", options.multiplicity},
               {"native_source_identity", generativeqc_get_source_identity()},
               {"grid",
                {{"radial_points", options.grid.radial_points},
                 {"angular_polar", options.grid.angular_polar},
                 {"angular_azimuth", options.grid.angular_azimuth}}},
               {"controls",
                {{"cache", invocation.cache},
                 {"integral_budget_bytes", invocation.integral_bytes},
                 {"grid_budget_bytes", invocation.grid_bytes},
                 {"max_nonlocal_pairs", invocation.max_nonlocal_pairs}}},
               {"solver",
                {{"rtol", invocation.response_rtol},
                 {"atol", invocation.response_atol},
                 {"restart", invocation.response_restart},
                 {"max_iterations", invocation.response_iterations},
                 {"max_workspace_bytes", invocation.response_bytes}}}};
  if (invocation.operation == "hvp") {
    request["direction"] = read_json(invocation.direction_file);
    check_panel(request["direction"], atoms.size(), 3);
  } else {
    const auto coordinates = 3ULL * atoms.size();
    if (coordinates > invocation.output_bytes / 16 / coordinates)
      throw UsageError("full Hessian and publication exceed --output-budget-bytes");
    request["controls"]["output_budget_bytes"] = invocation.output_bytes;
  }
  const Json result = execute_worker(invocation, request);
  for (const auto* key : {"operation", "native_source_identity", "charge", "multiplicity"})
    if (!result.contains(key) || result.at(key) != request.at(key))
      throw std::runtime_error("compiler Hessian result identity mismatch");
  if (result.at("schema") != "generativeqc.compiler-hessian-result/v1" ||
      result.at("method") != calculator.at("method") ||
      result.at("backend") != calculator.at("device") || result.at("atom_count") != atoms.size() ||
      result.at("units") != "hartree/bohr^2" ||
      result.at("native_library_path") != linked_library(invocation) ||
      !is_digest(result.at("identity").get<std::string>()))
    throw std::runtime_error("compiler Hessian result provenance mismatch");
  const auto& diagnostics = result.at("diagnostics");
  if (diagnostics.at("derivative_method") != "compiler-automatic-differentiation" ||
      diagnostics.at("displaced_scf_evaluations") != 0)
    throw std::runtime_error("compiler Hessian derivative contract mismatch");
  if (options.backend == GENERATIVEQC_BACKEND_CUDA &&
      (diagnostics.at("execution_residency") != "device-resident" ||
       diagnostics.at("numerical_host_fallbacks") != 0 ||
       diagnostics.at("intermediate_panel_downloads") != 0))
    throw std::runtime_error("compiler Hessian CUDA residency contract mismatch");
  if (options.backend == GENERATIVEQC_BACKEND_CUDA) {
    for (const auto* key : {"first_integral_backend", "second_integral_backend", "grid_ad_backend",
                            "response_backend"})
      if (diagnostics.at(key) != "cuda")
        throw std::runtime_error("compiler Hessian selected a non-CUDA derivative provider");
    for (const auto* key : {"ecp_ad_backend", "df_integral_ad_backend"})
      if (diagnostics.contains(key) && !diagnostics.at(key).is_null() &&
          diagnostics.at(key) != "cuda")
        throw std::runtime_error(
            "compiler Hessian selected a non-CUDA optional derivative provider");
  }
  const auto& panel = result.at(invocation.operation == "hvp" ? "value" : "matrix");
  check_panel(panel, invocation.operation == "hvp" ? atoms.size() : 3 * atoms.size(),
              invocation.operation == "hvp" ? 3 : 3 * atoms.size());
  if (options.json) {
    std::cout << result.dump() << '\n';
  } else {
    std::cout << "WB97M-V " << invocation.operation << " (hartree/bohr^2; compiler AD, "
              << calculator.at("device").get<std::string>() << ")\n";
    for (const auto& row : panel) std::cout << row.dump() << '\n';
  }
  return 0;
#else
  (void)options;
  (void)atoms;
  (void)invocation;
  throw UsageError("compiler hessian/hvp CLI process isolation requires a POSIX platform");
#endif
}

}  // namespace generativeqc::cli

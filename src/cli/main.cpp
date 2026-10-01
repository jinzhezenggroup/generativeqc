#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include "cli/native_basis.hpp"
#include "generativeqc/generativeqc.hpp"
#include "methods/generated_method_manifest.hpp"

#ifndef GENERATIVEQC_CLI_VERSION
#define GENERATIVEQC_CLI_VERSION "unknown"
#endif

namespace {

constexpr double kBohrPerAngstrom = 1.8897261254578281;

class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct RunOptions {
  std::string input;
  std::string method_name{"gfn2-xtb"};
  generativeqc_method method{GENERATIVEQC_METHOD_GFN2_XTB};
  std::string basis_name{"sto-3g"};
  generativeqc_basis_representation representation{GENERATIVEQC_BASIS_CARTESIAN};
  generativeqc_backend backend{GENERATIVEQC_BACKEND_CPU_REFERENCE};
  int device_id{0};
  int charge{0};
  std::uint32_t multiplicity{1};
  bool input_angstrom{true};
  bool basis_explicit{false};
  bool representation_explicit{false};
  bool forces{false};
  bool json{false};
};

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

int parse_int(std::string_view text, std::string_view name) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(std::string(text), &consumed);
  } catch (const std::exception&) {
    throw UsageError(std::string(name) + " must be an integer");
  }
  if (consumed != text.size() || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw UsageError(std::string(name) + " must fit int32");
  }
  return static_cast<int>(value);
}

std::uint32_t parse_positive_u32(std::string_view text, std::string_view name) {
  const int value = parse_int(text, name);
  if (value < 1) throw UsageError(std::string(name) + " must be positive");
  return static_cast<std::uint32_t>(value);
}

int atomic_number(std::string token) {
  if (!token.empty() && std::all_of(token.begin(), token.end(),
                                    [](unsigned char c) { return std::isdigit(c) != 0; })) {
    const int value = parse_int(token, "atomic number");
    if (value < 1 || value > 86) throw UsageError("native XYZ parsing supports atomic numbers 1 through 86");
    return value;
  }

  if (token.empty()) throw UsageError("empty element symbol");
  token = lower(std::move(token));
  token.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(token.front())));

  static constexpr std::array<std::string_view, 86> symbols{
      "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
      "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
      "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
      "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
      "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
      "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"};
  const auto found = std::find(symbols.begin(), symbols.end(), token);
  if (found == symbols.end())
    throw UsageError("unknown or unsupported native XYZ element symbol: " + token);
  return static_cast<int>(std::distance(symbols.begin(), found)) + 1;
}

std::vector<generativeqc_atom> read_xyz(const RunOptions& options) {
  std::ifstream stream(options.input);
  if (!stream) throw UsageError("cannot open XYZ input: " + options.input);

  std::string line;
  if (!std::getline(stream, line)) throw UsageError("XYZ input is empty");
  std::size_t consumed = 0;
  unsigned long long count = 0;
  try {
    count = std::stoull(line, &consumed);
  } catch (const std::exception&) {
    throw UsageError("XYZ first line must be a positive atom count");
  }
  while (consumed < line.size() && std::isspace(static_cast<unsigned char>(line[consumed])) != 0)
    ++consumed;
  if (consumed != line.size() || count == 0 || count > std::numeric_limits<std::uint32_t>::max()) {
    throw UsageError("XYZ atom count must be a positive uint32");
  }
  if (!std::getline(stream, line)) throw UsageError("XYZ input is missing its comment line");

  std::vector<generativeqc_atom> atoms;
  atoms.reserve(static_cast<std::size_t>(count));
  const double scale = options.input_angstrom ? kBohrPerAngstrom : 1.0;
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::getline(stream, line))
      throw UsageError("XYZ ended before all atom records were read");
    std::istringstream record(line);
    std::string element;
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(record >> element >> x >> y >> z))
      throw UsageError("invalid XYZ atom record at index " + std::to_string(index));
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      throw UsageError("XYZ coordinates must be finite");
    atoms.push_back({atomic_number(element), x * scale, y * scale, z * scale});
  }
  return atoms;
}

std::string_view family_name(generativeqc_method_family family) {
  switch (family) {
    case GENERATIVEQC_METHOD_FAMILY_HARTREE_FOCK:
      return "hartree_fock";
    case GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL:
      return "density_functional";
    case GENERATIVEQC_METHOD_FAMILY_COUPLED_CLUSTER:
      return "coupled_cluster";
    case GENERATIVEQC_METHOD_FAMILY_PERTURBATION:
      return "perturbation";
    case GENERATIVEQC_METHOD_FAMILY_SEMIEMPIRICAL:
      return "semiempirical";
    default:
      return "unknown";
  }
}

std::string properties(generativeqc_property_flags flags) {
  std::string value;
  if (flags & GENERATIVEQC_PROPERTY_ENERGY) value = "energy";
  if (flags & GENERATIVEQC_PROPERTY_FORCES) {
    if (!value.empty()) value += ",";
    value += "forces";
  }
  return value.empty() ? "-" : value;
}

std::string_view backend_name(generativeqc_backend backend) {
  switch (backend) {
    case GENERATIVEQC_BACKEND_CPU_REFERENCE:
      return "cpu_reference";
    case GENERATIVEQC_BACKEND_CUDA:
      return "cuda";
    case GENERATIVEQC_BACKEND_HYBRID_CUDA:
      return "hybrid_cuda";
    default:
      return "unknown";
  }
}

void print_usage(std::ostream& out) {
  out << "Usage:\n"
         "  generativeqc --version\n"
         "  generativeqc methods [--json]\n"
         "  generativeqc basis list [--json]\n"
         "  generativeqc run INPUT.xyz [options]\n"
         "  generativeqc profile show|clear\n"
         "  generativeqc autotune --show-profile|--clear-profile\n"
         "  generativeqc resources ...   # reserved; Python frontend currently owns it\n"
         "  generativeqc profile install|export|diagnose ...  # reserved; Python frontend owns it\n"
         "  generativeqc autotune ...    # tuning remains in the Python frontend\n\n"
         "Native run options:\n"
         "  --method gfn2-xtb|rhf|uhf  Native method (default: gfn2-xtb)\n"
         "  --basis NAME             Bundled Gaussian basis for RHF/UHF (default: sto-3g)\n"
         "  --representation cartesian|spherical  Gaussian AO representation (default: cartesian)\n"
         "  --backend cpu|cuda       Execution backend (default: cpu)\n"
         "  --device-id N            CUDA device index (default: 0)\n"
         "  --charge N               Molecular charge (default: 0)\n"
         "  --multiplicity N         Spin multiplicity (default: 1)\n"
         "  --units angstrom|bohr    XYZ coordinate units (default: angstrom)\n"
         "  --forces                 Request analytic forces\n"
         "  --json                   Emit machine-readable output\n";
}

int basis_command(int argc, char** argv) {
  if (argc < 3) throw UsageError("basis requires an operation");
  const std::string_view operation = argv[2];
  if (operation == "--help" || operation == "-h" || operation == "help") {
    print_usage(std::cout);
    return 0;
  }
  if (operation != "list") throw UsageError("unknown basis operation: " + std::string(operation));
  if (argc > 4 || (argc == 4 && std::string_view(argv[3]) != "--json"))
    throw UsageError("basis list accepts only the optional --json flag");

  const bool json = argc == 4;
  const auto names = generativeqc::cli::bundled_basis_names();
  if (json) {
    std::cout << "[";
    for (std::size_t index = 0; index < names.size(); ++index) {
      if (index) std::cout << ",";
      std::cout << "\"" << names[index] << "\"";
    }
    std::cout << "]\n";
  } else {
    for (const auto name : names) std::cout << name << '\n';
  }
  return 0;
}

void print_methods(bool json) {
  using generativeqc::methods::generated::kMethodManifest;
  if (json) std::cout << "[\n";
  if (!json) std::cout << "METHOD\tFAMILY\tPROPERTIES\tBATCH\tSTATUS\n";

  bool first = true;
  for (const auto& entry : kMethodManifest) {
    const auto capability = generativeqc::method_capabilities(entry.method);
    if (json) {
      if (!first) std::cout << ",\n";
      std::cout << "  {\"name\":\"" << entry.name << "\",\"family\":\""
                << family_name(capability.family) << "\",\"properties\":[";
      bool first_property = true;
      if (capability.supported_properties & GENERATIVEQC_PROPERTY_ENERGY) {
        std::cout << "\"energy\"";
        first_property = false;
      }
      if (capability.supported_properties & GENERATIVEQC_PROPERTY_FORCES) {
        if (!first_property) std::cout << ",";
        std::cout << "\"forces\"";
      }
      std::cout << "],\"supports_batch\":" << (capability.supports_batch ? "true" : "false")
                << ",\"available\":" << (capability.available ? "true" : "false") << "}";
    } else {
      std::cout << entry.name << '\t' << family_name(capability.family) << '\t'
                << properties(capability.supported_properties) << '\t'
                << (capability.supports_batch ? "yes" : "no") << '\t'
                << (capability.available ? "available" : "unavailable") << '\n';
    }
    first = false;
  }
  if (json) std::cout << "\n]\n";
}

bool is_gfn2(const RunOptions& options) {
  return options.method == GENERATIVEQC_METHOD_GFN2_XTB;
}

std::string_view representation_name(generativeqc_basis_representation representation) {
  return representation == GENERATIVEQC_BASIS_SPHERICAL ? "spherical" : "cartesian";
}

generativeqc_method_descriptor method_descriptor(const RunOptions& options) {
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = options.method;
  descriptor.max_iterations = 100;
  descriptor.diis_history = 8;
  descriptor.energy_tolerance = is_gfn2(options) ? 1.0e-10 : 1.0e-12;
  descriptor.density_tolerance = is_gfn2(options) ? 1.0e-8 : 1.0e-10;
  descriptor.screening_tolerance = is_gfn2(options) ? 0.0 : 1.0e-14;
  descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  descriptor.density_fitting_relative_threshold = 1.0e-10;
  descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
  descriptor.mp2_denominator_threshold = 1.0e-10;
  descriptor.ccsd_max_iterations = 100;
  descriptor.ccsd_diis_history = 6;
  descriptor.ccsd_energy_tolerance = 1.0e-11;
  descriptor.ccsd_residual_tolerance = 1.0e-9;
  descriptor.ccsd_denominator_threshold = 1.0e-10;
  return descriptor;
}

RunOptions parse_run(int argc, char** argv) {
  if (argc < 3) throw UsageError("run requires an XYZ input path");
  RunOptions options;
  options.input = argv[2];

  for (int index = 3; index < argc; ++index) {
    const std::string_view option = argv[index];
    auto value = [&]() -> std::string_view {
      if (++index >= argc) throw UsageError(std::string(option) + " requires a value");
      return argv[index];
    };

    if (option == "--method") {
      const std::string selected = lower(std::string(value()));
      if (selected == "gfn2-xtb" || selected == "gfn2") {
        options.method_name = "gfn2-xtb";
        options.method = GENERATIVEQC_METHOD_GFN2_XTB;
      } else if (selected == "rhf") {
        options.method_name = "rhf";
        options.method = GENERATIVEQC_METHOD_RHF;
      } else if (selected == "uhf") {
        options.method_name = "uhf";
        options.method = GENERATIVEQC_METHOD_UHF;
      } else {
        throw UsageError("native run method must be gfn2-xtb, rhf, or uhf");
      }
    } else if (option == "--basis") {
      options.basis_name = lower(std::string(value()));
      std::replace(options.basis_name.begin(), options.basis_name.end(), '_', '-');
      options.basis_explicit = true;
    } else if (option == "--representation") {
      const std::string selected = lower(std::string(value()));
      if (selected == "cartesian")
        options.representation = GENERATIVEQC_BASIS_CARTESIAN;
      else if (selected == "spherical")
        options.representation = GENERATIVEQC_BASIS_SPHERICAL;
      else
        throw UsageError("--representation must be cartesian or spherical");
      options.representation_explicit = true;
    } else if (option == "--backend") {
      const std::string selected = lower(std::string(value()));
      if (selected == "cpu")
        options.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
      else if (selected == "cuda")
        options.backend = GENERATIVEQC_BACKEND_CUDA;
      else
        throw UsageError("--backend must be cpu or cuda");
    } else if (option == "--device-id") {
      options.device_id = parse_int(value(), "device id");
      if (options.device_id < 0) throw UsageError("device id must be non-negative");
    } else if (option == "--charge") {
      options.charge = parse_int(value(), "charge");
    } else if (option == "--multiplicity") {
      options.multiplicity = parse_positive_u32(value(), "multiplicity");
    } else if (option == "--units") {
      const std::string selected = lower(std::string(value()));
      if (selected == "angstrom")
        options.input_angstrom = true;
      else if (selected == "bohr")
        options.input_angstrom = false;
      else
        throw UsageError("--units must be angstrom or bohr");
    } else if (option == "--forces") {
      options.forces = true;
    } else if (option == "--json") {
      options.json = true;
    } else {
      throw UsageError("unknown run option: " + std::string(option));
    }
  }
  if (is_gfn2(options) && (options.basis_explicit || options.representation_explicit))
    throw UsageError("GFN2-xTB owns its intrinsic basis; --basis/--representation apply to RHF/UHF");
  return options;
}

int run(const RunOptions& options) {
  const std::vector<generativeqc_atom> atoms = read_xyz(options);

  const generativeqc_context_descriptor context_descriptor{sizeof(generativeqc_context_descriptor),
                                                           GENERATIVEQC_ABI_VERSION,
                                                           options.device_id, options.backend};
  generativeqc::Context context(context_descriptor);

  generativeqc::cli::NativeBasisData basis;
  if (!is_gfn2(options)) {
    try {
      basis = generativeqc::cli::expand_bundled_basis(options.basis_name, atoms,
                                                       options.representation);
    } catch (const std::invalid_argument& error) {
      throw UsageError(error.what());
    }
  }
  const auto* shells = is_gfn2(options) ? nullptr : basis.shells.data();
  const auto* primitives = is_gfn2(options) ? nullptr : basis.primitives.data();
  const std::uint32_t shell_count =
      is_gfn2(options) ? 0u : static_cast<std::uint32_t>(basis.shells.size());
  const std::uint32_t primitive_count =
      is_gfn2(options) ? 0u : static_cast<std::uint32_t>(basis.primitives.size());
  const generativeqc_system_descriptor system_descriptor{
      sizeof(generativeqc_system_descriptor), GENERATIVEQC_ABI_VERSION,
      atoms.data(), static_cast<std::uint32_t>(atoms.size()), shells, shell_count,
      primitives, primitive_count, options.charge, options.multiplicity,
      is_gfn2(options) ? GENERATIVEQC_BASIS_CARTESIAN : options.representation};
  generativeqc::System system(context, system_descriptor);

  const generativeqc_method_descriptor method = method_descriptor(options);
  generativeqc::Calculation calculation(context, system, method);
  const generativeqc_property_flags requested =
      GENERATIVEQC_PROPERTY_ENERGY | (options.forces ? GENERATIVEQC_PROPERTY_FORCES : 0u);
  const auto result = calculation.execute(requested);

  std::cout << std::setprecision(17);
  if (options.json) {
    std::cout << "{\"method\":\"" << options.method_name << "\","
              << "\"backend\":\"" << backend_name(result.executed_backend) << "\"";
    if (!is_gfn2(options)) {
      std::cout << ",\"basis\":\"" << options.basis_name << "\","
                << "\"representation\":\"" << representation_name(options.representation)
                << "\"";
    }
    std::cout << ",\"energy_hartree\":" << result.energy << ","
              << "\"iterations\":" << result.iterations;
    if (result.forces) {
      std::cout << ",\"forces_hartree_per_bohr\":[";
      for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
        if (atom) std::cout << ",";
        const std::size_t offset = 3 * atom;
        std::cout << "[" << (*result.forces)[offset] << "," << (*result.forces)[offset + 1] << ","
                  << (*result.forces)[offset + 2] << "]";
      }
      std::cout << "]";
    }
    std::cout << "}\n";
  } else {
    std::cout << "method: " << options.method_name << '\n'
              << "backend: " << backend_name(result.executed_backend) << '\n';
    if (!is_gfn2(options)) {
      std::cout << "basis: " << options.basis_name << '\n'
                << "representation: " << representation_name(options.representation) << '\n';
    }
    std::cout << "energy_hartree: " << result.energy << '\n'
              << "iterations: " << result.iterations << '\n';
    if (result.forces) {
      std::cout << "forces_hartree_per_bohr:\n";
      for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
        const std::size_t offset = 3 * atom;
        std::cout << "  " << atoms[atom].atomic_number << " " << (*result.forces)[offset] << " "
                  << (*result.forces)[offset + 1] << " " << (*result.forces)[offset + 2] << '\n';
      }
    }
  }
  return 0;
}

std::filesystem::path expand_profile_cache_override(std::string value) {
  if (value == "~") {
    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0')
      throw UsageError("cannot expand GENERATIVEQC_PROFILE_CACHE without HOME");
    return std::filesystem::path(home);
  }
  if (value.rfind("~/", 0) == 0) {
    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0')
      throw UsageError("cannot expand GENERATIVEQC_PROFILE_CACHE without HOME");
    return std::filesystem::path(home) / value.substr(2);
  }
  return std::filesystem::path(std::move(value));
}

std::filesystem::path profile_cache_root() {
  if (const char* configured = std::getenv("GENERATIVEQC_PROFILE_CACHE");
      configured != nullptr && *configured != '\0')
    return expand_profile_cache_override(configured);

  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && *xdg != '\0')
    return std::filesystem::path(xdg) / "generativeqc" / "profiles";

  const char* home = std::getenv("HOME");
  if (home == nullptr || *home == '\0')
    throw UsageError("cannot resolve profile cache without HOME or XDG_CACHE_HOME");
  return std::filesystem::path(home) / ".cache" / "generativeqc" / "profiles";
}

std::string json_escape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  constexpr char hex[] = "0123456789abcdef";
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (c < 0x20) {
          escaped += "\\u00";
          escaped += hex[(c >> 4) & 0x0f];
          escaped += hex[c & 0x0f];
        } else {
          escaped.push_back(static_cast<char>(c));
        }
    }
  }
  return escaped;
}

std::string read_profile_index(const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) {
    if (error) throw std::runtime_error("cannot inspect profile index: " + error.message());
    return "{}";
  }

  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open profile index: " + path.string());
  std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const auto first = content.find_first_not_of(" \t\r\n");
  const auto last = content.find_last_not_of(" \t\r\n");
  if (first == std::string::npos || content[first] != '{' || content[last] != '}')
    throw std::runtime_error("profile index is not a JSON object: " + path.string());
  return content;
}

int profile_show() {
  const auto root = profile_cache_root();
  const auto active = read_profile_index(root / "active.json");
  std::cout << "{\n  \"cache\": \"" << json_escape(root.string()) << "\",\n  \"active\": " << active
            << "\n}\n";
  return 0;
}

class ProfileCacheLock {
 public:
  explicit ProfileCacheLock(const std::filesystem::path& path) {
#if defined(__unix__) || defined(__APPLE__)
    descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
    if (descriptor_ < 0)
      throw std::runtime_error("cannot open profile cache lock: " + path.string());
    if (::flock(descriptor_, LOCK_EX) != 0) {
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("cannot acquire profile cache lock: " + path.string());
    }
#else
    (void)path;
#endif
  }

  ~ProfileCacheLock() {
#if defined(__unix__) || defined(__APPLE__)
    if (descriptor_ >= 0) {
      (void)::flock(descriptor_, LOCK_UN);
      (void)::close(descriptor_);
    }
#endif
  }

  ProfileCacheLock(const ProfileCacheLock&) = delete;
  ProfileCacheLock& operator=(const ProfileCacheLock&) = delete;

 private:
#if defined(__unix__) || defined(__APPLE__)
  int descriptor_{-1};
#endif
};

int profile_clear() {
  const auto root = profile_cache_root();
  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error) throw std::runtime_error("cannot create profile cache: " + error.message());

  ProfileCacheLock lock(root / ".lock");
  const auto active = root / "active.json";
  auto pending = active;
  pending += ".tmp";
#if defined(__unix__) || defined(__APPLE__)
  pending += "." + std::to_string(static_cast<long long>(::getpid()));
#endif

  {
    std::ofstream stream(pending, std::ios::trunc);
    if (!stream)
      throw std::runtime_error("cannot write temporary profile index: " + pending.string());
    stream << "{}\n";
    stream.flush();
    if (!stream)
      throw std::runtime_error("cannot flush temporary profile index: " + pending.string());
  }

#if defined(_WIN32)
  std::filesystem::remove(active, error);
  error.clear();
#endif
  std::filesystem::rename(pending, active, error);
  if (error) {
    std::error_code ignored;
    std::filesystem::remove(pending, ignored);
    throw std::runtime_error("cannot activate cleared profile index: " + error.message());
  }
  std::cout << "Local profiles deactivated.\n";
  return 0;
}

int profile_command(int argc, char** argv) {
  if (argc != 3) throw UsageError("profile requires exactly one operation: show or clear");
  const std::string_view operation = argv[2];
  if (operation == "--help" || operation == "-h" || operation == "help") {
    print_usage(std::cout);
    return 0;
  }
  if (operation == "show") return profile_show();
  if (operation == "clear") return profile_clear();
  if (operation == "install" || operation == "export" || operation == "diagnose") {
    std::cerr << "generativeqc: native profile operation '" << operation
              << "' is reserved but not migrated yet; the Python frontend currently provides it\n";
    return 2;
  }
  throw UsageError("unknown profile operation: " + std::string(operation));
}

int reserved_python_subcommand(std::string_view command) {
  std::cerr << "generativeqc: native subcommand '" << command
            << "' is reserved but not migrated yet; the Python frontend currently provides it\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      print_usage(std::cout);
      return 0;
    }

    const std::string_view command = argv[1];
    if (command == "--help" || command == "-h" || command == "help") {
      print_usage(std::cout);
      return 0;
    }
    if (command == "--version" || command == "version") {
      std::cout << "generativeqc " << GENERATIVEQC_CLI_VERSION << " (ABI "
                << GENERATIVEQC_ABI_VERSION << ")\n";
      return 0;
    }
    if (command == "methods") {
      if (argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--json"))
        throw UsageError("methods accepts only the optional --json flag");
      print_methods(argc == 3);
      return 0;
    }
    if (command == "basis") return basis_command(argc, argv);
    if (command == "run") {
      if (argc == 3 &&
          (std::string_view(argv[2]) == "--help" || std::string_view(argv[2]) == "-h")) {
        print_usage(std::cout);
        return 0;
      }
      return run(parse_run(argc, argv));
    }
    if (command == "profile") return profile_command(argc, argv);
    if (command == "autotune") {
      if (argc == 3 && std::string_view(argv[2]) == "--show-profile") return profile_show();
      if (argc == 3 && std::string_view(argv[2]) == "--clear-profile") return profile_clear();
      return reserved_python_subcommand(command);
    }
    if (command == "resources") return reserved_python_subcommand(command);

    throw UsageError("unknown command: " + std::string(command));
  } catch (const UsageError& error) {
    std::cerr << "generativeqc: " << error.what() << "\n\n";
    print_usage(std::cerr);
    return 2;
  } catch (const generativeqc::Error& error) {
    std::cerr << "generativeqc: " << error.what() << " (status " << error.status() << ")\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "generativeqc: " << error.what() << '\n';
    return 1;
  }
}

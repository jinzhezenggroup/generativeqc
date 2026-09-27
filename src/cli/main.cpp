#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
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

#include "methods/generated_method_manifest.hpp"
#include "vibeqc/vibeqc.hpp"

#ifndef VIBEQC_CLI_VERSION
#define VIBEQC_CLI_VERSION "unknown"
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
  vibeqc_backend backend{VIBEQC_BACKEND_CPU_REFERENCE};
  int device_id{0};
  int charge{0};
  std::uint32_t multiplicity{1};
  bool input_angstrom{true};
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
    if (value < 1 || value > 86) throw UsageError("GFN2-xTB supports atomic numbers 1 through 86");
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
    throw UsageError("unknown or unsupported GFN2-xTB element symbol: " + token);
  return static_cast<int>(std::distance(symbols.begin(), found)) + 1;
}

std::vector<vibeqc_atom> read_xyz(const RunOptions& options) {
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

  std::vector<vibeqc_atom> atoms;
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

std::string_view family_name(vibeqc_method_family family) {
  switch (family) {
    case VIBEQC_METHOD_FAMILY_HARTREE_FOCK:
      return "hartree_fock";
    case VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL:
      return "density_functional";
    case VIBEQC_METHOD_FAMILY_COUPLED_CLUSTER:
      return "coupled_cluster";
    case VIBEQC_METHOD_FAMILY_PERTURBATION:
      return "perturbation";
    case VIBEQC_METHOD_FAMILY_SEMIEMPIRICAL:
      return "semiempirical";
    default:
      return "unknown";
  }
}

std::string properties(vibeqc_property_flags flags) {
  std::string value;
  if (flags & VIBEQC_PROPERTY_ENERGY) value = "energy";
  if (flags & VIBEQC_PROPERTY_FORCES) {
    if (!value.empty()) value += ",";
    value += "forces";
  }
  return value.empty() ? "-" : value;
}

std::string_view backend_name(vibeqc_backend backend) {
  switch (backend) {
    case VIBEQC_BACKEND_CPU_REFERENCE:
      return "cpu_reference";
    case VIBEQC_BACKEND_CUDA:
      return "cuda";
    case VIBEQC_BACKEND_HYBRID_CUDA:
      return "hybrid_cuda";
    default:
      return "unknown";
  }
}

void print_usage(std::ostream& out) {
  out << "Usage:\n"
         "  vibeqc --version\n"
         "  vibeqc methods [--json]\n"
         "  vibeqc run INPUT.xyz [options]\n"
         "  vibeqc resources ...   # reserved; Python frontend currently owns it\n"
         "  vibeqc profile ...     # reserved; Python frontend currently owns it\n"
         "  vibeqc autotune ...    # reserved; Python frontend currently owns it\n\n"
         "Native run options:\n"
         "  --method gfn2-xtb|gfn2   Native CLI execution method (default: gfn2-xtb)\n"
         "  --backend cpu|cuda       Execution backend (default: cpu)\n"
         "  --device-id N            CUDA device index (default: 0)\n"
         "  --charge N               Molecular charge (default: 0)\n"
         "  --multiplicity N         Spin multiplicity (default: 1)\n"
         "  --units angstrom|bohr    XYZ coordinate units (default: angstrom)\n"
         "  --forces                 Request analytic forces\n"
         "  --json                   Emit machine-readable output\n";
}

void print_methods(bool json) {
  using vibeqc::methods::generated::kMethodManifest;
  if (json) std::cout << "[\n";
  if (!json) std::cout << "METHOD\tFAMILY\tPROPERTIES\tBATCH\tSTATUS\n";

  bool first = true;
  for (const auto& entry : kMethodManifest) {
    const auto capability = vibeqc::method_capabilities(entry.method);
    if (json) {
      if (!first) std::cout << ",\n";
      std::cout << "  {\"name\":\"" << entry.name << "\",\"family\":\""
                << family_name(capability.family) << "\",\"properties\":[";
      bool first_property = true;
      if (capability.supported_properties & VIBEQC_PROPERTY_ENERGY) {
        std::cout << "\"energy\"";
        first_property = false;
      }
      if (capability.supported_properties & VIBEQC_PROPERTY_FORCES) {
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

vibeqc_method_descriptor gfn2_method() {
  vibeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = VIBEQC_ABI_VERSION;
  descriptor.method = VIBEQC_METHOD_GFN2_XTB;
  descriptor.max_iterations = 100;
  descriptor.diis_history = 8;
  descriptor.energy_tolerance = 1.0e-10;
  descriptor.density_tolerance = 1.0e-8;
  descriptor.screening_tolerance = 0.0;
  descriptor.density_fitting_mode = VIBEQC_DENSITY_FITTING_NONE;
  descriptor.density_fitting_relative_threshold = 1.0e-10;
  descriptor.precision_mode = VIBEQC_PRECISION_FP64;
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
      if (selected != "gfn2-xtb" && selected != "gfn2")
        throw UsageError(
            "native run currently supports gfn2-xtb; Gaussian-basis CLI "
            "resolution is not yet exposed");
      options.method_name = "gfn2-xtb";
    } else if (option == "--backend") {
      const std::string selected = lower(std::string(value()));
      if (selected == "cpu")
        options.backend = VIBEQC_BACKEND_CPU_REFERENCE;
      else if (selected == "cuda")
        options.backend = VIBEQC_BACKEND_CUDA;
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
  return options;
}

int run(const RunOptions& options) {
  const std::vector<vibeqc_atom> atoms = read_xyz(options);

  const vibeqc_context_descriptor context_descriptor{
      sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, options.device_id, options.backend};
  vibeqc::Context context(context_descriptor);

  const vibeqc_system_descriptor system_descriptor{sizeof(vibeqc_system_descriptor),
                                                   VIBEQC_ABI_VERSION,
                                                   atoms.data(),
                                                   static_cast<std::uint32_t>(atoms.size()),
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   0,
                                                   options.charge,
                                                   options.multiplicity,
                                                   VIBEQC_BASIS_CARTESIAN};
  vibeqc::System system(context, system_descriptor);

  const vibeqc_method_descriptor method = gfn2_method();
  vibeqc::Calculation calculation(context, system, method);
  const vibeqc_property_flags requested =
      VIBEQC_PROPERTY_ENERGY | (options.forces ? VIBEQC_PROPERTY_FORCES : 0u);
  const auto result = calculation.execute(requested);

  std::cout << std::setprecision(17);
  if (options.json) {
    std::cout << "{\"method\":\"" << options.method_name << "\","
              << "\"backend\":\"" << backend_name(result.executed_backend) << "\","
              << "\"energy_hartree\":" << result.energy << ","
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
              << "backend: " << backend_name(result.executed_backend) << '\n'
              << "energy_hartree: " << result.energy << '\n'
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

int reserved_python_subcommand(std::string_view command) {
  std::cerr << "vibeqc: native subcommand '" << command
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
      std::cout << "vibeqc " << VIBEQC_CLI_VERSION << " (ABI " << VIBEQC_ABI_VERSION << ")\n";
      return 0;
    }
    if (command == "methods") {
      if (argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--json"))
        throw UsageError("methods accepts only the optional --json flag");
      print_methods(argc == 3);
      return 0;
    }
    if (command == "run") {
      if (argc == 3 &&
          (std::string_view(argv[2]) == "--help" || std::string_view(argv[2]) == "-h")) {
        print_usage(std::cout);
        return 0;
      }
      return run(parse_run(argc, argv));
    }
    if (command == "resources" || command == "profile" || command == "autotune")
      return reserved_python_subcommand(command);

    throw UsageError("unknown command: " + std::string(command));
  } catch (const UsageError& error) {
    std::cerr << "vibeqc: " << error.what() << "\n\n";
    print_usage(std::cerr);
    return 2;
  } catch (const vibeqc::Error& error) {
    std::cerr << "vibeqc: " << error.what() << " (status " << error.status() << ")\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "vibeqc: " << error.what() << '\n';
    return 1;
  }
}

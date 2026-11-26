#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "generativeqc/ks.hpp"

namespace {
void write_result(const char* name, const generativeqc::CalculationResult& result) {
  std::cout << '"' << name << "\":{\"energy\":" << result.energy
            << ",\"iterations\":" << result.iterations << ",\"backend\":" << result.executed_backend
            << ",\"forces\":";
  if (result.forces) {
    std::cout << '[';
    for (std::size_t coordinate = 0; coordinate < result.forces->size(); ++coordinate) {
      if (coordinate) std::cout << ',';
      std::cout << (*result.forces)[coordinate];
    }
    std::cout << ']';
  } else {
    std::cout << "null";
  }
  std::cout << '}';
}
}  // namespace

/** Installed-header SDK consumer: no private headers, Python or reference work.
 * Publish only after all replays succeed, so a later failure cannot leave a
 * successful-looking partial JSON receipt. */
int main(int argc, char** argv) {
  try {
    if (argc != 7 && argc != 8)
      throw std::invalid_argument(
          "expected device, method, representation, spin, fitting, nonlocal level [packed bytes]");
    const std::string device = argv[1], method = argv[2], representation = argv[3];
    const std::string fitting = argv[5];
    const int spin = std::stoi(argv[4]);
    const bool automatic = method.starts_with("libxc:");
    if ((device != "cpu" && device != "cuda") || (spin != 0 && spin != 1) ||
        (representation != "cartesian" && representation != "spherical") ||
        (fitting != "none" && fitting != device && fitting != "auto"))
      throw std::invalid_argument("invalid SDK force context");
    const auto backend =
        device == "cpu" ? GENERATIVEQC_BACKEND_CPU_REFERENCE : GENERATIVEQC_BACKEND_CUDA;
    const generativeqc_context_descriptor context_descriptor{
        sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION, 0, backend};
    generativeqc::Context context(context_descriptor);
    const std::vector<generativeqc_atom> atoms =
        automatic ? std::vector<generativeqc_atom>{{1, 0.0, 0.0, 0.0},
                                                   {1, 0.1, 0.2, 1.4},
                                                   {1, 1.6, 0.4, 0.3},
                                                   {1, 1.8, 0.3, 1.8}}
                  : std::vector<generativeqc_atom>{{1, 0.12, -0.09, -0.71}, {1, -0.08, 0.13, 0.79}};
    generativeqc::System system(context, atoms, "sto-3g", spin, spin + 1,
                                representation == "spherical" ? GENERATIVEQC_BASIS_SPHERICAL
                                                              : GENERATIVEQC_BASIS_CARTESIAN);
    auto descriptor = generativeqc::default_method_descriptor(spin ? "pbe-uks" : "pbe-rks");
    descriptor.max_iterations = automatic ? 300 : 200;
    descriptor.energy_tolerance = automatic ? 1e-13 : 1e-12;
    descriptor.density_tolerance = automatic ? 1e-12 : 1e-10;
    descriptor.screening_tolerance = automatic ? 1e-12 : 1e-13;
    descriptor.density_fitting_mode = fitting == "none"   ? GENERATIVEQC_DENSITY_FITTING_NONE
                                      : fitting == "auto" ? GENERATIVEQC_DENSITY_FITTING_AUTO
                                      : device == "cpu" ? GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE
                                                        : GENERATIVEQC_DENSITY_FITTING_CUDA;
    const generativeqc::KsGrid grid = automatic
                                          ? generativeqc::KsGrid{1, 24, 8, 16, 3, 1e-12, 37, 37}
                                          : generativeqc::KsGrid{4, 0, 0, 0, 3, 1e-12, 256, 0};
    std::optional<generativeqc::KsGrid> nonlocal_grid;
    if (method == "wb97m-v") {
      nonlocal_grid = grid;
      nonlocal_grid->radial_points = std::stoul(argv[6]);
    }
    const std::string selector =
        automatic ? (spin ? "libxc-uks:" : "libxc:") + method.substr(6)
                  : (method == "lda" ? "lda_xc_pw" : method) + (spin ? "-uks" : "-rks");
    auto calculation = generativeqc::prepare_named_ks(
        context, system, descriptor, selector, grid, 256ull << 20, 256ull << 20, nullptr,
        nonlocal_grid, argc == 8 ? std::stoull(argv[7]) : 0);
    const auto properties = calculation.supported_properties();
    constexpr auto requested = GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES;
    if ((properties & requested) != requested)
      throw std::runtime_error("installed SDK prepared owner did not admit native forces");
    const auto cold = calculation.execute(requested);
    const auto warm = calculation.execute(requested);
    const auto energy_only = calculation.execute(GENERATIVEQC_PROPERTY_ENERGY);
    std::cout << std::setprecision(17) << "{\"supported_properties\":" << properties << ',';
    write_result("cold", cold);
    std::cout << ',';
    write_result("warm", warm);
    std::cout << ',';
    write_result("energy_only", energy_only);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}

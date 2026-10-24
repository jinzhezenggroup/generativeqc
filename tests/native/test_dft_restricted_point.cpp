#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/xc.hpp"
#include "generated_libxc_semilocal_registry.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void close(double observed, double expected, double tolerance) {
  require(std::isfinite(observed) && std::abs(observed - expected) <= tolerance,
          "restricted XC differs from independent reference");
}

std::vector<double> received_density;

dft::RestrictedSemilocalPointValue quadratic_point(double rho, const double (&)[3], double) {
  received_density.push_back(rho);
  dft::RestrictedSemilocalPointValue result{};
  result.energy = rho * rho;
  result.rho = 2.0 * rho;
  return result;
}

void registered_points() {
  // PySCF 2.14.0 / Libxc 7.0.0, unpolarized worker, complete E/vrho.
  // This active total-density tail is specifically not a polarized spin fold.
  struct Reference {
    std::string_view name;
    double tail_energy, tail_potential, interior_energy, interior_potential;
  };
  const std::array<Reference, 3> references{
      {{"LDA_C_VWN_4", -7.507157336220036e-21, -9.08794956105623e-6, -0.04193907322617088,
        -0.07581761675210714},
       {"LDA_C_CHACHIYO", -5.81986732777399e-21, -7.054113899082287e-6, -0.04063514652527078,
        -0.0736011811837693},
       {"LDA_C_PZ_MOD", -7.7127975777928e-21, -9.333867975378661e-6, -0.04143055425676998,
        -0.07470320169925493}}};
  const double gradient[3]{};
  for (const auto& reference : references) {
    const auto* program = dft::generated::automatic_restricted_libxc_program(reference.name);
    require(program && program->domain_version == 5 && program->ingredient_mask == 1,
            "density-only registry lacks its distinct restricted descriptor");
    dft::validate_semilocal_point_program(*program);
    const auto tail = program->evaluate(1.1e-15, gradient, 0.0);
    close(tail.energy, reference.tail_energy, 2e-32);
    close(tail.rho, reference.tail_potential, 2e-14);
    const auto interior = program->evaluate(0.62, gradient, 0.0);
    close(interior.energy, reference.interior_energy, 2e-14);
    close(interior.rho, reference.interior_potential, 2e-14);
    for (double rho : {0.0, 2.5e-16}) {
      const auto vacuum = program->evaluate(rho, gradient, 0.0);
      require(vacuum.energy == 0.0 && vacuum.rho == 0.0 && vacuum.kinetic == 0.0,
              "restricted worker did not screen total vacuum density");
      for (double coefficient : vacuum.gradient)
        require(coefficient == 0.0, "restricted vacuum retained a gradient coefficient");
    }
    for (double rho : {-1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
      bool rejected = false;
      try {
        (void)program->evaluate(rho, gradient, 0.0);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "restricted worker accepted invalid density");
    }
  }
  require(!dft::generated::automatic_restricted_libxc_program("GGA_X_PBE_SOL"),
          "unqualified ingredient family received restricted admission");
  require(!dft::generated::automatic_restricted_libxc_program("unknown"),
          "unknown registration received restricted admission");
}

void total_density_contraction() {
  core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  system.shells = {{0, 0, {{1.0, 1.0}}}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "invalid restricted contraction fixture");
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, dft::GridSpec{1, 2, 2, 3, 3, 1e-12});
  require(basis.nao == 1, "contraction fixture requires one AO");
  std::vector<double> ao(grid.point_count());
  basis.evaluate(grid.points().data(), grid.point_count(), 0, 0, 1, ao.data(), ao.size());
  const std::vector<double> density{3e-15};
  double energy = 0.0, potential = 0.0, electrons = 0.0;
  std::vector<double> expected_density;
  for (std::size_t point = 0; point < grid.point_count(); ++point) {
    const double rho = density[0] * ao[point] * ao[point];
    const double weight = grid.weights()[point];
    expected_density.push_back(rho);
    energy += weight * rho * rho;
    potential += weight * 2.0 * rho * ao[point] * ao[point];
    electrons += weight * rho;
  }
  const dft::RestrictedSemilocalPointProgram program{
      "test-quadratic", "independent-total-density-contraction", 1, 5, quadratic_point};
  for (std::size_t tile : {1U, 7U, 103U}) {
    received_density.clear();
    const auto observed = dft::integrate_semilocal_rks(basis, grid, density, program, tile);
    require(received_density.size() == expected_density.size(), "incomplete density evaluation");
    for (std::size_t point = 0; point < expected_density.size(); ++point)
      close(received_density[point], expected_density[point], 1e-30);
    close(observed.energy, energy, 1e-43);
    close(observed.potential.at(0), potential, 1e-28);
    close(observed.electrons, electrons, 1e-28);
  }
  const auto polarized = dft::generated::automatic_libxc_entry("LDA_C_CHACHIYO");
  bool rejected = false;
  try {
    (void)dft::integrate_semilocal_rks(basis, grid, density, *polarized.program);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "density-work RKS silently reused the polarized ABI");
}
}  // namespace

int main() {
  try {
    registered_points();
    total_density_contraction();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

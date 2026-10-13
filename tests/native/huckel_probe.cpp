#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/initial_guess/huckel.hpp"
#include "scf/preliminary_guess.hpp"
#include "scf/solver/proposal_control.hpp"

/** CPU-only source-linked oracle probe. Input is raw contracted shell data;
 * the production basis validator supplies exactly the target normalization. */
int main() {
  using namespace generativeqc;
  using namespace scf::initial_guess;
  try {
    core::System system;
    std::size_t atoms{}, shells{};
    int spherical{};
    if (!(std::cin >> atoms >> shells >> system.charge >> spherical)) return 3;
    system.basis_representation =
        spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      core::Atom item;
      std::cin >> item.atomic_number >> item.position[0] >> item.position[1] >> item.position[2];
      system.atoms.push_back(item);
    }
    for (std::size_t shell = 0; shell < shells; ++shell) {
      core::Shell item;
      std::size_t primitives{};
      std::cin >> item.atom_index >> item.angular_momentum >> primitives;
      item.primitives.resize(primitives);
      for (auto& primitive : item.primitives)
        std::cin >> primitive.exponent >> primitive.coefficient;
      system.shells.push_back(std::move(item));
    }
    std::string detail;
    if (!std::cin ||
        molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::invalid_argument(detail.empty() ? "invalid input" : detail);
    const auto integrals = integrals::build_integrals(system, false, false);
    PreliminaryOptions policy;
    policy.kind = PreliminaryKind::Huckel;
    PreliminaryDiagnostic diagnostic;
    const auto result = huckel_density(system, integrals, policy, diagnostic);
    scf::solver::validate_seed(integrals.overlap, result.density, integrals.nbf,
                               {static_cast<unsigned>(system.electron_count)}, 2.0);
    std::cout << integrals.nbf << ' ' << result.atomic_solves << ' '
              << result.occupied_atomic_orbitals << ' ' << diagnostic.preliminary_iterations << ' '
              << diagnostic.preliminary_fock_builds << ' '
              << preliminary_numeric_capacity(system, policy) << '\n'
              << std::setprecision(17);
    for (double value : result.density) std::cout << value << ' ';
    std::cout << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}

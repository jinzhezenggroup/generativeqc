#include <iomanip>
#include <iostream>

#include "integrals/coulomb_derivative_envelope.hpp"

int main() {
  using namespace generativeqc;
  core::System system;
  std::size_t count{}, limit{};
  double budget{};
  if (!(std::cin >> count >> budget >> limit) || !count || count > 200) return 1;
  system.atoms.resize(count);
  system.shells.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    auto& shell = system.shells[index];
    shell.atom_index = static_cast<unsigned>(index);
    auto& position = system.atoms[index].position;
    std::size_t primitives{};
    if (!(std::cin >> shell.angular_momentum >> position[0] >> position[1] >> position[2] >>
          primitives) || !primitives || primitives > 16)
      return 1;
    shell.primitives.resize(primitives);
    for (auto& primitive : shell.primitives)
      if (!(std::cin >> primitive.exponent >> primitive.coefficient)) return 1;
  }
  // Optional physical atom topology, distinct from coincident coordinates.
  // Repeated shells on an atom must share its certificate work ledger.
  std::size_t atom_count{};
  if (std::cin >> atom_count) {
    if (!atom_count || atom_count > count) return 1;
    std::vector<core::Atom> atoms(atom_count);
    std::vector<bool> seen(atom_count);
    for (std::size_t index = 0; index < count; ++index) {
      unsigned atom{};
      if (!(std::cin >> atom) || atom >= atom_count) return 1;
      const auto& position = system.atoms[index].position;
      if (seen[atom] && atoms[atom].position != position) return 1;
      seen[atom] = true;
      atoms[atom].position = position;
      system.shells[index].atom_index = atom;
    }
    system.atoms = std::move(atoms);
  }
  std::cout << std::setprecision(17);
  if (!limit && count == 4) {
    using Envelope = integrals::CoulombDerivativeEnvelope;
    const auto ab = Envelope::pair(system.shells[0], system.shells[1],
                                   system.atoms[system.shells[0].atom_index].position,
                                   system.atoms[system.shells[1].atom_index].position);
    const auto cd = Envelope::pair(system.shells[2], system.shells[3],
                                   system.atoms[system.shells[2].atom_index].position,
                                   system.atoms[system.shells[3].atom_index].position);
    if (!ab || !cd) return 2;
    std::cout << Envelope::quartet(*ab, *cd) << '\n';
    return 0;
  }
  const auto omitted = integrals::coulomb_derivative_omissions(system, budget, limit);
  std::cout << omitted.bound << '\n';
  std::cerr << "candidate_checks=" << omitted.candidate_checks << '\n';
  for (const auto& source : omitted.sources)
    std::cout << source[0] << ' ' << source[1] << ' ' << source[2] << ' ' << source[3] << '\n';
}

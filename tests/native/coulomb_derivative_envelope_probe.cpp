#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

#include "integrals/coulomb_derivative_envelope.hpp"

int main(int argc, char** argv) {
  using namespace generativeqc;
  if (argc == 2 && std::string_view(argv[1]) == "--rounding") {
    using Envelope = integrals::CoulombDerivativeEnvelope;
    using Limits = std::numeric_limits<double>;
    for (const auto value : {0.0, -0.0, Limits::denorm_min(), -Limits::denorm_min(),
                             std::nextafter(Limits::min(), 0.0), Limits::min(), 0.5, 1.0, -1.0,
                             Limits::max(), Limits::infinity(), -Limits::infinity()}) {
      if (std::bit_cast<std::uint64_t>(Envelope::upper(value)) !=
              std::bit_cast<std::uint64_t>(std::nextafter(value, Limits::infinity())) ||
          std::bit_cast<std::uint64_t>(Envelope::lower(value)) !=
              std::bit_cast<std::uint64_t>(std::max(0.0, std::nextafter(value, 0.0))))
        return 1;
    }
    return std::isnan(Envelope::upper(Limits::quiet_NaN())) &&
                   Envelope::lower(Limits::quiet_NaN()) == 0.0
               ? 0
               : 1;
  }
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
          primitives) ||
        !primitives || primitives > 16)
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
  if (argc == 2 && std::string_view(argv[1]) == "--value-pair-bounds") {
    using Envelope = integrals::CoulombDerivativeEnvelope;
    for (const auto& left : system.shells)
      for (const auto& right : system.shells) {
        const auto bound = Envelope::pair(left, right, system.atoms[left.atom_index].position,
                                          system.atoms[right.atom_index].position);
        if (!bound) return 2;
        const auto components = (2 * left.angular_momentum + 1) * (2 * right.angular_momentum + 1);
        std::cout << Envelope::upper((*bound)[0] / components) << '\n';
      }
    return 0;
  }
  if (!limit && count == 4) {
    using Envelope = integrals::CoulombDerivativeEnvelope;
    const auto ab = Envelope::pair(system.shells[0], system.shells[1],
                                   system.atoms[system.shells[0].atom_index].position,
                                   system.atoms[system.shells[1].atom_index].position);
    const auto cd = Envelope::pair(system.shells[2], system.shells[3],
                                   system.atoms[system.shells[2].atom_index].position,
                                   system.atoms[system.shells[3].atom_index].position);
    if (!ab || !cd) return 2;
    const auto bounds = Envelope::quartet(*ab, *cd);
    for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
      std::cout << bounds(
                       system.shells[0].atom_index == atom || system.shells[1].atom_index == atom,
                       system.shells[2].atom_index == atom || system.shells[3].atom_index == atom)
                << '\n';
    return 0;
  }
  const bool bitmap = argc == 2 && std::string_view(argv[1]) == "--rank-bitmap";
  const auto omitted = integrals::coulomb_derivative_omissions(system, budget, limit, bitmap);
  std::cout << omitted.bound << '\n';
  std::cerr << "candidate_checks=" << omitted.candidate_checks << '\n';
  if (bitmap) {
    std::cout << omitted.rank_count << ' ' << omitted.size() << '\n';
    for (auto word : omitted.omitted_rank_words) std::cout << word << '\n';
    return omitted.omitted_rank_words.empty() ? 3 : 0;
  }
  for (const auto& source : omitted.sources)
    std::cout << source[0] << ' ' << source[1] << ' ' << source[2] << ' ' << source[3] << '\n';
}

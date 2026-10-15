#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "molecule/basis_catalog.hpp"

namespace {
using namespace generativeqc::molecule::catalog;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejected(Function operation) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid catalog input was accepted");
}

void run() {
  const auto names = bundled_basis_names();
  require(names.size() >= 3 && std::is_sorted(names.begin(), names.end()),
          "catalog discovery lost ordered existing basis names");
  require(std::find(names.begin(), names.end(), "sto-3g") != names.end(),
          "the established STO-3G catalog entry is missing");
  std::array<generativeqc_atom, 3> atoms{{{8, 0, 0, 0}, {1, 0, 0, 1}, {3, 0, 0, 2}}};
  for (const auto name : names) {
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
      const auto capacity = bundled_basis_capacity(name, atoms, representation);
      const auto result = expand_bundled_basis(name, atoms, representation);
      require(result.shells.size() == capacity.shell_count &&
                  result.primitives.size() == capacity.primitive_count &&
                  capacity.buffer_bytes ==
                      result.shells.size() * sizeof(generativeqc_shell) +
                          result.primitives.size() * sizeof(generativeqc_primitive),
              "catalog capacity does not match detached buffers");
      std::size_t primitive_offset = 0, expected_ao = 0, previous_atom = 0;
      for (const auto& shell : result.shells) {
        require(shell.atom_index < atoms.size() && shell.atom_index >= previous_atom &&
                    shell.primitive_offset == primitive_offset,
                "catalog changed atom order or primitive ownership");
        previous_atom = shell.atom_index;
        primitive_offset += shell.primitive_count;
        const auto angular = static_cast<std::size_t>(shell.angular_momentum);
        expected_ao += representation == GENERATIVEQC_BASIS_SPHERICAL
                           ? 2 * angular + 1
                           : (angular + 1) * (angular + 2) / 2;
      }
      require(expected_ao == capacity.ao_count && primitive_offset == result.primitives.size() &&
                  result.representation == representation,
              "catalog representation or AO capacity differs");
      for (const auto& primitive : result.primitives)
        require(std::isfinite(primitive.exponent) && primitive.exponent > 0 &&
                    std::isfinite(primitive.coefficient),
                "catalog contains invalid primitives");
    }
  }
  const std::array<generativeqc_atom, 1> hydrogen{{{1, 0, 0, 0}}};
  const auto hydrogen_basis =
      expand_bundled_basis("STO_3G", hydrogen, GENERATIVEQC_BASIS_CARTESIAN);
  require(hydrogen_basis.shells.size() == 1 && hydrogen_basis.primitives.size() == 3 &&
              hydrogen_basis.primitives[0].exponent == 3.425250914 &&
              hydrogen_basis.primitives[0].coefficient == 0.1543289673 &&
              hydrogen_basis.primitives[2].exponent == 0.1688554040 &&
              hydrogen_basis.primitives[2].coefficient == 0.4446345422,
          "catalog changed pinned decimal primitives or established name aliases");
  atoms[0].x = std::numeric_limits<double>::quiet_NaN();
  require(bundled_basis_capacity("sto-3g", atoms, GENERATIVEQC_BASIS_CARTESIAN).ao_count > 0,
          "metadata capacity consulted coordinates");
  rejected([&] { bundled_basis_capacity("missing", hydrogen, GENERATIVEQC_BASIS_CARTESIAN); });
  rejected([&] { bundled_basis_capacity("sto-3g", {}, GENERATIVEQC_BASIS_CARTESIAN); });
  rejected([&] { bundled_basis_capacity("sto-3g", hydrogen, 99); });
  const std::array<generativeqc_atom, 1> unsupported{{{118, 0, 0, 0}}};
  require(!bundled_basis_supports("sto-3g", 118) && !bundled_basis_supports("missing", 1),
          "catalog advertised absent data");
  rejected([&] { expand_bundled_basis("sto-3g", unsupported, GENERATIVEQC_BASIS_CARTESIAN); });
}

void dump() {
  std::cout << std::setprecision(17);
  for (const auto name : bundled_basis_names()) {
    for (int atomic_number = 1; atomic_number <= 118; ++atomic_number) {
      if (!bundled_basis_supports(name, atomic_number)) continue;
      const std::array<generativeqc_atom, 1> atoms{{{atomic_number, 0, 0, 0}}};
      const auto result = expand_bundled_basis(name, atoms, GENERATIVEQC_BASIS_CARTESIAN);
      std::cout << "{\"basis\":\"" << name << "\",\"atomic_number\":" << atomic_number
                << ",\"pack_checksum\":\"" << bundled_basis_pack_checksum() << "\",\"shells\":[";
      for (std::size_t index = 0; index < result.shells.size(); ++index) {
        const auto& shell = result.shells[index];
        if (index) std::cout << ',';
        std::cout << '[' << shell.angular_momentum << ',' << shell.primitive_offset << ','
                  << shell.primitive_count << ']';
      }
      std::cout << "],\"primitives\":[";
      for (std::size_t index = 0; index < result.primitives.size(); ++index) {
        const auto& primitive = result.primitives[index];
        if (index) std::cout << ',';
        std::cout << '[' << primitive.exponent << ',' << primitive.coefficient << ']';
      }
      std::cout << "]}\n";
    }
  }
}
}  // namespace

int main(int argument_count, char** arguments) {
  try {
    run();
    if (argument_count == 2 && std::string(arguments[1]) == "--dump") dump();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

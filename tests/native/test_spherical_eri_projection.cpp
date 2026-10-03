#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/raw_source.hpp"

namespace {

using generativeqc::core::System;
using generativeqc::integrals::build_integrals;
using generativeqc::integrals::transform_integrals;
using generativeqc::posthf::RawSource;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, const char* message) {
  require(std::isfinite(actual) && std::isfinite(expected), "nonfinite spherical ERI");
  if (std::abs(actual - expected) > 3.0e-12)
    throw std::runtime_error(std::string(message) +
                             ": error=" + std::to_string(std::abs(actual - expected)));
}

std::size_t index(std::size_t i, std::size_t j, std::size_t k, std::size_t l, std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

System fixture(bool four_d, bool reversed) {
  System system;
  system.atoms = {
      {2, {0.1, -0.3, 0.2}}, {2, {-0.7, 0.8, 0.5}}, {2, {0.9, 0.2, -0.6}}, {2, {-0.4, -0.6, 0.7}}};
  if (four_d) {
    for (std::uint32_t atom = 0; atom < 4; ++atom)
      system.shells.push_back({atom, 2, {{0.35 + 0.1 * atom, 1.0}}});
  } else {
    system.shells = {{0, 0, {{0.9, 0.7}, {0.25, 0.3}}},
                     {1, 1, {{0.65, 0.8}, {0.18, -0.15}}},
                     {2, 2, {{0.4, 0.9}, {0.12, -0.1}}}};
  }
  if (reversed) std::reverse(system.shells.begin(), system.shells.end());
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  system.multiplicity = 1;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "spherical shell-local fixture normalization failed");
  return system;
}

void exercise(bool four_d, bool reversed) {
  const auto system = fixture(four_d, reversed);
  auto cartesian_system = system;
  cartesian_system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  const auto actual = build_integrals(system, false);
  const auto one_electron = build_integrals(system, false, false);
  require(one_electron.eri.empty() && one_electron.eri_derivative.empty() &&
              one_electron.overlap == actual.overlap && one_electron.hcore == actual.hcore &&
              one_electron.nuclear_repulsion == actual.nuclear_repulsion &&
              one_electron.nbf == actual.nbf && one_electron.ncoord == 0,
          "skip-ERI spherical preparation changed one-electron output or retained ERIs");
  auto cartesian = build_integrals(cartesian_system, false);
  // The generic transform's legacy metadata contract carries the physical
  // coordinate count even when every optional derivative array is absent.
  cartesian.ncoord = system.atoms.size() * 3;
  const auto reference = transform_integrals(cartesian, system);
  require(actual.nbf == reference.nbf && actual.eri.size() == reference.eri.size(),
          "shell-local spherical output extent changed");
  require(actual.overlap == reference.overlap && actual.hcore == reference.hcore &&
              actual.nuclear_repulsion == reference.nuclear_repulsion && actual.ncoord == 0 &&
              actual.eri_derivative.empty(),
          "shell-local projection changed one-electron or derivative output");
  const auto n = actual.nbf;
  double max_transform_error = 0.0;
  // This visits EVERY public element, including all equal-shell/equal-pair
  // cases and i=k,j!=l where global AO-pair and shell-pair ordering disagree.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          const double value = actual.eri[index(i, j, k, l, n)];
          const double expected = reference.eri[index(i, j, k, l, n)];
          close(value, expected, "shell-local versus Cartesian-transform ERI");
          max_transform_error = std::max(max_transform_error, std::abs(value - expected));
          for (const auto& permutation : std::array<std::array<std::size_t, 4>, 8>{{{i, j, k, l},
                                                                                    {j, i, k, l},
                                                                                    {i, j, l, k},
                                                                                    {j, i, l, k},
                                                                                    {k, l, i, j},
                                                                                    {l, k, i, j},
                                                                                    {k, l, j, i},
                                                                                    {l, k, j, i}}})
            require(actual.eri[index(permutation[0], permutation[1], permutation[2], permutation[3],
                                     n)] == value,
                    "shell-local ERI lost exact eightfold symmetry");
        }
      }
    }
  }

  RawSource source(system);
  double max_raw_error = 0.0;
  if (!four_d) {
    // Whole independent recurrence tensor with signed primitive contractions.
    std::vector<double> raw(actual.eri.size());
    source.read(RawSource::Operator::eri, {0, 0, 0, 0}, {n, n, n, n}, raw.data(), raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
      close(actual.eri[i], raw[i], "shell-local versus independent RawSource ERI");
      max_raw_error = std::max(max_raw_error, std::abs(actual.eri[i] - raw[i]));
    }
  } else {
    // Independent complete blocks include all-distinct dddd, equal pairs,
    // all-equal shells, and shared first shell with differently ordered pairs.
    for (const auto& begins : std::array<std::array<std::size_t, 4>, 4>{
             {{0, 5, 10, 15}, {15, 5, 15, 5}, {10, 10, 10, 10}, {15, 10, 15, 0}}}) {
      std::vector<double> raw(5 * 5 * 5 * 5);
      source.read(RawSource::Operator::eri, begins, {5, 5, 5, 5}, raw.data(), raw.size());
      for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
          for (std::size_t k = 0; k < 5; ++k)
            for (std::size_t l = 0; l < 5; ++l) {
              const double value =
                  actual.eri[index(begins[0] + i, begins[1] + j, begins[2] + k, begins[3] + l, n)];
              const double expected = raw[index(i, j, k, l, 5)];
              close(value, expected, "shell-local dddd versus independent RawSource ERI");
              max_raw_error = std::max(max_raw_error, std::abs(value - expected));
            }
    }
  }
  std::cout << "four_d=" << four_d << " reversed=" << reversed << " nbf=" << n
            << " complete_tensor_values=" << actual.eri.size()
            << " max_transform_error=" << max_transform_error << " max_raw_error=" << max_raw_error
            << '\n';
}

}  // namespace

int main() {
  try {
    for (bool four_d : {false, true})
      for (bool reversed : {false, true}) exercise(four_d, reversed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

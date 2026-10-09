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

System fixture(bool four_d, bool reversed, bool with_f = false) {
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
  if (with_f) system.shells.push_back({3, 3, {{0.55, 1.0}}});
  if (reversed) std::reverse(system.shells.begin(), system.shells.end());
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  system.multiplicity = 1;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "spherical shell-local fixture normalization failed");
  return system;
}

void exercise(bool four_d, bool reversed, bool with_f = false) {
  const auto system = fixture(four_d, reversed, with_f);
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
  if (with_f) {
    // Mixed bases must retain generated low-l work and independent high-l
    // values, regardless of shell order. Probe both sides of that boundary.
    for (const auto indices :
         {std::array<std::size_t, 4>{0, 0, 0, 0}, std::array<std::size_t, 4>{1, 3, 5, 8},
          std::array<std::size_t, 4>{2, 9, 13, 15}, std::array<std::size_t, 4>{15, 14, 13, 12}}) {
      double expected{};
      source.read(RawSource::Operator::eri, indices, {1, 1, 1, 1}, &expected, 1);
      const double value = actual.eri[index(indices[0], indices[1], indices[2], indices[3], n)];
      close(value, expected, "mixed s/p/d/f versus independent RawSource ERI");
      max_raw_error = std::max(max_raw_error, std::abs(value - expected));
    }
  } else if (!four_d) {
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
  std::cout << "four_d=" << four_d << " reversed=" << reversed << " with_f=" << with_f
            << " nbf=" << n << " complete_tensor_values=" << actual.eri.size()
            << " max_transform_error=" << max_transform_error << " max_raw_error=" << max_raw_error
            << '\n';
}

void exercise_f_derivatives() {
  // Physical-atom derivatives, unlike a derivative of one shell-center slot,
  // preserve ERI permutations and may use the producer's symmetric projection.
  System system;
  system.atoms = {{2, {0.1, -0.2, 0.3}}, {2, {-0.4, 0.6, -0.7}}};
  system.shells = {{0, 0, {{0.8, 1.0}}}, {1, 3, {{0.55, 1.0}}}};
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  system.multiplicity = 1;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "f derivative fixture normalization failed");
  auto cartesian_system = system;
  cartesian_system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  const auto actual = build_integrals(system, true);
  const auto reference = transform_integrals(build_integrals(cartesian_system, true), system);
  require(actual.eri_derivative.size() == reference.eri_derivative.size(),
          "f derivative projection changed tensor dimensions");
  for (std::size_t entry = 0; entry < actual.eri_derivative.size(); ++entry)
    close(actual.eri_derivative[entry], reference.eri_derivative[entry],
          "symmetric versus ordered physical-atom derivative projection");
}

void exercise_asymmetric_adapter() {
  // A caller-supplied tensor is not a symmetry-qualified integral producer.
  // The ordered adapter must not overwrite distinct input orientations.
  auto system = fixture(true, false);
  system.shells.resize(1);
  auto cartesian_system = system;
  cartesian_system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  auto cartesian = build_integrals(cartesian_system, false, false);
  cartesian.ncoord = system.atoms.size() * 3;
  cartesian.eri.resize(6 * 6 * 6 * 6);
  for (std::size_t entry = 0; entry < cartesian.eri.size(); ++entry)
    cartesian.eri[entry] = 0.5 + 0.01 * static_cast<double>(entry);
  const auto actual = transform_integrals(cartesian, system);
  const auto components = generativeqc::molecule::cartesian_components(2);
  const auto expansions = generativeqc::molecule::ao_expansions(2, GENERATIVEQC_BASIS_SPHERICAL);
  for (std::size_t first = 0; first < 5; ++first)
    for (std::size_t second = 0; second < 5; ++second)
      for (std::size_t third = 0; third < 5; ++third)
        for (std::size_t fourth = 0; fourth < 5; ++fourth) {
          double expected{};
          for (const auto& bra_first : expansions[first])
            for (const auto& bra_second : expansions[second])
              for (const auto& ket_first : expansions[third])
                for (const auto& ket_second : expansions[fourth]) {
                  const auto position = [&](const auto& term) {
                    return static_cast<std::size_t>(
                        std::find(components.begin(), components.end(), term.component) -
                        components.begin());
                  };
                  expected += bra_first.coefficient * bra_second.coefficient *
                              ket_first.coefficient * ket_second.coefficient *
                              cartesian.eri[index(position(bra_first), position(bra_second),
                                                  position(ket_first), position(ket_second), 6)];
                }
          close(actual.eri[index(first, second, third, fourth, 5)], expected,
                "ordered adapter symmetrized an arbitrary tensor");
        }
}

void exercise_distinct_f_shells() {
  // The largest shell buffer must also work when no within-block pair
  // triangle can hide a Cartesian component or a pair-ordering mistake.
  auto system = fixture(true, false);
  system.shells.clear();
  for (std::uint32_t atom = 0; atom < 4; ++atom)
    system.shells.push_back({atom, 3, {{0.35 + 0.1 * atom, 1.0}}});
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "four-f fixture normalization failed");
  const auto actual = build_integrals(system, false);
  RawSource source(system);
  std::vector<double> raw(7 * 7 * 7 * 7);
  source.read(RawSource::Operator::eri, {0, 7, 14, 21}, {7, 7, 7, 7}, raw.data(), raw.size());
  for (std::size_t first = 0; first < 7; ++first)
    for (std::size_t second = 0; second < 7; ++second)
      for (std::size_t third = 0; third < 7; ++third)
        for (std::size_t fourth = 0; fourth < 7; ++fourth)
          close(actual.eri[index(first, second + 7, third + 14, fourth + 21, actual.nbf)],
                raw[index(first, second, third, fourth, 7)],
                "shared four-f preparation differs from the scalar recurrence");
}

}  // namespace

int main() {
  try {
    for (bool four_d : {false, true})
      for (bool reversed : {false, true}) exercise(four_d, reversed);
    for (bool reversed : {false, true}) exercise(false, reversed, true);
    exercise_f_derivatives();
    exercise_asymmetric_adapter();
    exercise_distinct_f_shells();
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

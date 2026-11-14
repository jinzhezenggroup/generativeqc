#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "geometry/rigid_subspace.hpp"
#include "solver/cpu/symmetric_eigen.hpp"

namespace generativeqc::geometry {

/** Supplied-Hessian analysis only: no electronic or stationary-point claim.
 * Inputs are raw Hartree/Bohr^2, Bohr, and positive physical/isotopic dalton
 * masses. Geometry-derived mass-weighted rigid motions are projected out;
 * small vibrational eigenvalues are never deleted. Roundoff averaging is
 * permitted only after the raw absolute symmetry gate, leaving input intact.
 * Signed frequencies are unscaled cm^-1 (negative denotes imaginary).
 * The bound covers dense analysis storage, not caller inputs/JSON or RSS.
 */
struct HarmonicModes {
  std::size_t coordinates{}, rigid_modes{}, active_modes{}, negative_modes{};
  std::size_t numeric_reservation_bytes{};
  double raw_symmetry_error{};
  std::vector<double> eigenvalues, frequencies_cm1, mass_weighted_modes, cartesian_modes;
};

inline HarmonicModes harmonic_modes(std::span<const double> hessian,
                                    std::span<const double> positions,
                                    std::span<const double> masses,
                                    std::size_t maximum_numeric_bytes,
                                    double raw_symmetry_tolerance = 1e-10) {
  const auto count = positions.size();
  if (!count || count % 3 || count > 300 || hessian.size() != count * count ||
      masses.size() != count / 3 || !std::isfinite(raw_symmetry_tolerance) ||
      raw_symmetry_tolerance <= 0)
    throw std::invalid_argument("invalid harmonic analysis dimensions or symmetry gate");
  const auto reservation = sizeof(double) * (16 * count * count + 32 * count);
  if (reservation > maximum_numeric_bytes)
    throw std::length_error("harmonic analysis exceeds its numeric budget");
  HarmonicModes result;
  result.coordinates = count;
  result.numeric_reservation_bytes = reservation;
  for (std::size_t row = 0; row < count; ++row)
    for (std::size_t column = 0; column < count; ++column) {
      const double value = hessian[row * count + column];
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite raw harmonic Hessian");
      result.raw_symmetry_error =
          std::max(result.raw_symmetry_error, std::abs(value - hessian[column * count + row]));
    }
  if (result.raw_symmetry_error > raw_symmetry_tolerance)
    throw std::invalid_argument("raw harmonic Hessian exceeds symmetry tolerance");
  const auto subspace = rigid_subspace(positions, 1e-8, masses);
  const auto active = subspace.active_modes;
  result.rigid_modes = subspace.rigid_modes;
  result.active_modes = active;
  if (!active) return result;
  std::vector<double> weighted(count * count), product(count * active), reduced(active * active);
  for (std::size_t row = 0; row < count; ++row)
    for (std::size_t column = 0; column < count; ++column) {
      weighted[row * count + column] =
          (0.5 * hessian[row * count + column] + 0.5 * hessian[column * count + row]) /
          std::sqrt(masses[row / 3]) / std::sqrt(masses[column / 3]);
      if (!std::isfinite(weighted[row * count + column]))
        throw std::invalid_argument("harmonic mass weighting exceeds finite precision");
    }
  for (std::size_t row = 0; row < count; ++row)
    for (std::size_t inner = 0; inner < count; ++inner)
      for (std::size_t mode = 0; mode < active; ++mode)
        product[row * active + mode] +=
            weighted[row * count + inner] * subspace.basis[inner * active + mode];
  for (std::size_t row = 0; row < active; ++row)
    for (std::size_t column = 0; column < active; ++column) {
      double value = 0;
      for (std::size_t inner = 0; inner < count; ++inner)
        value += subspace.basis[inner * active + row] * product[inner * active + column];
      if (!std::isfinite(value)) throw std::overflow_error("harmonic projection overflow");
      reduced[row * active + column] = value;
    }
  const auto prepared = solver::cpu::prepare_owned_symmetric_eigen(
      active, solver::cpu::SymmetricEigenFamily::scalar_owned_row_major);
  const auto spectrum = solver::cpu::execute_symmetric_eigen(prepared, std::move(reduced));
  result.eigenvalues = spectrum.values;
  result.mass_weighted_modes.resize(count * active);
  result.cartesian_modes.resize(count * active);
  constexpr double hartree_joule = 4.3597447222060e-18;
  constexpr double bohr_metre = 5.29177210544e-11;
  constexpr double dalton_kilogram = 1.66053906892e-27;
  const double conversion = std::sqrt(hartree_joule / (dalton_kilogram * bohr_metre * bohr_metre)) /
                            (2 * std::numbers::pi * 299792458.0 * 100);
  for (std::size_t mode = 0; mode < active; ++mode) {
    const double value = spectrum.values[mode];
    const double frequency = std::copysign(std::sqrt(std::abs(value)) * conversion, value);
    if (!std::isfinite(frequency)) throw std::overflow_error("harmonic frequency overflow");
    result.frequencies_cm1.push_back(frequency);
    result.negative_modes += value < 0;
    std::size_t pivot = 0;
    for (std::size_t coordinate = 0; coordinate < count; ++coordinate) {
      double component = 0;
      for (std::size_t inner = 0; inner < active; ++inner)
        component +=
            subspace.basis[coordinate * active + inner] * spectrum.vectors[inner * active + mode];
      result.mass_weighted_modes[coordinate * active + mode] = component;
      if (std::abs(component) > std::abs(result.mass_weighted_modes[pivot * active + mode]))
        pivot = coordinate;
    }
    const double sign = result.mass_weighted_modes[pivot * active + mode] < 0 ? -1 : 1;
    for (std::size_t coordinate = 0; coordinate < count; ++coordinate) {
      auto& component = result.mass_weighted_modes[coordinate * active + mode];
      component *= sign;
      const double cartesian = component / std::sqrt(masses[coordinate / 3]);
      if (!std::isfinite(cartesian)) throw std::overflow_error("Cartesian harmonic mode overflow");
      result.cartesian_modes[coordinate * active + mode] = cartesian;
    }
  }
  return result;
}

}  // namespace generativeqc::geometry

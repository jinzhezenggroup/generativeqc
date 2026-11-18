#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace generativeqc::geometry {

/** Orthonormal vibrational coordinates in the selected Cartesian metric.
 * Empty masses preserve the unweighted corrector; positive masses select the
 * mass-weighted metric. Normalizing masses by their maximum changes neither
 * the rigid subspace nor its complement and avoids unnecessary mass scales.
 * Translation and rotation are removed by geometry, never by deleting small
 * Hessian eigenvalues. Two-pass modified Gram-Schmidt bounds cancellation for
 * linear molecules and permits the five-dimensional rigid-motion space.
 */
struct RigidSubspace {
  std::size_t coordinates{}, rigid_modes{}, active_modes{};
  std::vector<double> basis;
};

inline RigidSubspace rigid_subspace(std::span<const double> positions,
                                    double relative_tolerance = 1e-8,
                                    std::span<const double> masses = {}) {
  const auto count = positions.size();
  if (!count || count % 3 || count > std::numeric_limits<std::size_t>::max() / count ||
      !std::isfinite(relative_tolerance) || relative_tolerance <= 0 || relative_tolerance >= 1)
    throw std::invalid_argument("invalid Cartesian rigid-motion geometry");
  const auto atoms = count / 3;
  if (!masses.empty() && masses.size() != atoms)
    throw std::invalid_argument("rigid-motion masses must match the atom count");
  std::vector<double> weights(atoms, 1), root_weights(atoms, 1);
  if (!masses.empty()) {
    for (double mass : masses)
      if (!std::isfinite(mass) || mass <= 0)
        throw std::invalid_argument("rigid-motion masses must be finite and positive");
    const double maximum_mass = *std::max_element(masses.begin(), masses.end());
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      weights[atom] = masses[atom] / maximum_mass;
      if (weights[atom] == 0)
        throw std::invalid_argument("rigid-motion mass ratios exceed finite precision");
      root_weights[atom] = std::sqrt(weights[atom]);
    }
  }
  double total_weight = 0;
  for (double weight : weights) total_weight += weight;
  std::array<double, 3> center{};
  for (std::size_t coordinate = 0; coordinate < count; ++coordinate) {
    if (!std::isfinite(positions[coordinate]))
      throw std::invalid_argument("nonfinite Cartesian rigid-motion geometry");
    center[coordinate % 3] += (positions[coordinate] - positions[coordinate % 3]) *
                              weights[coordinate / 3] / total_weight;
  }
  std::vector<std::vector<double>> orthogonal;
  auto append = [&](std::vector<double> candidate, double threshold) {
    for (unsigned pass = 0; pass < 2; ++pass)
      for (const auto& previous : orthogonal) {
        double projection = 0;
        for (std::size_t coordinate = 0; coordinate < count; ++coordinate)
          projection += previous[coordinate] * candidate[coordinate];
        for (std::size_t coordinate = 0; coordinate < count; ++coordinate)
          candidate[coordinate] -= projection * previous[coordinate];
      }
    double squared_norm = 0;
    for (double value : candidate) squared_norm += value * value;
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm)) throw std::invalid_argument("rigid-motion norm overflow");
    if (norm <= threshold) return;
    for (double& value : candidate) value /= norm;
    orthogonal.push_back(std::move(candidate));
  };
  for (std::size_t axis = 0; axis < 3; ++axis) {
    std::vector<double> translation(count);
    for (std::size_t atom = 0; atom < atoms; ++atom)
      translation[3 * atom + axis] = root_weights[atom];
    append(std::move(translation), 0);
  }
  double rotational_scale = 0;
  std::array<std::vector<double>, 3> rotations;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    auto& rotation = rotations[axis];
    rotation.resize(count);
    double squared_norm = 0;
    const auto first = (axis + 1) % 3, second = (axis + 2) % 3;
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      rotation[3 * atom + first] =
          -root_weights[atom] * (positions[3 * atom + second] - positions[second] - center[second]);
      rotation[3 * atom + second] =
          root_weights[atom] * (positions[3 * atom + first] - positions[first] - center[first]);
    }
    for (double value : rotation) squared_norm += value * value;
    rotational_scale = std::max(rotational_scale, std::sqrt(squared_norm));
  }
  if (atoms > 1 && (!std::isfinite(rotational_scale) || rotational_scale == 0))
    throw std::invalid_argument("coincident molecular geometry has no rotational frame");
  for (auto& rotation : rotations)
    append(std::move(rotation), relative_tolerance * rotational_scale);
  const auto rigid = orthogonal.size();
  for (std::size_t coordinate = 0; coordinate < count && orthogonal.size() < count; ++coordinate) {
    std::vector<double> unit(count);
    unit[coordinate] = 1;
    append(std::move(unit), 1e-10);
  }
  if (orthogonal.size() != count)
    throw std::runtime_error("could not construct complete vibrational subspace");
  RigidSubspace result{count, rigid, count - rigid, {}};
  result.basis.resize(count * result.active_modes);
  for (std::size_t coordinate = 0; coordinate < count; ++coordinate)
    for (std::size_t mode = 0; mode < result.active_modes; ++mode)
      result.basis[coordinate * result.active_modes + mode] = orthogonal[rigid + mode][coordinate];
  return result;
}

}  // namespace generativeqc::geometry

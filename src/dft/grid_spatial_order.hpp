#ifndef GENERATIVEQC_DFT_GRID_SPATIAL_ORDER_HPP
#define GENERATIVEQC_DFT_GRID_SPATIAL_ORDER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dft/grid.hpp"

namespace generativeqc::dft {

/** Exact execution permutation, independent of quadrature and AO screening.
 * Visit centers by a deterministic nearest-neighbor walk, then stably group
 * points by their nearest visited center. Spatially compact point blocks avoid
 * carrying the union of unrelated AO supports through every contraction.
 * Ties retain the first center/index; no points or weights are discarded.
 * Temporary index storage is two point arrays and four atom arrays (size_t).
 */
inline std::vector<std::size_t> spatial_grid_order(const MolecularGrid& grid) {
  const auto& atoms = grid.system().atoms;
  const auto& points = grid.points();
  const auto n = grid.point_count(), na = atoms.size();
  if (!na || points.size() / 3 != n || points.size() % 3)
    throw std::invalid_argument("invalid spatial grid shape");
  std::vector<std::size_t> path(na), visited(na), offsets(na), counts(na);
  std::size_t current = 0;
  for (std::size_t a = 0; a < na; ++a) {
    for (double x : atoms[a].position)
      if (!std::isfinite(x)) throw std::invalid_argument("nonfinite grid center");
    if (atoms[a].position[0] < atoms[current].position[0]) current = a;
  }
  const auto distance = [](const double* a, const double* b) {
    double out = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double delta = a[axis] - b[axis];
      out += delta * delta;
    }
    return out;
  };
  for (std::size_t i = 0; i < na; ++i) {
    path[i] = current;
    visited[current] = 1;
    double nearest = std::numeric_limits<double>::infinity();
    std::size_t next = na;
    for (std::size_t a = 0; a < na; ++a) {
      if (visited[a]) continue;
      const auto d = distance(atoms[a].position.data(), atoms[current].position.data());
      if (next == na || d < nearest) {
        nearest = d;
        next = a;
      }
    }
    current = next;
  }
  std::vector<std::size_t> groups(n), order(n);
  for (std::size_t p = 0; p < n; ++p) {
    for (unsigned axis = 0; axis < 3; ++axis)
      if (!std::isfinite(points[3 * p + axis])) throw std::invalid_argument("nonfinite grid point");
    double nearest = std::numeric_limits<double>::infinity();
    std::size_t group = 0;
    for (std::size_t i = 0; i < na; ++i) {
      const auto d = distance(points.data() + 3 * p, atoms[path[i]].position.data());
      if (d < nearest) {
        nearest = d;
        group = i;
      }
    }
    groups[p] = group;
    ++counts[group];
  }
  std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), std::size_t{0});
  for (std::size_t p = 0; p < n; ++p) order[offsets[groups[p]]++] = p;
  return order;
}

}  // namespace generativeqc::dft
#endif

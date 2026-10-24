#ifndef GENERATIVEQC_DFT_GRID_SPATIAL_ORDER_HPP
#define GENERATIVEQC_DFT_GRID_SPATIAL_ORDER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dft/grid.hpp"

namespace generativeqc::dft {

namespace spatial_grid_detail {
/** Partition whole execution tiles along the widest coordinate extent.
 * Each leaf retains original-index order, so equal coordinates and nth_element's
 * unspecified within-partition order cannot change the resulting permutation.
 * Recursion halves the number of tiles and uses only logarithmic stack space.
 */
inline void bisect(std::vector<std::size_t>& order, std::size_t first, std::size_t last,
                   const std::vector<double>& points, std::size_t tile_points) {
  const auto count = last - first;
  if (count <= tile_points) {
    std::sort(order.begin() + first, order.begin() + last);
    return;
  }
  std::array<double, 3> low, high;
  low.fill(std::numeric_limits<double>::infinity());
  high.fill(-std::numeric_limits<double>::infinity());
  for (auto i = first; i < last; ++i)
    for (unsigned axis = 0; axis < 3; ++axis) {
      const auto x = points[3 * order[i] + axis];
      low[axis] = std::min(low[axis], x);
      high[axis] = std::max(high[axis], x);
    }
  unsigned axis = 0;
  for (unsigned candidate = 1; candidate < 3; ++candidate)
    if (high[candidate] - low[candidate] > high[axis] - low[axis]) axis = candidate;
  const auto tiles = 1 + (count - 1) / tile_points;
  const auto middle = first + (tiles / 2) * tile_points;
  std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + last,
                   [&](auto a, auto b) {
                     const auto x = points[3 * a + axis], y = points[3 * b + axis];
                     return x != y ? x < y : a < b;
                   });
  bisect(order, first, middle, points, tile_points);
  bisect(order, middle, last, points, tile_points);
}
}  // namespace spatial_grid_detail

/** Exact tile-aligned spatial bisection, with one point-index array and no
 * change to quadrature, owner identities or AO screening. The final leaf may
 * be short. Only the execution copy is permuted; force consumers keep the
 * canonical MolecularGrid. This alternative fits the center-group allowance.
 */
inline std::vector<std::size_t> bisected_grid_order(const MolecularGrid& grid,
                                                    std::size_t tile_points) {
  const auto& points = grid.points();
  const auto n = grid.point_count();
  if (!tile_points || !n || points.size() / 3 != n || points.size() % 3)
    throw std::invalid_argument("invalid bisected grid shape or tile size");
  for (const auto coordinate : points)
    if (!std::isfinite(coordinate)) throw std::invalid_argument("nonfinite grid point");
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  spatial_grid_detail::bisect(order, 0, n, points, tile_points);
  return order;
}

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

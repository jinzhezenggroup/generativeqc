"""CPU adjoint contraction of native atomic measures through the Becke graphs.

One forward and one reverse pair pass per point replace coordinate-wise JVP
traversals. Local scalar partials are differentiated from Graph roots and the
normalized-product reverse traversal is emitted by the compiler for CPU/CUDA.

Rationale: .agents/notes/implemented/architecture/2026-09-20-becke-adjoint-compiler-owner.md
"""

import ctypes as ct
import typing
from pathlib import Path

import numpy as np

from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.source_cache import cache_source
from generativeqc_compiler.dft.grid import checked_int
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from .grid_response import grid_response_program

_GRID_ADJOINT_SOURCE = r"""#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__CUDACC__)
#define GENERATIVEQC_GRID_HD __host__ __device__
#else
#define GENERATIVEQC_GRID_HD
#endif

namespace generativeqc_grid_adjoint {
GENERATIVEQC_GRID_HD inline double portable_abs(double value) { return value < 0.0 ? -value : value; }

GENERATIVEQC_GRID_HD inline double portable_exp(double value) {
#if defined(__CUDA_ARCH__)
  return ::exp(value);
#else
  return std::exp(value);
#endif
}

// Shared two-pass Becke reverse composition. Runtime owners supply O(natom)
// scratch per worker and transactional reduction storage. Local mathematical
// partials come exclusively from the compiler's grid_response Graphs.
template <class Norm>
GENERATIVEQC_GRID_HD std::array<double, 4> distance(const double* a, const double* b, Norm norm,
                                              bool& valid) {
  double delta[3], scale = 0;
  for (size_t k = 0; k < 3; ++k) {
    delta[k] = a[k] - b[k];
    scale = std::max(scale, portable_abs(delta[k]));
  }
  if (!(scale > 0) || !std::isfinite(scale)) {
    valid = false;
    return {};
  }
  auto result = norm(delta[0] / scale, delta[1] / scale, delta[2] / scale);
  result[0] *= scale;
  for (double v : result)
    if (!std::isfinite(v)) {
      valid = false;
      return {};
    }
  return result;
}

// Exact triangular geometry storage. The ratio fields are emitted from the same
// local AD graph, not a separately implemented force formula.
struct CenterPair {
  std::array<double, 4> distance;
  std::array<double, 2> ratio;
};
static_assert(sizeof(CenterPair) == 6 * sizeof(double));
GENERATIVEQC_GRID_HD inline size_t center_pair_index(size_t a, size_t b) {
  return a * (a - 1) / 2 + b; // canonical a > b orientation
}
template <class Norm, class PrepareRatio>
GENERATIVEQC_GRID_HD bool prepare_center_geometry(const double* centers, size_t na,
                                                  double tolerance, CenterPair* pairs,
                                                  Norm norm, PrepareRatio prepare_ratio) {
  bool valid = true;
  for (size_t a = 0; a < na; ++a) {
    for (size_t k = 0; k < 3; ++k)
      if (!std::isfinite(centers[3 * a + k])) return false;
    for (size_t b = 0; b < a; ++b) {
      const auto separation = distance(centers + 3 * a, centers + 3 * b, norm, valid);
      if (!valid || separation[0] <= tolerance) return false;
      if (pairs) pairs[center_pair_index(a, b)] = {separation, prepare_ratio(separation[0])};
    }
  }
  return valid;
}
template <class Norm, class Ratio>
struct DirectCenterGeometry {
  const double* centers;
  Norm norm;
  Ratio ratio;
  GENERATIVEQC_GRID_HD std::array<double, 4> separation(size_t a, size_t b, bool& valid) const {
    return distance(centers + 3 * a, centers + 3 * b, norm, valid);
  }
  GENERATIVEQC_GRID_HD std::array<double, 3> coordinate(double difference, size_t, size_t,
                                                      double separation) const {
    return ratio(difference, separation);
  }
};
template <class PreparedRatio>
struct PreparedCenterGeometry {
  const CenterPair* pairs;
  PreparedRatio ratio;
  GENERATIVEQC_GRID_HD std::array<double, 4> separation(size_t a, size_t b, bool&) const {
    return pairs[center_pair_index(a, b)].distance;
  }
  GENERATIVEQC_GRID_HD std::array<double, 3> coordinate(double difference, size_t a, size_t b,
                                                      double) const {
    return ratio(difference, pairs[center_pair_index(a, b)].ratio.data());
  }
};


// Point-dependent state belongs to one point worker only. The cooperative
// schedule retains it until every reverse pair has consumed the forward pass.
struct PointPair {
  std::array<double, 2> ratio;
  std::array<double, 2> factor;
  std::array<double, 2> logarithm[2];
};
static_assert(sizeof(PointPair) == 8 * sizeof(double));
template <bool RetainLogs = true, class Geometry, class Log, class Pair>
GENERATIVEQC_GRID_HD PointPair point_pair(double difference, size_t a, size_t b,
                                         double separation, Geometry geometry,
                                         Log logarithm, Pair pair) {
  const auto r = geometry.coordinate(difference, a, b, separation);
  const bool clipped = portable_abs(r[0]) >= 1;
  auto f = pair(std::clamp(r[0], -1.0, 1.0));
  if (clipped || f[0] < 0 || f[0] > 1) f[1] = 0;
  f[0] = std::clamp(f[0], 0.0, 1.0);
  PointPair result{{r[1], r[2]}, f, {}};
  for (size_t side = 0; side < 2; ++side) {
    const double v = side ? 1 - f[0] : f[0];
    if constexpr (RetainLogs)
      if (v > 0) result.logarithm[side] = logarithm(v);
  }
  return result;
}
template <bool RetainedLogs, class Logs, class Products, class Bars, class Zeros, class Log>
GENERATIVEQC_GRID_HD inline double pair_adjoint(const PointPair& state, size_t a, size_t b,
                                                Logs logs, Products products,
                                                Bars bar_product, Zeros zeros,
                                                double maximum, Log logarithm) {
  // Saturated branches have exactly zero pullback. Skip before exponentiation
  // to avoid inf*0. One exact zero leaves the other factors; two kill the JVP.
  if (state.factor[1] == 0) return 0;
  double bar_mu = 0;
  for (size_t side = 0; side < 2; ++side) {
    const size_t atom = side ? b : a;
    const double v = side ? 1 - state.factor[0] : state.factor[0];
    double derivative = 0;
    if (!zeros[atom]) {
      if constexpr (RetainedLogs) derivative = products[atom] * state.logarithm[side][1];
      else derivative = products[atom] * logarithm(v)[1];
    }
    else if (zeros[atom] == 1 && v == 0)
      derivative = portable_exp(logs[atom] - maximum);
    bar_mu += (side ? -1 : 1) * bar_product[atom] * derivative * state.factor[1];
  }
  return bar_mu;
}

template <class Logs, class Zeros>
GENERATIVEQC_GRID_HD inline double maximum_log_product(size_t na, Logs logs, Zeros zeros) {
  double maximum = -std::numeric_limits<double>::infinity();
  for (size_t a = 0; a < na; ++a)
    if (!zeros[a]) maximum = std::max(maximum, logs[a]);
  return maximum;
}
template <class Logs, class Products, class Bars, class Zeros, class Ratio>
GENERATIVEQC_GRID_HD double normalized_product_adjoint(size_t na, size_t owner, double seed,
    Logs logs, Products products, Bars bar_product, Zeros zeros,
    double maximum, Ratio ratio) {
  double total = 0;
  for (size_t a = 0; a < na; ++a) {
    products[a] = zeros[a] ? 0 : portable_exp(logs[a] - maximum);
    total += products[a];
  }
  // The selected objective uses the SAME ratio graph. A frozen log scale
  // cancels between numerator and denominator.
  const auto objective = ratio(products[owner], total);
  for (size_t a = 0; a < na; ++a)
    bar_product[a] = seed * (objective[2] + (a == owner ? objective[1] : 0));
  // Indexed consumers broadcast this same denominator adjoint instead of
  // materializing it for products whose first derivatives are annihilated.
  // Preserve the dense branch's addition and multiplication order.
  return seed * (objective[2] + 0.0);
}
GENERATIVEQC_GRID_HD inline void point_motion_adjoint(size_t na, size_t owner,
    const double* bar_distance, const std::array<double, 4>* distances, double* gradient) {
  for (size_t a = 0; a < na; ++a)
    for (size_t k = 0; k < 3; ++k) {
      const double value = bar_distance[a] * distances[a][k + 1];
      gradient[3 * a + k] -= value;
      gradient[3 * owner + k] += value;
    }
}

template <class Norm, class Ratio, class Log, class Pair, class Geometry>
GENERATIVEQC_GRID_HD bool contract_point_impl(const double* point, const double* centers, size_t na,
                                   size_t owner, double seed, double* gradient, double* logs,
                                   double* products, double* bar_product, double* bar_distance,
                                   size_t* zeros, std::array<double, 4>* distances, Norm norm,
                                   Ratio ratio, Log logarithm, Pair pair, Geometry geometry) {
  bool valid = true;
  for (size_t a = 0; a < na; ++a) distances[a] = distance(point, centers + 3 * a, norm, valid);
  if (!valid) return false;
  if (na == 1) return true;
  for (size_t a = 0; a < na; ++a) logs[a] = 0;
  for (size_t a = 0; a < na; ++a) zeros[a] = 0;
  for (size_t a = 0; a < na; ++a) bar_distance[a] = 0;
  for (size_t a = 0; a < na; ++a)
    for (size_t b = 0; b < a; ++b) {
      const double separation = geometry.separation(a, b, valid)[0];
      const auto state = point_pair(distances[a][0] - distances[b][0], a, b, separation,
                                    geometry, logarithm, pair);
      const auto& f = state.factor;
      for (size_t side = 0; side < 2; ++side) {
        const size_t atom = side ? b : a;
        const double v = side ? 1 - f[0] : f[0];
        if (v > 0)
          logs[atom] += state.logarithm[side][0];
        else
          ++zeros[atom];
      }
    }
  const double maximum = maximum_log_product(na, logs, zeros);
  if (!std::isfinite(maximum)) return false;
  normalized_product_adjoint(na, owner, seed, logs, products, bar_product, zeros, maximum, ratio);
  for (size_t a = 0; a < na; ++a)
    for (size_t b = 0; b < a; ++b) {
      const auto separation = geometry.separation(a, b, valid);
      const auto state = point_pair<false>(distances[a][0] - distances[b][0], a, b, separation[0],
                                    geometry, logarithm, pair);
      if (state.factor[1] == 0) continue;
      const double bar_mu = pair_adjoint<false>(state, a, b, logs, products, bar_product, zeros, maximum, logarithm);
      bar_distance[a] += bar_mu * state.ratio[0];
      bar_distance[b] -= bar_mu * state.ratio[0];
      for (size_t k = 0; k < 3; ++k) {
        const double value = bar_mu * state.ratio[1] * separation[k + 1];
        gradient[3 * a + k] += value;
        gradient[3 * b + k] -= value;
      }
    }
  point_motion_adjoint(na, owner, bar_distance, distances, gradient);
  return valid;
}
// Keep the bounded direct route for owners without retained geometry capacity.
template <class Norm, class Ratio, class Log, class Pair>
GENERATIVEQC_GRID_HD bool contract_point(const double* point, const double* centers, size_t na,
                                   size_t owner, double seed, double* gradient, double* logs,
                                   double* products, double* bar_product, double* bar_distance,
                                   size_t* zeros, std::array<double, 4>* distances, Norm norm,
                                   Ratio ratio, Log logarithm, Pair pair) {
  return contract_point_impl(point, centers, na, owner, seed, gradient, logs, products,
                             bar_product, bar_distance, zeros, distances, norm, ratio,
                             logarithm, pair, DirectCenterGeometry<Norm, Ratio>{centers, norm, ratio});
}
template <class Norm, class Ratio, class Log, class Pair, class PreparedRatio>
GENERATIVEQC_GRID_HD bool contract_point_prepared(const double* point, const double* centers, size_t na,
                                   size_t owner, double seed, double* gradient, double* logs,
                                   double* products, double* bar_product, double* bar_distance,
                                   size_t* zeros, std::array<double, 4>* distances, Norm norm,
                                   Ratio ratio, Log logarithm, Pair pair,
                                   const CenterPair* pairs, PreparedRatio prepared_ratio) {
  if (!pairs) return contract_point(point, centers, na, owner, seed, gradient, logs, products,
                                   bar_product, bar_distance, zeros, distances, norm, ratio,
                                   logarithm, pair);
  return contract_point_impl(point, centers, na, owner, seed, gradient, logs, products,
                             bar_product, bar_distance, zeros, distances, norm, ratio,
                             logarithm, pair, PreparedCenterGeometry<PreparedRatio>{pairs, prepared_ratio});
}

// A full block owns one point worker. No floating-point atomic operations and
// no cross-point state: barriers delimit distance, forward, reverse and gather
// lifetimes. Team::sync is a block barrier (also host-thread emulatable).
template <class Team, class Norm, class Ratio, class Log, class Pair, class Geometry>
GENERATIVEQC_GRID_HD bool contract_point_cooperative_impl(
    const double* point, const double* centers, size_t na, size_t owner, double seed,
    double* gradient, double* logs, double* products, double* bar_product,
    double* bar_distance, size_t* zeros, std::array<double, 4>* distances,
    PointPair* states, Team team, Norm norm, Ratio ratio, Log logarithm, Pair pair,
    Geometry geometry) {
  for (size_t a = team.rank(); a < na; a += team.size()) {
    bool valid = true;
    distances[a] = distance(point, centers + 3 * a, norm, valid);
    if (!valid) distances[a][0] = 0;
  }
  team.sync();
  // Every participant makes the same decision, including inactive atom lanes.
  for (size_t a = 0; a < na; ++a)
    if (!(distances[a][0] > 0)) return false;
  if (na == 1) { team.sync(); return true; }
  const size_t count = na * (na - 1) / 2;
  for (size_t index = team.rank(); index < count; index += team.size()) {
    size_t a = 1;
    while (a * (a + 1) / 2 <= index) ++a;
    const size_t b = index - center_pair_index(a, 0);
    bool valid = true;
    const auto separation = geometry.separation(a, b, valid);
    states[index] = point_pair(distances[a][0] - distances[b][0], a, b, separation[0],
                               geometry, logarithm, pair);
    if (!valid) states[index].factor[0] = std::numeric_limits<double>::quiet_NaN();
  }
  team.sync();
  for (size_t index = 0; index < count; ++index)
    if (!std::isfinite(states[index].factor[0])) return false;
  for (size_t atom = team.rank(); atom < na; atom += team.size()) {
    logs[atom] = 0;
    zeros[atom] = 0;
    // For one atom this is exactly its subsequence of the triangular traversal.
    for (size_t other = 0; other < na; ++other) {
      if (other == atom) continue;
      const size_t a = std::max(atom, other), b = std::min(atom, other);
      const size_t side = atom == b;
      const auto& state = states[center_pair_index(a, b)];
      const double v = side ? 1 - state.factor[0] : state.factor[0];
      if (v > 0) logs[atom] += state.logarithm[side][0];
      else ++zeros[atom];
    }
  }
  team.sync();
  const double maximum = maximum_log_product(na, logs, zeros);
  if (!std::isfinite(maximum)) return false;
  if (team.rank() == 0)
    normalized_product_adjoint(na, owner, seed, logs, products, bar_product, zeros, maximum, ratio);
  team.sync();
  for (size_t index = team.rank(); index < count; index += team.size()) {
    size_t a = 1;
    while (a * (a + 1) / 2 <= index) ++a;
    const size_t b = index - center_pair_index(a, 0);
    auto& state = states[index];
    std::array<double, 4> pullback{};
    bool valid = true;
    const auto separation = geometry.separation(a, b, valid);
    // Do not multiply zero slope by a potentially infinite ratio derivative.
    if (state.factor[1] != 0) {
      const double bar_mu = pair_adjoint<true>(state, a, b, logs, products, bar_product, zeros, maximum, logarithm);
      pullback[0] = bar_mu * state.ratio[0];
      for (size_t k = 0; k < 3; ++k)
        pullback[k + 1] = bar_mu * state.ratio[1] * separation[k + 1];
    }
    // Forward fields have one reverse consumer. Reuse their shared bytes only
    // after that consumer finishes; the following barrier publishes pullbacks.
    state.ratio = {pullback[0], pullback[1]};
    state.factor = {pullback[2], pullback[3]};
  }
  team.sync();
  for (size_t atom = team.rank(); atom < na; atom += team.size()) {
    bar_distance[atom] = 0;
    for (size_t other = 0; other < na; ++other) {
      if (other == atom) continue;
      const size_t a = std::max(atom, other), b = std::min(atom, other);
      const auto& state = states[center_pair_index(a, b)];
      const double sign = atom == a ? 1 : -1;
      bar_distance[atom] += sign * state.ratio[0];
      gradient[3 * atom] += sign * state.ratio[1];
      gradient[3 * atom + 1] += sign * state.factor[0];
      gradient[3 * atom + 2] += sign * state.factor[1];
    }
  }
  team.sync();
  // This cheap ordered final traversal keeps owner movement accumulation in the
  // original order and avoids any concurrent writers to the selected center.
  if (team.rank() == 0) point_motion_adjoint(na, owner, bar_distance, distances, gradient);
  team.sync();
  return true;
}
template <class Team, class Norm, class Ratio, class Log, class Pair, class PreparedRatio>
GENERATIVEQC_GRID_HD bool contract_point_cooperative(
    const double* point, const double* centers, size_t na, size_t owner, double seed,
    double* gradient, double* logs, double* products, double* bar_product,
    double* bar_distance, size_t* zeros, std::array<double, 4>* distances,
    PointPair* states, Team team, Norm norm, Ratio ratio, Log logarithm, Pair pair,
    const CenterPair* pairs, PreparedRatio prepared_ratio) {
  if (pairs)
    return contract_point_cooperative_impl(point, centers, na, owner, seed, gradient, logs,
        products, bar_product, bar_distance, zeros, distances, states, team, norm, ratio,
        logarithm, pair, PreparedCenterGeometry<PreparedRatio>{pairs, prepared_ratio});
  return contract_point_cooperative_impl(point, centers, na, owner, seed, gradient, logs,
      products, bar_product, bar_distance, zeros, distances, states, team, norm, ratio,
      logarithm, pair, DirectCenterGeometry<Norm, Ratio>{centers, norm, ratio});
}

// Large domains stream a bounded strip of triangular rows through one block.
// Each atom owns its log/gradient entries and gathers only incident strip edges,
// in its original triangular order. Normalization separates two complete pair
// passes; rematerialization keeps storage O(rows*natom), not O(natom^2).
// Team::all publishes votes collectively and completes every participant's read
// before a later vote can overwrite the same shared flag.
template <class Team, class Norm, class Ratio, class Log, class Pair, class Geometry>
GENERATIVEQC_GRID_HD bool contract_point_tiled_cooperative_impl(
    const double* point, const double* centers, size_t na, size_t owner, double seed,
    double* gradient, double* logs, double* products, double* bar_product,
    double* bar_distance, size_t* zeros, std::array<double, 4>* distances,
    PointPair* states, size_t rows, Team team, Norm norm, Ratio ratio, Log logarithm,
    Pair pair, Geometry geometry) {
  if (!rows) return false;
  bool point_valid = true;
  for (size_t atom = team.rank(); atom < na; atom += team.size()) {
    bool valid = true;
    distances[atom] = distance(point, centers + 3 * atom, norm, valid);
    point_valid = point_valid && valid;
    logs[atom] = 0;
    zeros[atom] = 0;
    bar_distance[atom] = 0;
  }
  if (!team.all(point_valid)) return false;
  if (na == 1) { team.sync(); return true; }
  double maximum = 0;
  for (size_t pass = 0; pass < 2; ++pass) {
    for (size_t begin = 1; begin < na; begin += rows) {
      const size_t end = std::min(na, begin + rows);
      const size_t offset = center_pair_index(begin, 0);
      const size_t count = center_pair_index(end, 0) - offset;
      bool tile_valid = true;
      for (size_t index = team.rank(); index < count; index += team.size()) {
        size_t a = begin;
        while (center_pair_index(a + 1, 0) <= offset + index) ++a;
        const size_t b = offset + index - center_pair_index(a, 0);
        bool valid = true;
        const auto separation = geometry.separation(a, b, valid);
        auto state = pass == 0
            ? point_pair(distances[a][0] - distances[b][0], a, b, separation[0],
                         geometry, logarithm, pair)
            : point_pair<false>(distances[a][0] - distances[b][0], a, b, separation[0],
                                geometry, logarithm, pair);
        tile_valid = tile_valid && valid && std::isfinite(state.factor[0]);
        if (pass == 1) {
          std::array<double, 4> pullback{};
          if (state.factor[1] != 0) {
            const double bar_mu = pair_adjoint<false>(state, a, b, logs, products,
                bar_product, zeros, maximum, logarithm);
            pullback[0] = bar_mu * state.ratio[0];
            for (size_t k = 0; k < 3; ++k)
              pullback[k + 1] = bar_mu * state.ratio[1] * separation[k + 1];
          }
          state.ratio = {pullback[0], pullback[1]};
          state.factor = {pullback[2], pullback[3]};
        }
        states[index] = state;
      }
      // The team vote publishes the strip and makes failures collective without
      // every participant rescanning all pair states.
      if (!team.all(tile_valid)) return false;
      for (size_t atom = team.rank(); atom < na; atom += team.size()) {
        // Lower neighbors exist only in atom's own row; upper neighbors exist
        // only in the strip's later rows. Every edge is read twice in total.
        const size_t lower = atom >= begin && atom < end ? atom : 0;
        const size_t upper_begin = std::max(atom + 1, begin);
        const size_t upper = end > upper_begin ? end - upper_begin : 0;
        for (size_t incident = 0; incident < lower + upper; ++incident) {
          const bool side = incident >= lower;
          const size_t a = side ? upper_begin + incident - lower : atom;
          const size_t b = side ? atom : incident;
          const auto& state = states[center_pair_index(a, b) - offset];
          if (pass == 0) {
            const double v = side ? 1 - state.factor[0] : state.factor[0];
            if (v > 0) logs[atom] += state.logarithm[side][0];
            else ++zeros[atom];
          } else {
            const double sign = side ? -1 : 1;
            bar_distance[atom] += sign * state.ratio[0];
            gradient[3 * atom] += sign * state.ratio[1];
            gradient[3 * atom + 1] += sign * state.factor[0];
            gradient[3 * atom + 2] += sign * state.factor[1];
          }
        }
      }
      team.sync();  // all incident-edge consumers finish before strip reuse
    }
    if (pass == 0) {
      maximum = maximum_log_product(na, logs, zeros);
      if (!std::isfinite(maximum)) return false;
      if (team.rank() == 0)
        normalized_product_adjoint(na, owner, seed, logs, products, bar_product, zeros,
                                   maximum, ratio);
      team.sync();
    }
  }
  if (team.rank() == 0) point_motion_adjoint(na, owner, bar_distance, distances, gradient);
  team.sync();
  return true;
}
template <class Team, class Norm, class Ratio, class Log, class Pair, class PreparedRatio>
GENERATIVEQC_GRID_HD bool contract_point_tiled_cooperative(
    const double* point, const double* centers, size_t na, size_t owner, double seed,
    double* gradient, double* logs, double* products, double* bar_product,
    double* bar_distance, size_t* zeros, std::array<double, 4>* distances,
    PointPair* states, size_t rows, Team team, Norm norm, Ratio ratio, Log logarithm, Pair pair,
    const CenterPair* pairs, PreparedRatio prepared_ratio) {
  if (pairs)
    return contract_point_tiled_cooperative_impl(point, centers, na, owner, seed, gradient,
        logs, products, bar_product, bar_distance, zeros, distances, states, rows, team,
        norm, ratio, logarithm, pair, PreparedCenterGeometry<PreparedRatio>{pairs, prepared_ratio});
  return contract_point_tiled_cooperative_impl(point, centers, na, owner, seed, gradient,
      logs, products, bar_product, bar_distance, zeros, distances, states, rows, team,
      norm, ratio, logarithm, pair, DirectCenterGeometry<Norm, Ratio>{centers, norm, ratio});
}
}  // namespace generativeqc_grid_adjoint
#undef GENERATIVEQC_GRID_HD
"""


def emit_grid_adjoint() -> str:
    """Emit the shared bounded Becke normalized-product reverse traversal."""
    return _GRID_ADJOINT_SOURCE


def _emit_prepared_ratio(*, device: bool) -> str:
    """Cut geometry-only nodes from the existing ratio AD graph, without rewriting it."""
    program = grid_response_program("ratio")
    graph, primal = program.graph, program.roots[0]
    roots = (
        primal,
        *(graph.differentiate(primal, graph.variable(n)) for n in ("a", "b")),
    )
    order = tuple(graph.topological_order(roots))
    dependent = {}
    for identifier in order:
        node = graph.nodes[identifier]
        dependent[identifier] = (
            node.operation == "variable"
            and node.payload == "a"
            or any(dependent[argument] for argument in node.arguments)
        )
    boundary = {
        argument
        for identifier in order
        if dependent[identifier]
        for argument in graph.nodes[identifier].arguments
        if not dependent[argument]
        and graph.nodes[argument].operation not in ("constant", "variable")
    }
    boundary.update(root.identifier for root in roots if not dependent[root.identifier])
    # The retained ABI is reciprocal(R), -pow(R, -2), including the original
    # operation order. Fail generation if the authoritative graph changes it.
    retained = tuple(identifier for identifier in order if identifier in boundary)
    if len(retained) != 2:
        raise ValueError("unexpected geometry-only ratio graph boundary")
    from generativeqc_compiler.integral.expr import Expr

    expressions = tuple(Expr(graph, identifier) for identifier in retained)
    prefix = "__device__" if device else "static"
    prepare = ScalarCEmitter(graph, {"b": "b"})
    prepare.emit(expressions)
    consume = ScalarCEmitter(graph, {"a": "a", "b": "b"})
    # Bind every invariant node, not just the cut, so emission cannot leave dead
    # reciprocal/pow work in the pointwise body.
    for identifier in order:
        if not dependent[identifier]:
            consume.names[identifier] = "unused_geometry_node"
    for index, identifier in enumerate(retained):
        consume.names[identifier] = f"geometry[{index}]"
    consume.emit(roots)
    if "unused_geometry_node" in "\n".join(consume.lines):
        raise ValueError("incomplete geometry-only ratio graph boundary")
    return (
        "\n".join(
            [
                f"{prefix} std::array<double, 2> local_ratio_geometry(double b) {{",
                *prepare.lines,
                "return {"
                + ", ".join(prepare.reference(root) for root in expressions)
                + "};",
                "}",
                f"{prefix} std::array<double, 3> local_ratio_prepared(double a, const double* geometry) {{",
                *consume.lines,
                "return {"
                + ", ".join(consume.reference(root) for root in roots)
                + "};",
                "}",
            ]
        )
        + "\n"
    )


def emit_grid_partials(
    iterations: typing.Any = 3, *, device: typing.Any = False
) -> typing.Any:
    """Shared local AD construction for CPU and CUDA traversal owners."""
    lines = []
    identities = {}
    lines.append(_emit_prepared_ratio(device=device))
    for kind, names in (
        ("norm", ("x", "y", "z")),
        ("ratio", ("a", "b")),
        ("log", ("p",)),
        ("becke", ("mu",)),
    ):
        program = grid_response_program(kind, iterations)
        graph, primal = program.graph, program.roots[0]
        roots = (
            primal,
            *(graph.differentiate(primal, graph.variable(name)) for name in names),
        )
        emitter = ScalarCEmitter(graph, {name: name for name in names})
        emitter.emit(roots)
        lines += [
            f"{'__device__' if device else 'static'} std::array<double, {len(roots)}> local_{kind}({', '.join('double ' + name for name in names)}) {{",
            *emitter.lines,
            "return {" + ", ".join(emitter.reference(root) for root in roots) + "};",
            "}",
        ]
        identities[kind] = program.identity
    identity = canonical_hash({"schema": "grid-cpu-adjoint-v1", "programs": identities})
    lines.append(f"// Shared grid graphs: {identity}")
    return "\n".join(lines) + "\n"


def emit_grid_contraction(iterations: typing.Any = 3) -> typing.Any:
    """Generate the bounded Becke adjoint from compiler-owned composition."""
    lines = [
        emit_grid_adjoint(),
        '#include "grid_response_cpu.hpp"',
        emit_grid_partials(iterations),
    ]
    lines += [
        'extern "C" int grid_contract(const double* points, size_t np, const double* centers, size_t na, const int64_t* owners, const double* seeds, double* output, size_t no, size_t budget, size_t max_pairs, double tolerance) noexcept {',
        "return generativeqc_grid_cpu::contract(points, np, centers, na, owners, seeds, output, no, budget, max_pairs, tolerance, local_norm, local_ratio, local_log, local_becke, local_ratio_geometry, local_ratio_prepared);",
        "}",
    ]
    return "\n".join(lines) + "\n"


class NativeGridContraction:
    """Strict FP64 CPU ABI; seeds are energy-density times raw atomic measure.

    Points move with their integer owner. The result contracts into ALL nuclear
    coordinates, with no point/atom/coordinate Jacobian and no scalar interpreter.
    max_bytes covers native scratch plus conservative adapter staging; it is not
    a simultaneous endpoint/SCF resource reservation.
    """

    def __init__(
        self,
        *,
        compiler: typing.Any,
        cache: typing.Any,
        iterations: typing.Any = 3,
        max_bytes: typing.Any = 8 * 1024 * 1024,
        max_pair_visits: typing.Any = 100_000_000,
    ) -> None:
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("native grid requires a CPU compiler adapter")
        checked_int(max_bytes, "grid byte budget", high=(1 << 63) - 1)
        checked_int(max_pair_visits, "grid work budget", low=0, high=(1 << 63) - 1)
        self.max_pair_visits = max_pair_visits
        self.max_bytes = max_bytes
        source = emit_grid_contraction(iterations)
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (canonical_hash(source) + ".cpp")
        cache_source(path, source)
        header = asset_path("src/dft/grid_response_cpu.hpp")
        self.artifact = compile_runtime(
            compiler,
            cache,
            path,
            headers=(header,),
            options=("-ffp-contract=off", f"-I{header.parent}"),
        )
        self.identity = self.artifact.metadata["key"]
        self.library = ct.CDLL(str(self.artifact.library))
        self.call = self.library.grid_contract
        self.call.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_int64),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_double,
        ]
        self.call.restype = ct.c_int

    def contract(
        self,
        points: typing.Any,
        centers: typing.Any,
        owners: typing.Any,
        seeds: typing.Any,
        *,
        coincident_tolerance: typing.Any = 1e-12,
    ) -> typing.Any:
        """Return a detached gradient, publishing nothing on a late native error."""
        points, centers, owners, seeds = map(
            np.asarray, (points, centers, owners, seeds)
        )
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or centers.ndim != 2
            or centers.shape[1:] != (3,)
            or not len(centers)
        ):
            raise ValueError("grid points/centers require (n,3) and nonempty centers")
        npnt, natom = len(points), len(centers)
        if (1 + 2 * npnt) * natom * (natom - 1) // 2 > self.max_pair_visits:
            raise ValueError("grid work budget exceeded")
        if 8 * (10 * npnt + 30 * natom) > self.max_bytes:
            raise ValueError("grid byte budget exceeded")
        if (
            owners.shape != (npnt,)
            or owners.dtype != np.int64
            or seeds.shape != (npnt,)
        ):
            raise ValueError("grid owners require int64 and seeds require point shape")
        if any(
            v.dtype != np.float64 or not np.isfinite(v).all()
            for v in (points, centers, seeds)
        ):
            raise ValueError("grid inputs require finite float64")
        points, centers, owners, seeds = map(
            np.ascontiguousarray, (points, centers, owners, seeds)
        )
        output = np.empty((natom, 3))
        ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
        code = self.call(
            ptr(points),
            npnt,
            ptr(centers),
            natom,
            owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
            ptr(seeds),
            ptr(output),
            output.size,
            self.max_bytes,
            self.max_pair_visits,
            coincident_tolerance,
        )
        if code:
            raise ValueError(
                f"native grid contraction failed ({code}); invalid or nonsmooth inputs"
            )
        return immutable(output)

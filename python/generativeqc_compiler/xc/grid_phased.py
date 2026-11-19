"""Bounded lifetime plan for the shared Becke reverse graph.

The owner must admit this entire scratch reservation in
addition to its other live storage, or use the existing bounded strip schedule.
All scalar mathematics comes from grid_native/grid_response. Only the storage,
phase boundaries, and independent iteration domains differ here.
"""

from __future__ import annotations

from dataclasses import dataclass

from generativeqc_compiler.dft.grid import checked_int


@dataclass(frozen=True)
class PhasedBeckePlan:
    """Scratch and semantic work for one tile, not endpoint FLOPs or peak usage.

    Pair factors and logarithms are overwritten by four pullbacks only after every
    atom log-product and point normalization consumer has completed. Same-stream
    phase boundaries protect this alias. Point lanes are contiguous in all fields.
    Center preparation and deterministic output reduction belong to the owner.
    """

    atoms: int
    points: int

    @property
    def pairs(self) -> int:
        return self.atoms * (self.atoms - 1) // 2

    @property
    def pair_bytes(self) -> int:
        return 4 * 8 * self.pairs * self.points

    @property
    def scratch_bytes(self) -> int:
        return self.pair_bytes + (12 * self.atoms + 1) * self.points * 8

    @property
    def pair_evaluations(self) -> int:
        return self.pairs * self.points

    @property
    def phases(self) -> tuple[str, ...]:
        return (
            "distance",
            "pair_primal",
            "atom_logs",
            "point_normalize",
            "pair_reverse",
            "atom_gather",
            "point_motion",
        )


def plan_phased_becke(
    *, atoms: int, points: int, budget_bytes: int, occupied_bytes: int = 0
) -> PhasedBeckePlan | None:
    """Admit a whole tile; None requires the caller's bounded strip fallback.

    Occupied bytes must include input/output, center metadata, and all concurrent
    endpoint owners. No claim that this isolated plan is an endpoint admission.
    """
    for name, value, low in (
        ("atoms", atoms, 1),
        ("points", points, 0),
        ("budget_bytes", budget_bytes, 0),
        ("occupied_bytes", occupied_bytes, 0),
    ):
        checked_int(value, name, low=low, high=(1 << 63) - 1)
    plan = PhasedBeckePlan(atoms, points)
    return plan if occupied_bytes + plan.scratch_bytes <= budget_bytes else None


_PHASED_SOURCE = r"""
#if defined(__CUDACC__)
#define GENERATIVEQC_PHASE_HD __host__ __device__
#else
#define GENERATIVEQC_PHASE_HD
#endif
namespace generativeqc_grid_phased {
using namespace generativeqc_grid_adjoint;

template <class Value> struct Strided {
  Value* data;
  size_t stride;
  GENERATIVEQC_PHASE_HD Value& operator[](size_t index) const {
    return data[index * stride];
  }
};

// Distances[4], logs, products, bar_product, bar_distance, gradient[3].
// Zeros use a separate size_t panel; maximum is one double per point. Owners
// allocate all panels, and must not publish any output after a failed phase.
struct Workspace {
  size_t atoms, points;
  double* pairs;
  double* fields;
  size_t* zeros;
  double* maximum;
  GENERATIVEQC_PHASE_HD Strided<double> field(size_t word, size_t point) const {
    return {fields + word * atoms * points + point, points};
  }
  GENERATIVEQC_PHASE_HD Strided<size_t> zero_counts(size_t point) const {
    return {zeros + point, points};
  }
  GENERATIVEQC_PHASE_HD double& pair(size_t word, size_t index, size_t point) const {
    return pairs[(word * (atoms * (atoms - 1) / 2) + index) * points + point];
  }
};

template <class Norm>
GENERATIVEQC_PHASE_HD bool distance_phase(Workspace work, size_t point, size_t atom,
    const double* points, const double* centers, Norm norm) {
  bool valid = true;
  const auto values = distance(points + 3 * point, centers + 3 * atom, norm, valid);
  for (size_t word = 0; word < 4; ++word) work.field(word, point)[atom] = values[word];
  return valid;
}

template <class Geometry, class Log, class Pair>
GENERATIVEQC_PHASE_HD bool pair_primal_phase(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log logarithm, Pair pair) {
  bool valid = true;
  const auto separation = geometry.separation(first, second, valid);
  const auto state = point_pair(work.field(0, point)[first] -
      work.field(0, point)[second], first, second, separation[0], geometry, logarithm, pair);
  const size_t index = center_pair_index(first, second);
  // Produce each logarithm in the parallel pair domain. The atom traversal
  // below retains its ordered sum, but no longer serializes transcendental
  // evaluation across all neighbors. Reuse the two old ratio words: those
  // cheap partials can be rematerialized from the same canonical AD graph.
  work.pair(0, index, point) = state.logarithm[0][0];
  work.pair(1, index, point) = state.logarithm[1][0];
  work.pair(2, index, point) = state.factor[0];
  work.pair(3, index, point) = state.factor[1];
  return valid && std::isfinite(state.factor[0]);
}

GENERATIVEQC_PHASE_HD void atom_logs_phase(Workspace work, size_t point,
    size_t atom) {
  double sum = 0;
  size_t zeros = 0;
  // Lower neighbors followed by upper neighbors reproduce triangular order.
  for (size_t neighbor = 0; neighbor < work.atoms; ++neighbor) {
    if (neighbor == atom) continue;
    const bool upper = neighbor > atom;
    const size_t index = center_pair_index(std::max(atom, neighbor), std::min(atom, neighbor));
    const double factor = work.pair(2, index, point);
    const double value = upper ? 1 - factor : factor;
    if (value > 0) sum += work.pair(upper ? 1 : 0, index, point);
    else ++zeros;
  }
  work.field(4, point)[atom] = sum;
  work.zero_counts(point)[atom] = zeros;
}

template <class Ratio>
GENERATIVEQC_PHASE_HD bool point_normalize_phase(Workspace work, size_t point,
    size_t owner, double seed, Ratio ratio) {
  if (owner >= work.atoms || !std::isfinite(seed)) return false;
  const double maximum = maximum_log_product(work.atoms, work.field(4, point),
                                             work.zero_counts(point));
  if (!std::isfinite(maximum)) return false;
  work.maximum[point] = maximum;
  normalized_product_adjoint(work.atoms, owner, seed, work.field(4, point),
      work.field(5, point), work.field(6, point), work.zero_counts(point), maximum, ratio);
  return true;
}

struct PairPullbackCoefficients {
  std::array<double, 2> values{};
  std::array<double, 4> separation{};
};

GENERATIVEQC_PHASE_HD inline double pair_pullback_component(double difference,
    double separation, size_t word, double direction) {
  return word ? separation * direction : difference;
}

template <class Geometry, class Log, class Bars>
GENERATIVEQC_PHASE_HD bool pair_pullback_coefficients(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log logarithm, Bars bars,
    PairPullbackCoefficients& result) {
  // A saturated pair must overwrite a reused output with exact zero state.
  result = {};
  const size_t index = center_pair_index(first, second);
  PointPair state{{}, {work.pair(2, index, point), work.pair(3, index, point)}, {}};
  bool valid = true;
  if (state.factor[1] != 0) {
    result.separation = geometry.separation(first, second, valid);
    const auto coordinate = geometry.coordinate(work.field(0, point)[first] -
        work.field(0, point)[second], first, second, result.separation[0]);
    state.ratio = {coordinate[1], coordinate[2]};
    const double bar_mu = pair_adjoint<false>(state, first, second,
        work.field(4, point), work.field(5, point), bars,
        work.zero_counts(point), work.maximum[point], logarithm);
    result.values = {bar_mu * state.ratio[0], bar_mu * state.ratio[1]};
  }
  for (double value : result.values) valid = valid && std::isfinite(value);
  return valid;
}

template <class Geometry, class Log, class Bars>
GENERATIVEQC_PHASE_HD bool pair_reverse_with_bars_phase(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log logarithm, Bars bars) {
  PairPullbackCoefficients coefficients;
  bool valid = pair_pullback_coefficients(work, point, first, second, geometry,
                                         logarithm, bars, coefficients);
  const size_t index = center_pair_index(first, second);
  // All primal consumers finished at the normalization boundary. Reuse storage.
  for (size_t word = 0; word < 4; ++word) {
    const double value = pair_pullback_component(coefficients.values[0],
        coefficients.values[1], word, coefficients.separation[word]);
    work.pair(word, index, point) = value;
    valid = valid && std::isfinite(value);
  }
  return valid;
}

template <class Geometry, class Log>
GENERATIVEQC_PHASE_HD bool pair_reverse_phase(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log logarithm) {
  return pair_reverse_with_bars_phase(work, point, first, second, geometry,
                                     logarithm, work.field(6, point));
}

struct AllNeighbors {
  GENERATIVEQC_PHASE_HD size_t operator()(size_t index) const { return index; }
};

struct PairWords {
  Workspace work;
  size_t point;
  GENERATIVEQC_PHASE_HD double operator()(size_t word, size_t index) const {
    return work.pair(word, index, point);
  }
};

template <class Neighbors, class Values>
GENERATIVEQC_PHASE_HD void atom_gather_with_values_phase(Workspace work, size_t point,
    size_t atom, size_t count, Neighbors neighbors, Values values) {
  std::array<double, 4> pullback{};
  for (size_t cursor = 0; cursor < count; ++cursor) {
    const size_t neighbor = neighbors(cursor);
    if (neighbor == atom) continue;
    const size_t index = center_pair_index(std::max(atom, neighbor), std::min(atom, neighbor));
    const double sign = neighbor > atom ? -1 : 1;
    for (size_t word = 0; word < 4; ++word)
      pullback[word] += sign * values(word, index);
  }
  for (size_t word = 0; word < 4; ++word) work.field(7 + word, point)[atom] = pullback[word];
}

template <class Neighbors>
GENERATIVEQC_PHASE_HD void atom_gather_selected_phase(Workspace work, size_t point,
    size_t atom, size_t count, Neighbors neighbors) {
  atom_gather_with_values_phase(work, point, atom, count, neighbors, PairWords{work, point});
}

GENERATIVEQC_PHASE_HD void atom_gather_phase(Workspace work, size_t point, size_t atom) {
  atom_gather_selected_phase(work, point, atom, work.atoms, AllNeighbors{});
}

GENERATIVEQC_PHASE_HD bool point_motion_phase(Workspace work, size_t point,
    size_t atom, size_t owner) {
  bool valid = true;
  for (size_t axis = 0; axis < 3; ++axis) {
    double result = work.field(8 + axis, point)[atom];
    if (atom == owner) {
      for (size_t source = 0; source < work.atoms; ++source) {
        const double value = work.field(7, point)[source] * work.field(axis + 1, point)[source];
        if (source == owner) result -= value;
        result += value;
      }
    } else result -= work.field(7, point)[atom] * work.field(axis + 1, point)[atom];
    work.field(8 + axis, point)[atom] = result;
    valid = valid && std::isfinite(result);
  }
  return valid;
}

// Analytic route: forward-mode partition derivatives with no pair panel.
// Lane (point, X) forms P_X = prod_{Y!=X} s_XY in the phased neighbor order as
// a binary-scaled product of its nonzero factors plus a zero count. A point lane
// normalizes by the largest nonzero exponent. Lane (point, G!=owner) sums
// dP_B/dG over B with the same point_pair switch and coordinate partials; the
// owner takes -sum_G by translation invariance of the moving grid. Fields:
// 4 mantissa, 6 exponent, 5 normalized P (0 with a zero factor), 7 normalized
// nonzero-factor product, maximum[point] = 1 / sum_X P_X.
// Prepared center pairs for one lane atom whose elemental adjustments against
// every neighbor are evaluated once (null keeps equal radii). adjustments[Y]
// is coordinate_adjustment(max(atom, Y), min(atom, Y)), the canonical order
// in which point_pair queries the pair.
template <class PreparedRatio>
struct LaneAdjustedGeometry {
  const generativeqc_grid_adjoint::CenterPair* pairs;
  PreparedRatio ratio;
  const double* adjustments;
  size_t atom;
  GENERATIVEQC_PHASE_HD std::array<double, 4> separation(size_t a, size_t b, bool&) const {
    return pairs[generativeqc_grid_adjoint::center_pair_index(a, b)].distance;
  }
  GENERATIVEQC_PHASE_HD std::array<double, 3> coordinate(double difference, size_t a, size_t b,
                                                         double) const {
    const auto& stored = pairs[generativeqc_grid_adjoint::center_pair_index(a, b)].ratio;
    const double narrowed[2]{stored[0], stored[1]};
    const auto value = ratio(difference, narrowed);
    return adjustments ? generativeqc_grid_adjoint::adjust_coordinate_with(
                             value, adjustments[a == atom ? b : a])
                       : value;
  }
};

GENERATIVEQC_PHASE_HD inline double analytic_frexp(double value, int* exponent) {
#if defined(__CUDA_ARCH__)
  return ::frexp(value, exponent);
#else
  return std::frexp(value, exponent);
#endif
}
GENERATIVEQC_PHASE_HD inline double analytic_ldexp(double value, int exponent) {
#if defined(__CUDA_ARCH__)
  return ::ldexp(value, exponent);
#else
  return std::ldexp(value, exponent);
#endif
}

template <class Geometry, class Log, class Pair>
GENERATIVEQC_PHASE_HD bool analytic_products_phase(Workspace work, size_t point, size_t atom,
    Geometry geometry, Log logarithm, Pair pair) {
  const auto distance = work.field(0, point);
  const double own = distance[atom];
  bool valid = true;
  double mantissa = 1;
  int exponent = 0;
  size_t zeros = 0;
  for (size_t neighbor = 0; neighbor < work.atoms; ++neighbor) {
    if (neighbor == atom) continue;
    const bool upper = neighbor > atom;
    const size_t first = upper ? neighbor : atom, second = upper ? atom : neighbor;
    const double other = distance[neighbor];
    const auto separation = geometry.separation(first, second, valid);
    const auto state = point_pair<false>(upper ? other - own : own - other, first, second,
                                         separation[0], geometry, logarithm, pair);
    valid = valid && std::isfinite(state.factor[0]);
    const double value = upper ? 1 - state.factor[0] : state.factor[0];
    if (value > 0) {
      int shift;
      mantissa = analytic_frexp(mantissa * value, &shift);
      exponent += shift;
    } else {
      ++zeros;
    }
  }
  work.field(4, point)[atom] = mantissa;
  work.field(6, point)[atom] = exponent;
  work.zero_counts(point)[atom] = zeros;
  return valid;
}

GENERATIVEQC_PHASE_HD inline bool analytic_normalize_phase(Workspace work, size_t point,
    size_t owner, double seed) {
  const auto mantissa = work.field(4, point), exponent = work.field(6, point);
  const auto zeros = work.zero_counts(point);
  double largest = -std::numeric_limits<double>::infinity();
  for (size_t atom = 0; atom < work.atoms; ++atom)
    if (!zeros[atom]) largest = std::max(largest, exponent[atom]);
  if (owner >= work.atoms || !std::isfinite(seed) || !std::isfinite(largest)) return false;
  double sum = 0;
  for (size_t atom = 0; atom < work.atoms; ++atom) {
    const double scaled = analytic_ldexp(mantissa[atom], int(exponent[atom] - largest));
    work.field(7, point)[atom] = scaled;
    const double product = zeros[atom] ? 0.0 : scaled;
    work.field(5, point)[atom] = product;
    sum += product;
  }
  work.maximum[point] = 1 / sum;
  return sum > 0 && std::isfinite(sum);
}

// One pair (G, B) of an analytic motion lane. d s_BG / dG = f1 (r0 u_G - r1 e)
// in both orientations, and s_GB = 1 - s_BG has the opposite slope. Each scale
// keeps the factor value its product lane used; a zero factor takes the
// single-zero excluded product.
struct AnalyticPairMotion {
  double neighbor_scale, atom_scale, along, across;
  std::array<double, 4> separation;
};

template <class Geometry, class Log, class Pair, class Values, class Counts>
GENERATIVEQC_PHASE_HD bool analytic_pair_motion(size_t atom, size_t neighbor, double own,
    double other, double atom_product, Values products, Values partial_products, Counts zeros,
    Geometry geometry, Log logarithm, Pair pair, bool& valid, AnalyticPairMotion& result) {
  const bool upper = neighbor > atom;
  const size_t first = upper ? neighbor : atom, second = upper ? atom : neighbor;
  result.separation = geometry.separation(first, second, valid);
  const auto state = point_pair<false>(upper ? other - own : own - other, first, second,
                                       result.separation[0], geometry, logarithm, pair);
  if (state.factor[1] == 0) return false;  // Saturated switch: exact zero pullback.
  const double neighbor_factor = upper ? state.factor[0] : 1 - state.factor[0];
  const double atom_factor = upper ? 1 - state.factor[0] : state.factor[0];
  const double joint = neighbor_factor * atom_factor;
  if (joint >= 2.2250738585072014e-308) {
    // Both factors are nonzero, so a zero-count product is stored as exact zero.
    // One reciprocal serves both factors while their product stays normal.
    const double reciprocal = 1 / joint;
    result.neighbor_scale = products[neighbor] * atom_factor * reciprocal;
    result.atom_scale = -atom_product * neighbor_factor * reciprocal;
  } else {
    result.neighbor_scale = result.atom_scale = 0;
    if (!zeros[neighbor]) result.neighbor_scale = products[neighbor] / neighbor_factor;
    else if (zeros[neighbor] == 1 && neighbor_factor == 0)
      result.neighbor_scale = partial_products[neighbor];
    if (!zeros[atom]) result.atom_scale = -atom_product / atom_factor;
    else if (zeros[atom] == 1 && atom_factor == 0) result.atom_scale = -partial_products[atom];
  }
  result.along = state.factor[1] * state.ratio[0];
  result.across = state.factor[1] * state.ratio[1];
  return true;
}

// Writes seed * dw/dG for one atom G != owner into output[0..2]. The neighbor
// sum keeps the u_G coefficient and the e vector apart; the owner pair is
// evaluated once more after it.
template <class Geometry, class Log, class Pair>
GENERATIVEQC_PHASE_HD bool analytic_motion_phase(Workspace work, size_t point, size_t atom,
    size_t owner, double seed, Geometry geometry, Log logarithm, Pair pair, double* output) {
  const auto distance = work.field(0, point);
  const auto products = work.field(5, point), partial_products = work.field(7, point);
  const auto zeros = work.zero_counts(point);
  const double own = distance[atom], atom_product = products[atom];
  bool valid = true;
  double along = 0, across[3]{};
  AnalyticPairMotion term;
  for (size_t neighbor = 0; neighbor < work.atoms; ++neighbor) {
    if (neighbor == atom ||
        !analytic_pair_motion(atom, neighbor, own, distance[neighbor], atom_product, products,
                              partial_products, zeros, geometry, logarithm, pair, valid, term))
      continue;
    const double scale = term.neighbor_scale + term.atom_scale;
    along += scale * term.along;
    for (size_t axis = 0; axis < 3; ++axis)
      across[axis] += scale * term.across * term.separation[1 + axis];
  }
  double owner_term[3]{};
  if (analytic_pair_motion(atom, owner, own, distance[owner], atom_product, products,
                           partial_products, zeros, geometry, logarithm, pair, valid, term))
    for (size_t axis = 0; axis < 3; ++axis)
      owner_term[axis] = term.neighbor_scale * (term.along * work.field(1 + axis, point)[atom] -
                                                term.across * term.separation[1 + axis]);
  const double inverse = work.maximum[point];
  const double weight = products[owner] * inverse;
  for (size_t axis = 0; axis < 3; ++axis) {
    const double others = along * work.field(1 + axis, point)[atom] - across[axis];
    const double value = seed * inverse * (owner_term[axis] - weight * others);
    valid = valid && std::isfinite(value);
    output[axis] = value;
  }
  return valid;
}

// The moving owner receives minus the sum over every other atom, in atom order.
GENERATIVEQC_PHASE_HD inline void analytic_owner_phase(size_t atoms, size_t owner, double* row) {
  for (size_t axis = 0; axis < 3; ++axis) {
    double sum = 0;
    for (size_t atom = 0; atom < atoms; ++atom)
      if (atom != owner) sum += row[3 * atom + axis];
    row[3 * owner + axis] = -sum;
  }
}
} // namespace generativeqc_grid_phased
#undef GENERATIVEQC_PHASE_HD
"""


def emit_phased_becke() -> str:
    """Emit schedule-only helpers after emit_grid_adjoint; never load a runtime."""
    return _PHASED_SOURCE

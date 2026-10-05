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

template <class Geometry, class Log>
GENERATIVEQC_PHASE_HD bool pair_reverse_phase(Workspace work, size_t point,
    size_t first, size_t second, Geometry geometry, Log logarithm) {
  const size_t index = center_pair_index(first, second);
  PointPair state{{}, {work.pair(2, index, point), work.pair(3, index, point)}, {}};
  std::array<double, 4> pullback{};
  bool valid = true;
  if (state.factor[1] != 0) {
    const auto separation = geometry.separation(first, second, valid);
    const auto coordinate = geometry.coordinate(work.field(0, point)[first] -
        work.field(0, point)[second], first, second, separation[0]);
    state.ratio = {coordinate[1], coordinate[2]};
    const double bar_mu = pair_adjoint<false>(state, first, second,
        work.field(4, point), work.field(5, point), work.field(6, point),
        work.zero_counts(point), work.maximum[point], logarithm);
    pullback[0] = bar_mu * state.ratio[0];
    for (size_t axis = 0; axis < 3; ++axis)
      pullback[axis + 1] = bar_mu * state.ratio[1] * separation[axis + 1];
  }
  // All primal consumers finished at the normalization boundary. Reuse storage.
  for (size_t word = 0; word < 4; ++word) {
    work.pair(word, index, point) = pullback[word];
    valid = valid && std::isfinite(pullback[word]);
  }
  return valid;
}

GENERATIVEQC_PHASE_HD void atom_gather_phase(Workspace work, size_t point, size_t atom) {
  std::array<double, 4> pullback{};
  for (size_t neighbor = 0; neighbor < work.atoms; ++neighbor) {
    if (neighbor == atom) continue;
    const size_t index = center_pair_index(std::max(atom, neighbor), std::min(atom, neighbor));
    const double sign = neighbor > atom ? -1 : 1;
    for (size_t word = 0; word < 4; ++word)
      pullback[word] += sign * work.pair(word, index, point);
  }
  for (size_t word = 0; word < 4; ++word) work.field(7 + word, point)[atom] = pullback[word];
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
} // namespace generativeqc_grid_phased
#undef GENERATIVEQC_PHASE_HD
"""


def emit_phased_becke() -> str:
    """Emit schedule-only helpers after emit_grid_adjoint; never load a runtime."""
    return _PHASED_SOURCE

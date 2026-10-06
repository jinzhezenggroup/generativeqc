"""Dense Becke reverse schedule with two shared-AD coefficients per pair.

This is a storage/dataflow experiment, not a new partition formula. It keeps
the parallel pair domain and the ordinary normalization. Only validated cached
center geometry can expand coefficients during the existing ordered gather.
Whole-composition recognition and native-owner promotion remain separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from generativeqc_compiler.xc.becke_partition import (
    BeckePartitionDerivativeOp,
    validate_becke_partition_derivative,
)
from generativeqc_compiler.xc.grid_phased import (
    PhasedBeckePlan,
    emit_phased_becke,
    plan_phased_becke,
)


@dataclass(frozen=True)
class BeckePairCoefficientPlan:
    """Keep the four-word primal reservation, overwrite only two reverse words.

    Logical reverse writes and gather reads exclude geometry metadata and cache
    transactions. Gather additionally reads the three center unit components
    per incident pair. Neither these models nor scratch bytes measure an
    endpoint's concurrent peak; the native owner must admit all live storage.
    """

    operation: BeckePartitionDerivativeOp
    phased: PhasedBeckePlan

    @property
    def scratch_bytes(self) -> int:
        return self.phased.scratch_bytes

    @property
    def reverse_pair_visits(self) -> int:
        return self.phased.pair_evaluations

    @property
    def reverse_write_bytes(self) -> int:
        return 2 * 8 * self.reverse_pair_visits

    @property
    def gather_pair_read_bytes(self) -> int:
        return 2 * 2 * 8 * self.reverse_pair_visits

    @property
    def gather_geometry_read_bytes(self) -> int:
        return 2 * 3 * 8 * self.reverse_pair_visits


def plan_becke_pair_coefficients(
    *,
    operation: BeckePartitionDerivativeOp,
    atoms: int,
    points: int,
    budget_bytes: int,
    cached_geometry: bool,
    occupied_bytes: int = 0,
) -> BeckePairCoefficientPlan | None:
    """Fail closed without cached geometry or whole concurrent admission.

    The owner must prepare/validate and rebind center geometry before execution,
    and include that metadata in occupied_bytes. A false precondition requires
    the ordinary phased/generic route, never on-demand gather geometry rebuilds.
    """
    if type(cached_geometry) is not bool:
        raise ValueError("cached_geometry must be a boolean")
    phased = plan_phased_becke(
        atoms=atoms,
        points=points,
        budget_bytes=budget_bytes,
        occupied_bytes=occupied_bytes,
    )
    if (
        phased is None
        or not cached_geometry
        or operation.atoms is not None
        and operation.atoms != atoms
    ):
        return None
    return BeckePairCoefficientPlan(operation, phased)


_SOURCE = r"""
#if defined(__CUDACC__)
#define GENERATIVEQC_COEFFICIENT_HD __host__ __device__
#else
#define GENERATIVEQC_COEFFICIENT_HD
#endif
namespace generativeqc_grid_coefficients {
using namespace generativeqc_grid_phased;

template <class Geometry, class Log>
GENERATIVEQC_COEFFICIENT_HD bool pair_coefficient_reverse_phase(Workspace work,
    size_t point, size_t first, size_t second, Geometry geometry, Log logarithm) {
  PairPullbackCoefficients coefficients;
  const bool valid = pair_pullback_coefficients(work, point, first, second, geometry,
      logarithm, work.field(6, point), coefficients);
  const size_t index = center_pair_index(first, second);
  // Primal consumers are complete. Retain the same peak reservation but write
  // only the scalar distance/separation pullbacks, not a full xyz pair cache.
  for (size_t word = 0; word < 2; ++word)
    work.pair(word, index, point) = coefficients.values[word];
  return valid;
}

struct CoefficientPairWords {
  Workspace work;
  size_t point;
  const CenterPair* centers;
  GENERATIVEQC_COEFFICIENT_HD double operator()(size_t word, size_t index) const {
    // Cached center directions are already finite unit components. Expand in
    // exactly the dense reverse's multiplication order before the ordered sum.
    if (!word) return work.pair(0, index, point);
    return pair_pullback_component(0.0, work.pair(1, index, point), word,
                                   centers[index].distance[word]);
  }
};

GENERATIVEQC_COEFFICIENT_HD void atom_gather_coefficient_phase(Workspace work,
    size_t point, size_t atom, const CenterPair* centers) {
  atom_gather_with_values_phase(work, point, atom, work.atoms, AllNeighbors{},
                               CoefficientPairWords{work, point, centers});
}
} // namespace generativeqc_grid_coefficients
#undef GENERATIVEQC_COEFFICIENT_HD
"""


def emit_becke_pair_coefficients(operation: BeckePartitionDerivativeOp) -> str:
    """Emit only shared AD owners plus a dense coefficient storage schedule."""
    validate_becke_partition_derivative(operation)
    return (
        emit_phased_becke()
        + f"// Canonical Becke coefficient operation: {operation.identity}\n"
        + _SOURCE
    )

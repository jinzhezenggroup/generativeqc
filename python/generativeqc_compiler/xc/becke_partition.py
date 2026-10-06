"""Recognize canonical Becke AD and plan exact normalized-product dependencies.

This candidate changes the reverse iteration domain, not the switch, logarithm,
ratio, saturation or single-zero pullback. It still executes the shared emitted
AD primitives. Runtime owners must admit the index panels and keep the ordinary
phased/generic routes for unsupported graphs or insufficient concurrent memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.grid import checked_int
from generativeqc_compiler.xc.grid_partition_ir import (
    grid_partition_program,
    partition_graph_kind,
)
from generativeqc_compiler.xc.grid_phased import (
    PhasedBeckePlan,
    emit_phased_becke,
    plan_phased_becke,
)
from generativeqc_compiler.xc.grid_response_ir import (
    GridResponseProgram,
    grid_response_graph_identity,
    grid_response_program,
)


def _actual_identity(program: GridResponseProgram, kind: str) -> str:
    """Authenticate reachable AD roots instead of trusting cached metadata."""
    return grid_response_graph_identity(program.graph, program.roots, kind)


@dataclass(frozen=True)
class BeckePartitionDerivativeOp:
    """A recognized canonical Becke normalized-product derivative contract.

    Whole-graph operations bind an authenticated primal/JVP composition and its
    atom domain. Legacy scalar-only operations preserve the prototype identity
    but do not establish production whole-composition recognition.

    Two exact zero factors annihilate every first derivative of an atom product.
    One zero does not: rounded switch zeros can retain a nonzero slope. The
    experimental index domain therefore retains every atom with zero count <= 1,
    including saturated and underflowed products. This is not a floating-point
    magnitude cutoff, an early log-product stop or a full xyz derivative cache.
    """

    iterations: int
    program_identities: tuple[tuple[str, str], ...]
    composition_identity: str | None = None
    atoms: int | None = None

    @property
    def identity(self) -> str:
        """Bind scientific graphs, zero semantics and ordered reduction policy."""
        if self.composition_identity is not None:
            return canonical_hash(
                {
                    "schema": "generativeqc.becke-partition-derivative-op/v2",
                    "programs": self.program_identities,
                    "composition": self.composition_identity,
                    "atoms": self.atoms,
                    "partition": "becke-equal-radius",
                    "zero_semantics": "single-zero-retains-pullback-multi-zero-annihilates",
                    "normalization": "selected-product-over-all-products",
                    "geometry": "validated-point-and-center-distance-bindings",
                    "reduction": "ascending-neighbors-no-fp-atomics",
                }
            )
        return canonical_hash(
            {
                "schema": "generativeqc.becke-partition-derivative-op/v1",
                "programs": self.program_identities,
                "zero_semantics": "retain-zero-count-0-and-1",
                "normalization": "ordered-live-products-plus-zero-owner-broadcast-denominator-bar",
                "pair_owner": "higher-live-atom-or-sole-live-atom",
                "gather": "ascending-incident-atoms-no-fp-atomics",
                "geometry": "rebind-canonical-center-pairs-no-point-state-reuse",
            }
        )


def recognize_becke_partition_derivative(
    *,
    ratio: GridResponseProgram,
    logarithm: GridResponseProgram,
    switch: GridResponseProgram,
    iterations: int = 3,
    partition: str = "becke-equal-radius",
) -> BeckePartitionDerivativeOp | None:
    """Fail closed for a changed canonical graph or unsupported partition.

    Radial element radii remain owned by grid preparation; this recognizer does
    not invent a heteronuclear partition adjustment for the current equal-radius
    contract. A future adjusted switch must have its own independently qualified
    pattern. No functional, method or molecule name participates in matching.
    """
    checked_int(iterations, "partition iterations", high=5)
    if partition != "becke-equal-radius":
        return None
    identities = []
    for kind, program in (("ratio", ratio), ("log", logarithm), ("becke", switch)):
        expected = grid_response_program(kind, iterations).identity
        if _actual_identity(program, kind) != expected:
            return None
        identities.append((kind, expected))
    return BeckePartitionDerivativeOp(iterations, tuple(identities))


def recognize_becke_partition_graph(
    program: GridResponseProgram,
    *,
    atoms: int,
    iterations: int = 3,
    partition: str = "becke-equal-radius",
) -> BeckePartitionDerivativeOp | None:
    """Match the complete reachable normalized-product primal AND generated JVP.

    Shape, complementary pair orientation, all denominator products, integer
    owner selection, saturation and rounded-zero branch semantics participate
    in matching. Supplied identities and unreachable companion roots do not.
    Unsupported dimensions/partitions or cross-graph roots fail closed.
    """
    if (
        type(atoms) is not int
        or not 1 <= atoms <= 128
        or partition != "becke-equal-radius"
    ):
        return None
    expected = grid_partition_program(atoms, iterations)
    try:
        actual = _actual_identity(program, partition_graph_kind(atoms))
    except ValueError:
        return None
    if actual != expected.identity:
        return None
    scalar = recognize_becke_partition_derivative(
        ratio=grid_response_program("ratio", iterations),
        logarithm=grid_response_program("log", iterations),
        switch=grid_response_program("becke", iterations),
        iterations=iterations,
        partition=partition,
    )
    if scalar is None:
        return None
    return BeckePartitionDerivativeOp(
        iterations, scalar.program_identities, actual, atoms
    )


def validate_becke_partition_derivative(operation: BeckePartitionDerivativeOp) -> None:
    """Reauthenticate canonical roots before emitting a matched operation."""
    if operation.composition_identity is not None:
        if operation.atoms is None:
            raise ValueError("Becke primitive composition lacks its atom domain")
        current = recognize_becke_partition_graph(
            grid_partition_program(operation.atoms, operation.iterations),
            atoms=operation.atoms,
            iterations=operation.iterations,
        )
    else:
        current = recognize_becke_partition_derivative(
            ratio=grid_response_program("ratio", operation.iterations),
            logarithm=grid_response_program("log", operation.iterations),
            switch=grid_response_program("becke", operation.iterations),
            iterations=operation.iterations,
        )
    if current != operation:
        raise ValueError("Becke primitive no longer matches its canonical AD graphs")


@dataclass(frozen=True)
class BeckePartitionDerivativePlan:
    """Bounded dense primal panels plus point-major exact reverse indices.

    Index membership is recomputed after every point normalization, never cached
    across geometry or seed binds. All pair primal/log work remains unchanged.
    Runtime live-atom counts, not this capacity bound, determine reverse visits.
    """

    operation: BeckePartitionDerivativeOp
    phased: PhasedBeckePlan

    @property
    def index_bytes(self) -> int:
        return (self.phased.atoms + 2) * self.phased.points * 8

    @property
    def scratch_bytes(self) -> int:
        return self.phased.scratch_bytes + self.index_bytes


def plan_becke_partition_derivative(
    *,
    operation: BeckePartitionDerivativeOp,
    atoms: int,
    points: int,
    budget_bytes: int,
    occupied_bytes: int = 0,
) -> BeckePartitionDerivativePlan | None:
    """Charge both index/count panels against the same concurrent owner budget."""
    phased = plan_phased_becke(
        atoms=atoms,
        points=points,
        budget_bytes=budget_bytes,
        occupied_bytes=occupied_bytes,
    )
    if phased is None or operation.atoms is not None and operation.atoms != atoms:
        return None
    plan = BeckePartitionDerivativePlan(operation, phased)
    return plan if occupied_bytes + plan.scratch_bytes <= budget_bytes else None


_SOURCE = r"""
#if defined(__CUDACC__)
#define GENERATIVEQC_PARTITION_HD __host__ __device__
#else
#define GENERATIVEQC_PARTITION_HD
#endif
namespace generativeqc_grid_partition {
using namespace generativeqc_grid_phased;

struct DerivativeWorkspace {
  Workspace work;
  size_t* indices;
  size_t* counts;
  double* common_bars;
  struct LiveNeighbors {
    const size_t* indices;
    size_t stride;
    GENERATIVEQC_PARTITION_HD size_t operator()(size_t index) const {
      return indices[index * stride];
    }
  };
  GENERATIVEQC_PARTITION_HD LiveNeighbors neighbors(size_t point) const {
    return {indices + point, work.points};
  }
  template <class Value> struct Indexed {
    Strided<Value> values;
    LiveNeighbors neighbors;
    GENERATIVEQC_PARTITION_HD Value& operator[](size_t index) const {
      return values[neighbors(index)];
    }
  };
  struct Bars {
    Strided<double> values;
    Strided<size_t> zeros;
    double common;
    GENERATIVEQC_PARTITION_HD double operator[](size_t atom) const {
      return zeros[atom] <= 1 ? values[atom] : common;
    }
  };
};

template <class Ratio>
GENERATIVEQC_PARTITION_HD bool partition_normalize_phase(DerivativeWorkspace input,
    size_t point, size_t owner, double seed, Ratio ratio) {
  input.counts[point] = 0;
  if (owner >= input.work.atoms || !std::isfinite(seed)) return false;
  size_t owner_slot = input.work.atoms;
  for (size_t atom = 0; atom < input.work.atoms; ++atom)
    if (input.work.zero_counts(point)[atom] <= 1) {
      if (atom == owner) owner_slot = input.counts[point];
      input.indices[input.counts[point] * input.work.points + point] = atom;
      ++input.counts[point];
    }
  size_t normalized_count = input.counts[point];
  if (owner_slot == input.work.atoms) {
    // The selected numerator is exactly zero, but its ratio AD still belongs
    // to the canonical objective. Include the zero owner only in normalization,
    // not in the reverse dependency list. Capacity remains at most natom.
    owner_slot = normalized_count;
    input.indices[normalized_count * input.work.points + point] = owner;
    ++normalized_count;
  }
  const auto neighbors = input.neighbors(point);
  const DerivativeWorkspace::Indexed<double> logs{input.work.field(4, point), neighbors};
  const DerivativeWorkspace::Indexed<size_t> zeros{input.work.zero_counts(point), neighbors};
  const double maximum = maximum_log_product(normalized_count, logs, zeros);
  if (!std::isfinite(maximum)) return false;
  input.work.maximum[point] = maximum;
  input.common_bars[point] = normalized_product_adjoint(normalized_count, owner_slot,
      seed, logs, DerivativeWorkspace::Indexed<double>{input.work.field(5, point), neighbors},
      DerivativeWorkspace::Indexed<double>{input.work.field(6, point), neighbors},
      zeros, maximum, ratio);
  return true;
}

template <class Geometry, class Log>
GENERATIVEQC_PARTITION_HD bool partition_reverse_atom_phase(DerivativeWorkspace input,
    size_t point, size_t neighbor, Geometry geometry, Log logarithm, size_t& visits) {
  visits = 0;
  const auto live = input.neighbors(point);
  const DerivativeWorkspace::Bars bars{input.work.field(6, point),
      input.work.zero_counts(point), input.common_bars[point]};
  for (size_t cursor = 0; cursor < input.counts[point]; ++cursor) {
    const size_t atom = live(cursor);
    if (atom == neighbor) continue;
    // A pair with two live endpoints belongs only to its higher endpoint.
    // A dead endpoint still receives geometry response from a live neighbor.
    if (neighbor > atom && input.work.zero_counts(point)[neighbor] <= 1) continue;
    ++visits;
    if (!pair_reverse_with_bars_phase(input.work, point, std::max(atom, neighbor),
          std::min(atom, neighbor), geometry, logarithm, bars)) return false;
  }
  return true;
}

GENERATIVEQC_PARTITION_HD void partition_gather_phase(DerivativeWorkspace input,
    size_t point, size_t atom) {
  if (input.work.zero_counts(point)[atom] <= 1)
    atom_gather_phase(input.work, point, atom);
  else
    atom_gather_selected_phase(input.work, point, atom, input.counts[point],
                              input.neighbors(point));
}
} // namespace generativeqc_grid_partition
#undef GENERATIVEQC_PARTITION_HD
"""


def emit_becke_partition_derivative(operation: BeckePartitionDerivativeOp) -> str:
    """Emit a dependency lowering that calls the authoritative AD phase bodies."""
    validate_becke_partition_derivative(operation)
    return (
        emit_phased_becke()
        + f"// Canonical Becke derivative operation: {operation.identity}\n"
        + _SOURCE
    )

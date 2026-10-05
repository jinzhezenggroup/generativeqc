# Decision: require proved accesses for streamed reduction groups

Status: implemented
Date: 2026-10-04
Agent: dot

## Problem

The initial #1866 grouping used only flattened reduction extent, output shape,
and precision. Equal-size row/column reductions therefore appeared compatible
despite reading different producer elements for the same output/reduction
coordinates. A consumer could also depend on another proposed sibling, including
through a materialized frontier.

## Decision

Use equality of canonical TensorIR reduction attributes and operand shapes as a
conservative root-access proof. Propagate it through the existing elementwise
primitive set, casts, and reshapes, which preserve the logical flat index. Other
virtual primitives are opaque boundaries: their values may be shared, but their
ancestors require a separate proof. Multiple different or unproved access paths
to one producer exclude that producer for the consumer. Reject a whole consumer
set if any member transitively depends on another member.

This reuses the IR's canonical labels and primitive semantics without importing
the emitter into planning or adding a second indexing algebra. Exact structural
equality deliberately misses some equivalent mappings. Maximizing groups is not
required for these diagnostics. Overlapping groups remain descriptive candidates,
not a partition or a launch plan.

## Invariants and evidence

- This only corrects dormant schedule diagnostics; CUDA execution, precision,
  source generation, default schedules, and provider selection are unchanged
- Tests cover row/column coordinates, transpose aliases, einsum labels, ambiguous
  repeated operands, direct/transitive dependencies, and positive pointwise,
  cast, reshape, and opaque-producer cases
- The seven negative cases fail against the original #1866 implementation
- Host planner/search/cost/structure tests and the compiler dependency audit pass
- Reproducible host comparisons preserve source, plan, layout/execution identities
  and all pre-existing static estimates; no GPU execution or speedup is claimed

## Revisit when

A shared indexing representation can prove additional composed mappings, or an
explicit phased execution/storage contract supports dependent reductions. Any
future lowering must also select among overlapping candidates and qualify the
complete numerical and runtime behavior independently.

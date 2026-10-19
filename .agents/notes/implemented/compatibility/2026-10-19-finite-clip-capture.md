# Decision: bounded finite-real clip capture

Status: implemented
Date: 2026-10-19

## Problem

The public eager 2025.12 namespace already owns standard clip. Its TensorIR
adapter had no capture lowering. Adding another eager implementation would
duplicate that contract and risk restricting IEEE behavior.

## Decision

Lower generic finite float32/float64 clip to existing typed comparisons, where
and explicit broadcasting. Absent bounds are omitted, static scalars reuse exact
literal conversion, and array bounds must match the input dtype. Reject scientific
annotations before even a both-None identity. Apply the lower bound before the
upper bound; reversed bounds select the upper bound, an explicitly chosen policy
in a range the standard leaves unspecified. Keep the current branch on equality,
preserving its signed zero. Exact static literals cannot encode negative zero.

## Rejected alternatives

- A new clip primitive or AD engine duplicates existing typed selection algebra.
- Promoting mixed array bounds silently changes output dtype in an unspecified
  standard domain. Reject them explicitly instead.
- Rejecting NaN in the public eager path would regress its existing contract.
- Selecting the bound on equality changes signed-zero behavior relative to the
  NumPy 2.5 reference used during development. Strict comparisons retain the
  current operand on equality. Neither tie branch defines a derivative there.

## Invariants

All live operands retain finite/domain checking and resource budgets, including
masked bounds. No implicit transfer, scientific-domain relabeling, new scientific
consumer policy, or default AD admission. First-order reference AD requires the
existing comparison-selector opt-in. Native capability qualification remains
unset even though the graph can use previously admitted FP64 CPU primitives.

## Evidence

`tests/python/test_array_api_clip.py` compares reference capture and eager values
against NumPy and installed array-api-strict 2.6.1, plus exact Fraction controls.
It exercises identity, one-sided/equal/reversed bounds, empty/scalar/multi-axis
shapes, dtype/annotation/domain rejection, deterministic serialization, interior
JVP/VJP/dot/finite differences and the explicit tie convention. A native FP64
forward/JVP/VJP regression forbids reference interpretation and must run in a
supported cached C++ environment; Windows reference tests alone do not qualify it.

## Consequences

This is one finite generic capture slice of #2111, with no eager API changes or
full-standard/backend/performance completion claim.

## Revisit when

Independent qualification supports extending dtype, IEEE, scientific annotations
or backend admission. Such changes require their own explicit acceptance gates.

## References

- Refs #2111; parent scope remains broader.
- [Current contract](../../../../docs/user/experimental_array_api.md).
- [Pinned standard](https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.clip.html).

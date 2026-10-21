# Decision: bounded finite max/min capture

Status: implemented
Date: 2026-10-21

## Decision

Issue #2111's public eager max/min already exist. Lower the missing generic
finite float32/float64 capture with existing gather, comparison, where and
reshape, without adding an IR operation, AD engine or backend implementation.
Reduce axes from last to first; within an axis compare adjacent pairs and
halve its extent each round. Duplicate the last element for an odd extent.
Strict comparison keeps the left operand on equality, so this ordering retains
the earliest C-order extremum and its signed zero across multiple axes.

Validate all metadata, axes and keepdims before identity/empty shortcuts.
Reject actual empty reduced axes, including when another axis gives an empty
output. Empty retained dimensions return a detached empty reshape. Empty-axis
and singleton reductions are identities. Nonfinite inputs remain rejected by
the existing compiled input/interpreter contract; eager IEEE behavior stays intact.

## Bounds and rejected alternatives

Preflight total gather positions <= 65,536, pairwise rounds <= 64, and newly
retained intermediate tensor bytes <= 256 MiB. A round accounts conservatively
for four real arrays plus one Boolean predicate of the halved shape; it does
not replace backend budgets or count existing input graph storage. Scalar-by-
scalar Python unrolling would create a node per input value and deep chains.
Pairing contiguous halves would change the earliest-index tie convention.
Arithmetic extrema formulas can overflow for finite inputs and lose zero signs.
Do not promote native capability merely because existing primitives can execute.

## AD and evidence

Use the existing allow_comparison_where=True selected-branch convention.
At unique extrema, independent scalar-index JVP/VJP/dot and finite differences
are acceptance gates. Ties have no claimed mathematical derivative. Identity
graphs keep the existing identity AD boundary. The independent tests cover
NumPy and array-api-strict 2.6.1, unequal axes, odd extents, noncontiguous input,
serialization/optimization/replay, empty/scalar/rank64 behavior, error domains,
budgets and native FP64 forward/AD with interpreter fallback forbidden.
Test definitions alone are not execution receipts. Native float32, CUDA,
scientific methods, performance and whole-issue completion remain excluded.

## Revisit when

A reviewed native reduction primitive or larger-scale resource contract exists.
The internal compiler still advertises array_api_version=None; public eager
2025.12 discovery is a separate existing contract.

# Decision: finite generic TensorIR selection remains a forward reference primitive

Status: implemented
Date: 2026-10-13

## Problem

The standard eager facade already exposes `where`, but captured TensorIR needs
typed selection without encoding predicates as floating masks or weakening
scientific index domains. Selected-real-branch differentiation and native
qualification are separate work in #2137.

## Decision

Use the existing immutable Node/Program, Boolean data, explicit real casts,
broadcasts, serialization, CSE and NumPy interpreter. `where` has three ordered
live operands and fresh real storage. Frontend promotion applies only to the
two branches; the condition stays Boolean. Admission is generic, general,
unannotated arrays only, even when scientific extents happen to agree.

Both branches remain live under constant conditions or identical branch values.
The optimizer does not simplify selection to one branch: doing so would discard
the interpreter's finite/domain errors in the unselected branch. No derivative
rule is registered; all native preparation rejects live selection before compiler
discovery, allocation or device execution. Existing native CPU Boolean comparisons
and typed views retain their previous qualification.

## Rejected alternatives

Floating 0/1 masks lose Boolean type identity and can execute different arithmetic.
Short-circuit selection would silently change retained live work and error
semantics. Domain matching by extent would mix AO/occupied/virtual/spin values.
A separate selection IR would duplicate identity and resource ownership.

## Invariants and evidence

`tests/python/test_array_api_where_reference.py` uses direct NumPy and
array-api-strict oracles across float32/float64, promotion, scalar/empty/broadcast
and noncontiguous inputs. It checks signed zeros, extremes, detached replay,
failure behavior, serialization/CSE, retention bounds, AD projection and native
rejection. Selection work is one logical selection per output element; logical
storage and returned-copy budgets use the existing interpreter accounting.
No performance, complete Array API conformance, IEEE nonfinite, native device or
scientific endpoint acceptance claim follows from these reference tests.

## Revisit when

Selected-real JVP/VJP/HVP semantics, broadcast adjoints and scientific domain
admission have independent acceptance gates, or a native backend has separately
qualified typed predicate storage and complete execution semantics.

## References

- #2137 and parent #2111 remain open; this slice uses Refs.
- Current contract: `docs/user/experimental_array_api.md`.

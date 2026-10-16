# Decision: capture uniform repeat using existing shape primitives

Status: implemented
Date: 2026-10-16

## Decision

Admit only a static nonnegative Python integer repeat count for generic Boolean
and real arrays. Insert a singleton dimension after the selected axis, broadcast
it to the count, and merge it with that axis using row-major reshape. Axis None
first flattens the input. Identity counts need no broadcast; zero counts retain
ordinary empty broadcast/reduction behavior. Scientific spaces require explicit
index semantics and remain rejected, including zero and identity counts.

## Alternatives and boundaries

An explicit gather map grows with the repeated axis and would need an arbitrary
static-map admission cap. Uniform repetition already has an exact representation
in the shared reshape/broadcast algebra, with metadata proportional to rank.
Per-element count arrays cannot use that representation and remain unsupported.
No new IR serialization, primitive or differentiation algebra is introduced.

This is not a performance claim: broadcast and reshape execution can still move
or materialize output-sized data. Existing backend byte, work and node budgets
remain mandatory; the feature does not imply work-bounded execution at arbitrary
output sizes or CUDA qualification. Native dtype admission is unchanged.

## Evidence and revisit conditions

`tests/python/test_array_api_repeat.py` checks NumPy values, signed zeros, empty
and scalar shapes, scientific rejection, overflow and native budgets. JVP/VJP
use finite differences and an adjoint dot gate, including zero repeats, with
native reference execution/AD forbidden. The capability matrix exercises source
emission through the existing native primitives. Revisit nonuniform counts only
with explicit shape/resource semantics and independent derivative qualification.

Refs #2111.

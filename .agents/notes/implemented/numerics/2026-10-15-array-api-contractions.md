# Decision: generic real Array API contractions reuse TensorIR

Status: implemented
Date: 2026-10-15

## Problem

The public eager namespace exposes `tensordot` and `vecdot`, but captured
programs need static contraction semantics, dtype promotion and derivative gates.
Implicit shape relabeling must not erase scientific AO/occ/vir domains.

## Decision

Lower generic float32/float64 contractions to the existing einsum, transpose,
broadcast and cast nodes. Keep left free axes followed by right free axes for
tensordot. Normalize signed axes and require equal contracted extents before
any vecdot batch broadcasting. Reject Boolean and annotated scientific inputs;
scientific consumers can supply explicit typed einsum expressions instead.
Retain the existing 52 single-character einsum label bound. Native CPU remains
subject to the existing float64, byte/work/node admission checks; CUDA is not
newly qualified. No new IR operation, derivative rule or device fallback is added.

## Rejected alternatives

Elementwise multiply followed by a new reduction implementation would duplicate
existing contraction algebra. Broadcasting the selected vecdot dimension would
accept an invalid length-one vector against a longer vector. Relabeling typed
scientific inputs by shape would discard index-domain identity. Complex support
would require conjugation and different derivative semantics, so is excluded.

## Invariants

Preserve output axis order, exact contracted lengths, independent output storage,
static axis validation and explicit float32/float64 promotion. Keep scientific
index domains strict. Capability entries describe the bounded subset rather than
claiming full Array API conformance.

## Evidence

`tests/python/test_array_api_contractions.py` compares eager and captured values
against NumPy for signed/reordered axes, outer products, scalar and empty results,
broadcast batches and mixed precision. It checks JVP finite differences, coordinate
finite-difference VJPs and the adjoint dot identity, plus native CPU derivatives
with reference evaluators forbidden. The coverage matrix exercises CPU source
emission for float64 and rejection of float32 native programs.

## Revisit when

An owned complex/conjugation algebra or scientific axis inference contract exists,
or TensorIR gains contraction notation beyond the current label inventory.

## References

Refs #2111. Current contract: `docs/user/experimental_array_api.md` and
`docs/developer/array_api_frontend.md`.

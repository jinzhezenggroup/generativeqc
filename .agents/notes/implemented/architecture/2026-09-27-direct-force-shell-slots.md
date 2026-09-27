# Decision: one generated shell-slot owner for low-order Direct force

Status: implemented in source; CUDA/endpoint qualification pending
Date: 2026-09-27

## Problem

Order-two force carried three class-specific shell permutations, while order-three
force carried a separate pair-class sorter. These permutations connect Cartesian
component weights, primitive-pair orientation and recovered derivatives. Treating
them as incidental indexing risks changing scientific center ownership during
future retirement work for #356.

## Decision

The existing low-order Weighted IntegralIR header now emits one batch-independent
`canonicalize_direct_shell_slots` helper. Order-two generic/exact consumers and
the order-three exact consumer call that helper. The helper applies only the
existing eight ERI pair symmetries: order shells within each pair, then order the
two pairs by angular class. It returns canonical-to-original slot indices.

This is a narrow shared emission/ownership slice, not a new recurrence, scheduler,
public domain or claim that all Direct force adapters are runtime-only. The
remaining AO component indexing and primitive contraction keep their conservative
scientific ownership. No aggregate retirement family is reclassified.

## Invariants

- Equal angular momenta preserve shell order; equal pair classes preserve pair
  order. Do not add an AO/shell/atom-ID tie breaker or change `<` to `<=`.
- The order-two supported-class guards remain fail-closed. The generic helper
  does not grant production support to additional angular classes.
- Native queue/screening decisions, pair-cache orientation, component packing,
  generated geometry/force-root bodies and recovered-center/scatter arithmetic
  remain unchanged. PSPS retains its original primitive-pair ownership.
- Only the low-order production header gains the helper. The standalone psss
  weighted-header interface and existing force/geometry emitter functions remain
  unchanged; no new module or CMake generator dependency is needed.

## Rejected alternatives

Do not copy another PSPS/PPSS/DSSS-specific map into the compiler. One generic
pair-symmetry helper covers the already-admitted domain. Sorting by shell ID is
also incorrect here: it changes the historical tie semantics even if the ERI
value itself is symmetric.

## Evidence and limits

The host-compiled emitted helper agrees with an independent enumeration of the
eight ERI permutations for 1,192 input cases, including all s/p/d/f angular
quadruples, noncontiguous/aliased shell IDs, and two integer-type combinations.
Additional checks preserve the historical order-two specialized maps, component
flattening, recovered-center mapping and repeated-atom coalescing.

Local validation used the pinned three-file source subset and an AST-extracted
pure emitter: five isolated tests passed; the full-header integration test was
not run locally. This is not full-package import, CUDA compilation or GPU
endpoint evidence. The committed tests use normal compiler-package imports in
repository CI. No performance, resource or complete-force qualification is
claimed. Exact-head CUDA/RHF/UHF gates remain authoritative before promotion.

Physical native source reduction: 107 lines across the two force headers. This
is not a regenerated repository-wide scientific-CUDA ownership count.

## Revisit when

A new admitted operator breaks real-ERI pair symmetry, or runtime qualification
shows the shared permutation changes resource/endpoint behavior. Do not broaden
production coverage based only on the generic helper's representational range.

## References

- #356; dependency stack #1432 -> #1438.
- Source baseline: `2e76b45623a3440c3f71976d91b59073c0d1c9f5`.
- `tests/python/test_direct_shell_canonicalization.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

# Decision: share correlated HF warm-reference lifecycle

Status: implemented
Date: 2026-10-02

## Problem

MP2 already retained validated HF warm densities for prepared replay, while RCCSD and RCCSD(T)
exposed incomplete warm-state contracts and all three correlated methods carried overlapping
geometry, checkpoint, capture, and reservation logic. The CC paths also need the retained seed and
candidate snapshot charged beside every correlation phase rather than treated as free external memory.

## Decision

Use one method-layer correlated warm-reference helper for geometry snapshots, checkpoint validation,
numeric reservation, and successful-reference capture. MP2, RCCSD, and RCCSD(T) use the same
state-lifecycle primitives. RCCSD and RCCSD(T) both seed the same strict RHF
reference owner, retry the unchanged cold reference after a failed warm proposal, and publish a new
warm snapshot only after the complete requested correlated endpoint succeeds.

The reservation includes the last-good input seed and, while updates are enabled, the candidate
snapshot. Correlation, triples, and force phases run inside the remaining method budget. The
reported endpoint capacity adds the external warm reservation back exactly once.

## Rejected alternatives

Transporting CC amplitudes together with the HF density was rejected because orbital-frame/T1/T2
transport belongs to #190 and needs separate validation. Treating the warm snapshots as uncharged
batch metadata was rejected because it would violate the method numeric-memory contract.

## Invariants

- A warm density is only an SCF initial proposal and never proves target convergence.
- Warm-reference failure retains a cold retry.
- A failed CC, triples, or force endpoint does not replace the last-good warm snapshot.
- Geometry/basis density validation happens before restored state is committed.
- No T1/T2, DIIS, triples, or response state is transported by this lifecycle.
- Preserve #1701's explicit MP2 density branches when calling the shared capture
  helper: move a populated iterative vector, otherwise copy the immutable physical
  reference density once. Combining those operands in a conditional expression
  would reintroduce an extra density allocation on the CPU path.
- CUDA warm/cold references use the native CUDA RHF owner. Optional correlation
  source preparation follows successful reference validation and remains inside
  the phase budget after both retained warm payloads have been reserved.

## Evidence

Public MP2, RCCSD, and RCCSD(T) batch tests cover cold-to-warm replay, geometry movement, clearing,
checkpoint restore, budget admission, and item-local failure behavior. Existing strict RHF and
correlated numerical gates remain unchanged.

Canonical RCCSD and RCCSD(T) now have explicit resolved model identities so the
existing portable checkpoint save/load path can validate them. The schema admits
only the current all-electron, unfrozen, conventional, closed-shell CC contract;
method, basis, ordered nuclei, geometry and spin identities remain strict. This
identity adapter does not admit correlated target-accuracy or progressive
projection execution. Cross-method imports remain rejected even with warm
admission, and changed geometry still requires explicit warm admission followed
by a fresh native reference solve.

## Consequences

MP2, RCCSD, and RCCSD(T) now share the same HF warm-reference state primitives. MP2 retains its
already-qualified endpoint semantics, while the CC paths add the same validated batch lifecycle and
explicit reservation accounting.

## Revisit when

A versioned prepared physical-reference identity can safely skip more RHF setup, or #190 provides
validated orbital/amplitude transport.

## References

- #1503
- #1696
- #1701
- #190

Original author attribution: ChatGPT
Integration review: dot

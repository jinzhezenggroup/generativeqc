# Decision: retain three-index unrestricted DF factors and reverse them directly

Status: implemented
Date: 2026-11-04

## Problem

The initial bounded batching change for #2978–#2980 is preserved in
[the earlier note](2026-11-04-unrestricted-admission-and-force-batching.md).
It fixes singleton UMP2 force requests, but reconstructing the fitted interaction
as an AO fourth-order tensor still defeats DF storage scaling. Repeating a
whole-AO transform for every orbital-gradient slice is unnecessary even when
each individual request fits the memory budget.

## Decision

The immutable fitted source owns raw three-center integrals, the native FP64
metric inverse square root, AO three-index factors, and alpha/beta MO
three-index factors. The compiler owns the staged two-orbital transformation,
rectangular pair-panel Gram contraction and complete reverse program. The
reverse includes both coefficient legs, raw three-center and metric-root
cotangents. The existing fixed-rank spectral derivative and nuclear integral
owners complete the physical response; no independent-reference oracle runs
inside a production endpoint.

The unused dense-AO native evaluation/pullback interface and its generated
buffers are retired rather than kept as a compatibility shim. The small
interpreted AO Gram algebra remains an independent compiler test alongside the
new native representation; production generation emits only three-index maps
and rectangular blocks.

The native provider verifies the accepted physical orbital frame before
replaying factors. Ordered same-spin pair requests reconstruct only their
rectangular MO block, not an AO interaction tensor. UMP2 constructs initial
weights by occupied pairs and virtual-pair blocks, including exchange by
permutation. Structural force weights accumulate both Gram-leg cotangents
directly into spin factors. Their complete generated VJP supplies the orbital
gradient and the final raw/root nuclear cotangents. Full DF UMP2 therefore has
neither an AO fourth-order value nor an AO fourth-order force cotangent.

The CUDA owner borrows immutable host factors and optionally owns device factor
copies. A refused optional allocation falls back to admitted selected-panel
uploads and the same generated CUDA Gram contraction, never host contraction.
Every published result follows stream completion; shared execution is serialized
and device allocations carry stream-aware cleanup. Capacity admission includes
retained factors, simultaneous panels, output copies, selection indices,
generated forward/reverse arenas and backend nuclear-response staging.

UCC fitted nuclear response also reverses factor cotangents without an AO
fourth-order transform. UCC's generated MO tensors, Lambda equations and ordered
triples evaluation are distinct costs, not removed by this source change. The
finite shared work policy remains a necessary resource boundary. Progress
scopes separate graph programs, Lambda, triples, orbital and nuclear response
so complete-endpoint costs can be measured rather than inferred from RCC or
integral microbenchmarks.

## Rejected alternatives

- Raising a scalar-work limit cannot repair repeated transforms or storage.
- A full spin-channel MO value is a useful bounded conventional fallback, not
  the required resident representation of a fitted interaction.
- A permanently resident device cache would turn an optional performance
  optimization into a correctness/resource precondition.
- Ignoring either coefficient leg or the metric/nuclear derivative produces
  an incomplete force even if unrelaxed energy integrals agree.

## Invariants and limits

Retain exact frame/backend ownership, checked arithmetic, finite outputs,
complete output publication, independent scientific gates and the conventional
bounded slice fallback. Correlation-only DF still owns a conventional reference
response explicitly. Native host FP64 metric conditioning and detached
factor/cotangent transport are intentional hybrid work, not a silent backend
substitution. No broad production-size scalability or RCC/UCC speed ratio is
claimed. The qualified finite physical ladder must not be extrapolated to the
issue's hundreds-of-AO workload.

## Evidence

The compiler test compares forward factors and rectangular Gram blocks against
independent NumPy algebra, and checks directional VJPs separately for raw
integrals, the root and both physical spin frames. Native contracts compare
requested fitted blocks to an independent dense contraction, reject stale
frames, and exercise resource admission. The opt-in CUDA contract compares
resident and zero-residency panel execution across spin channels, padded slots
and zero-beta factors. Physical endpoint qualification uses independent PySCF
energies and two-step reconverged force differences, never native finite
differences as their own oracle.

Source-matched final endpoint receipts and limitations are recorded below after
the complete-library qualification.

## Revisit when

An independently qualified tiled UCC triples/Lambda schedule replaces ordered
evaluation, or a backend supplies bounded device-resident DF nuclear derivative
contractions that remove detached coordinate staging. Preserve complete endpoint
qualification and both metric/orbital response when changing these boundaries.

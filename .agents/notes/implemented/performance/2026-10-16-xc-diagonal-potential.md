# Decision: contract diagonal XC bilinears only once

Status: implemented; complete-endpoint qualification in progress
Date: 2026-10-16

## Problem

After the sparse MD-J storage change in #2563, the source-matched 96-atom
r²SCAN-3c cold profile spends about 100 device seconds in J and 108 in XC at
24 actual Fock builds. XC matrix contractions dominate its point expressions
and AO evaluation. Changing launch widths does not remove their arithmetic.

## Decision

The existing compiler-owned AO-pair bilinear remains the mathematical owner.
Its compact value panel contains the scalar term and both gradient legs.
Additional panels are diagonal kinetic-density bilinears. Derive their full
coefficients from that same graph and contract each additional panel once,
rather than materializing half coefficients and evaluating both symmetric
cross products. A productive four-jet AO-pair/point contraction consequently
uses five multiplications instead of eight. This is not a smaller candidate
enumeration or a change of screening.

Only strict-double generated four-work-jet consumers select this layout. Both
dense/mapped tiled execution and compact batches use it. The authoritative
triangle, mirrored publication, accumulation order between tiles, point-total
reduction and error propagation remain owned by their existing implementations.

The generic two-legged consumer remains available for arbitrary work panels,
FP32 contractions, LDA/GGA and library providers. Tiny/out-of-domain shapes keep
the existing scalar fallback. No allocation, grid, screening, provider budget,
CTA/page width, named-functional selector or integral approximation is added.

## Invariants

- The value/gradient panel must retain both legs; only graph-derived additional
  diagonal panels may use one-sided contraction.
- Full-coefficient panels and the one-sided consumer are a paired layout.
  Never feed them to the generic symmetric provider or reuse density work
  before its last consumer.
- Preserve strict FP64, independent spin coefficients, signed quadrature
  weights, mapped/tail domains, existing nonfinite rejection and exact J/K.
- Mathematical equivalence does not mean bitwise equality: the location of the
  point coefficient and the half-coefficient products change FP64 rounding.
  Complete independent energy/force gates remain mandatory.

## Rejected alternative

A generated four-right-hand-side density contraction shared each symmetric
density tile across all AO jets, preserving each reduction's order and reducing
density preparation and CTA barriers by four. Independent tests passed, but
scheduled RTX 5090 ABBA microbenchmarks regressed most representative domains:
64 active AOs about 0.0184 → 0.0308 ms; 384 about 0.442 → 0.498 ms; 544 about
0.812 → 0.932 ms; 1088 about 3.224 → 3.249 ms. It is not retained. The complete
patch, samples and receipts remain in the qualification evidence/ignored raw
artifacts. Reduced preparation alone is not an endpoint speedup.

## Qualification boundaries

Independent signed-coefficient bilinear tests and actual CUDA panel/consumer
tests cover both spin counts, mapped/global outputs, partial tiles, accumulation
and nonfinite rejection. Scheduled memcheck, synccheck and racecheck pass.
Representative contraction microbenchmarks improve about 33–37%; they are not
complete E+force claims.

Nsight Compute's dynamic instruction counters are unavailable to this account
(`ERR_NVGPUCTRPERM`); do not describe symbolic multiplication counts or AO-map
logical work as measured hardware counters. A separate node2 sanitizer attempt
stopped before tests because installed NVML 595.99.02 does not match its loaded
595.91.07 driver. Qualification uses scheduled node1 RTX 5090 execution instead;
this change does not modify system drivers or performance-counter permissions.

## Revisit when

The remaining matrix-density contractions dominate complete cold execution,
or a prepared density-factor representation can remove arithmetic without
changing input-density semantics, scientific gates or bounded resource ownership.
The independently qualified Direct-force stack remains a separate dependency;
do not combine historical timings arithmetically to claim a stacked endpoint.

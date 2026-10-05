# Decision: cuBLASLt participates in the shared source-response region

Status: implemented
Date: 2026-10-05

## Problem and decision

The cached cuBLASLt executor and contiguous mode-group proof can implement the
eight semantic contractions in streamed DF MO source response. Add it to the
same emitted region portfolio. Keep scientific traversal and the native
post-HF consumer unchanged; provider setup, lifetime, ranking, reservations and
fallback remain in `PreparedContractionRegion`.

## Invariants

- cuTENSOR and cuBLASLt have independent simultaneous resource profiles. Never
  apply one provider's bounds to the other because both accept affine views.
- All eight plans are admitted/prepared before the first source read. Repeated
  rows reuse cached algorithms; no provider discovery occurs in a callback.
- Partial optional preparation drains provisional plans before a generated
  same-precision retry. Execution failures never trigger arithmetic replay.
- No production resource profile is installed. Synthetic test reservations and
  ranking are absent from production, which retains its qualified incumbent.
- Semantic descriptors, row order, beta accumulation, source immutability,
  provisional publication and complete-budget admission retain their contracts.

## Validation and remaining work

The actual native source-response owner is compared with the independent full
NumPy expression for unit and rectangular dimensions. Both optional providers
exercise successful preparation, rejection after two plans, generated fallback
and sticky arithmetic failure. A build with neither optional provider preserves
the incumbent. Shared executor tests independently protect cached algorithms
and semantic work accounting.

This is real consumer execution, not evidence for promoting a method default.
Opaque host/cache and lazy/OOM resource bounds, molecular force endpoint
performance selection and CUTLASS/CuTe remain open under #1888/#1886.

Local consumer timing sanity checks used n1/RTX 5090 under a finite Slurm
allocation, with other device work present. At `(n,q)=(4,6)` and `(16,24)`, all
four providers agreed with the independent full expression and reported 3,456
and 884,736 contraction summands respectively. Timing included allocations,
uploads, preparation of eight plans, both source passes, downloads, publication
and teardown. Repeated calls warmed the process but recreated the region; they
are not replay-only timings. These shared-device observations are not promotion
evidence. The local `.artifacts/1888-region/consumer-timing.json` retains samples,
separate prepare timing, resource counts and source/library SHA-256 identities.

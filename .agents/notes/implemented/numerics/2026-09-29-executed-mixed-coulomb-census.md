# Decision: Count mixed Coulomb work at the executing provider

Status: implemented
Date: 2026-09-29

## Problem

The precision-work query introduced by #1472 can represent an operator census,
but aggregate SCF counters cannot say how many mixed Coulomb AO-ERI recurrences
actually executed. Screening and zero total-density rejection make a launch or
Fock-build count an invalid substitute for recurrence work.

## Decision

The Direct-J CUDA provider accepts an optional caller-owned device `uint64_t`
counter for mixed Coulomb execution. It clears the counter on the provider
stream before every enqueue and increments it only after the same screening and
nonzero total-density predicates that guard the mixed contracted-ERI call.
Each 32-thread output block reduces its local count before one atomic addition.

The counter is optional so existing numerical callers and arithmetic remain
unchanged. It is admitted only for mixed Direct-J execution; a counter supplied
for strict prepared Fock execution fails closed. Execution owners may consume
the value after stream completion as a `CoulombRecurrence` operator record, but
must record strict logical work separately rather than inventing a strict
recurrence count.

## Rejected alternatives

- Deriving recurrence work from matrix dimensions, aggregate Fock builds, or
  `precision=auto` would count screened or zero-density work that did not run.
- Counting launches would lose the operator arithmetic multiplicity required by
  the precision-work census.
- A host-side counter or synchronization inside the provider would break the
  resident execution boundary and change scheduling costs.
- Instrumenting generated strict Coulomb kernels in this slice would widen the
  ownership boundary without being needed to prove mixed recurrence work.

## Invariants

- The counter is ordered on the provider stream and starts from zero on every
  enqueue, including replay after a previous nonzero result.
- Only evaluated mixed Coulomb recurrences contribute; screening, zero total
  density, exchange work, and strict work do not.
- Instrumented and uninstrumented mixed J values are elementwise identical.
- The counter must be an aligned allocation on the current CUDA device and must
  not alias density, outputs, or the numerical-error flag.
- Absence of a counter is absence of detailed recurrence evidence, not evidence
  of zero mixed work.

## Evidence

The CUDA Fock provider test covers the exact unscreened recurrence count,
screened and zero-density zero counts, replay clearing, numerical identity, and
alias/alignment rejection. On exact head `43b34f21b3b151b27f8c70febeff9b4c196f952d`,
the complete provider target built with CUDA 12.9.86 for `sm_90`, and the focused
`--mixed-census-only` gate passed on an NVIDIA H100 80GB HBM3 (driver 595.58.03).

The target's pre-existing range-derivative gate failed afterward with `CUDA
radial derivative disagrees with displaced CPU range energy`. The same failure
reproduced on exact base `e3907be9a72db15f3edce30f2c9e82bb17422c07` in the
same build and device environment. This evidence qualifies the new census but
does not claim that the full provider target passes on H100.

## Consequences

The execution owner can now report actual mixed Direct-J arithmetic instead of
reconstructing it from aggregate counters. This producer alone does not make a
precision-work record complete: ordered SCF, conversion, retry/fallback,
post-SCF, final-audit, returned-state, and the remaining operator records still
belong to the CUDA-KS execution owner.

## Revisit when

Extend provider-owned counting only when another admitted arithmetic path needs
an exact executed-work record. Keep distinct operator identities and do not
collapse logical applications and recurrence evaluations into one inferred
count.

## References

- [Separate precision-work query from aggregate provenance](2026-09-27-precision-work-query-contract.md)
- #1303
- #1189
- #1190
- #1191

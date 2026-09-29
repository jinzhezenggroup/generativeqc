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
alias/alignment rejection. CUDA compilation and device execution remain required
qualification gates for the implementing change.

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

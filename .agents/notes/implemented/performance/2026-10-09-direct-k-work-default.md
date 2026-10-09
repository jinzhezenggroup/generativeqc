# Decision: default primitive-work-aware exact-K scheduling

Status: implemented
Date: 2026-10-09

## Problem

The qualified work-bucket schedule improves fixed-density K wall time at both
384 and 768 AOs, covers high-angular generated classes, and passes independent
matrix, derivative, force and synchronization gates. The user explicitly asks
to make it the default and open a PR after reviewing the PBE0 cold qualification.
This acceptance does not turn the unmatched cold trajectories into a stable
complete-endpoint speedup claim.

## Decision

During raw-K preparation, an unset or empty
`GENERATIVEQC_DIRECT_K_TASK_SCHEDULE` selects `work`. Explicit `work` selects
the same schedule. Explicit `fill` restores the previous bounded cross-chunk
schedule; `incumbent` restores per-original-chunk execution; `primitive` remains
an experimental alternative. Invalid values still fail closed. Prepared owners
freeze the selection and never reread the environment during execution.

This supersedes only the default-selection parts of the
[fill-default decision](2026-10-07-direct-k-fill-default.md) and
[opt-in work-bucket decision](2026-10-08-direct-k-work-buckets.md).
It does not rewrite their historical evidence or change the qualified kernels.
The frozen benchmark summary continues to record `fill` as the default at
qualification time. Promotion changes the selector, not the measured candidate.

## Invariants

- Zero-initialized non-K topology remains incumbent; J, combined-HF preparation,
  native `dddd`, whole-CTA `ddpp`, unsupported classes and explicit alternative
  lowerings keep their existing workers.
- Screening order, thresholds, recurrence, FP64 policy, spin/nonsymmetric matrix
  semantics and force equations remain unchanged.
- Eight bounded queues per bra group post-screening primitive-pair work and ket
  contraction class; angular class and bra work are fixed in each worker.
- The saturated 8+ bucket does not cap primitive traversal or guarantee bounded
  work ratios for arbitrarily long custom contractions.
- Explicit fill rollback is frozen before output accumulation, not an execution
  retry into partially accumulated matrices.

## Evidence

The [compact qualification](../../../../benchmarks/results/pbe0-k-work-20261008/README.md)
retains conditions, work-count coverage, independent gates and checksums.
Six clean fixed-initial-density K samples per mode reduce median K wall time by
22.82% at 384 AOs and 17.36% at 768 AOs. Generated streaming admissions are
identical per class: totals 23,480,495 and 81,907,624; native/fallback admissions
are not observed by that census. Full-size independent K error is at most
1.8902e-11 against the retained 1e-8 gate. The native J/K and derivative suite,
16 public matrix cases, complete energy/force endpoints, memcheck and synccheck
all pass; both sanitizers report zero errors.

There are only two clean complete cold samples per mode. The observed fill/work
medians are 103.37/91.66 s at 384 AOs and 203.49/198.74 s at 768 AOs, but Fock
counts differ: 26/23 versus 23/23 and 24/28 versus 28/25. These are observations,
not a stable cold gain. The separate 768-AO baseline trace attributes 60.30% of
K device time to classes containing d and only 14.40% to psss/psps; no few
classes monopolize K. Trace sums are not clean endpoint timings.

Selector regression tests check unset/empty/work, explicit fill/incumbent/
primitive rollback, fail-closed invalid values, frozen defaults after environment
mutation, and incumbent zero-initialized non-K topology. The selector-only
promotion reuses the explicit-work GPU qualification rather than repeating the
large cold benchmarks.

Integration onto `master` retains its stricter compiler leaf-emitter boundary:
work-kernel eligibility belongs to `production_selection`, while
`production_exchange_queue` stays independent of selection/registry owners.
Moving the eligibility helper leaves all 22 emitted streaming sources and the
SM120 registry byte-identical; the generated workers are not reimplemented.
The post-integration focused suite passes 1,184 tests. Compiler structure checks
501 modules with zero dependency errors; formatting, staged evidence retention
and the repository ownership/default-inventory checks pass.

## Rejected alternatives

- Claiming a repeatable cold speedup from the two unmatched samples: insufficient
  trajectory and sampling evidence.
- Removing fill or incumbent rollback: loses a bounded independently qualified
  comparison path and a response to target- or contraction-specific regressions.
- Optimizing only psss/psps: inconsistent with measured 768-AO angular coverage.

## Revisit when

Clean matched-trajectory complete endpoints show a regression on another target,
basis, contraction distribution or SCF consumer. Refine high-cost classes or the
saturated bucket only with independent numerical gates, bounded-resource/work
counts and complete-endpoint evidence.

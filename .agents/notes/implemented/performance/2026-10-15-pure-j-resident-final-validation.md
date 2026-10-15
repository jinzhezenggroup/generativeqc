# Decision: reuse the resident final-state proof for large pure-J KS owners

Status: implemented; bounded pure-J default, benefit size is not an enablement gate
Date: 2026-10-15

The exact-direct pure-J default scope below is superseded by the
[1088-AO endpoint hold](2026-10-16-bound-pure-j-direct-final-validation.md).
The original measurements and rationale remain historical; changed SCF work
does not waive the retained failed complete-endpoint nonregression gate.

## Problem

The defining r²SCAN-3c def2-mTZVPP basis increases the AO count relative to
def2-SVP. At 48/96 water-cluster atoms this is 544/1088 rather than 384/768 AOs.
Two order-six force recurrence experiments failed to improve complete Direct
endpoints and were removed. The next bottleneck is not another recurrence:
the final exported-state reference proof repeats scalar host dense products,
whose cubic AO cost grows with this basis.

A source-qualified DF/48 diagnostic measured 0.627 s in native snapshot export,
but only 0.108 s in the subsequent Python validation. Caching the Python check
would address the smaller consumer and introduce new identity/lifetime policy;
no such cache was implemented. The existing strict resident proof removes two
redundant full-AO products (11 to 9) and executes the remaining products through
the existing matrix-library owner. The occupied-rank density reconstruction is
unchanged: the resident trace therefore counts ten products, not nine total.
It was already available explicitly but not automatically selected for pure-J KS.

## Decision

Extend automatic admission to full-precision, single-spin, all-electron,
full-range Coulomb-only KS owners with at least 544 AOs. The native Fock
specification must have unit Coulomb coefficient, exact or density-fitted J,
value order zero and no exchange source. Range correction, fitted exchange,
nonlocal composition, incremental Direct J/K, ECP and lower-precision owners
are excluded. The previous direct-global-hybrid 384-AO policy is unchanged.

The proof consumes the final matrices, not the XC label or a benchmark geometry.
Separate D4/gCP corrections do not change this matrix algebra and are not
removed from the endpoint. The AO lower bound avoids promoting small owners
without a demonstrated endpoint benefit. It is not a broad measured performance
claim for every pure-J functional, basis or GPU architecture.

Explicit `0` retains the reference route; explicit `1` retains its existing
experimental request semantics. Both automatic and explicit requests retain
the library-handle, charged-packet, current-token and unpublished stationary
scratch guards. A repeated read cannot reuse a published W/total-D lease as
scratch. Numerical failures propagate rather than silently retrying the other
route. No new allocations, derivative recurrence or scientific tolerances are
introduced. The selector is included in resource/restart schedule identity.

## Invariants

- Canonical spherical def2-mTZVPP, FP64, original grid/screening/SCF gates,
  D4/gCP and complete host energy/force publication remain unchanged.
- Final proof acceptance and the orbital-energy expression for W are unchanged.
- Independent energy/force gates remain 1e-8 Eh / 1e-7 Eh/Bohr for every call.
- Source inventory, native binary and Python/force consumers must match; a stale
  image or edited AOT receipt cannot qualify a default or README observation.
- Direct and DF require separate complete endpoint cohorts, with all semantic
  SCF work and all five fixed-density warm repeats retained.

## Qualification protocol

The explicit-selector DF/48 pilot is separate from default qualification. It
uses the prior frozen image with `0` versus `1`: warm medians 6.354/5.898 s and
moved-warm 6.374/5.893 s, with all 24 returned E/F endpoints passing independent
gates. It is not evidence that the new automatic policy executed.

The new campaign compares forced `0` against a genuinely unset selector in one
source-qualified binary. The force-domain selector is also unset, preserving
ordinary production scheduling. Sizes 3/12 are non-admission controls; 48/96
exercise the new policy in Direct and DF separately. Cold includes preparation;
each arm retains cold, five warm, moved and five moved-warm calls.

The descriptive robust-benefit label is
`max(2%, 2*(relative_MAD_control + relative_MAD_candidate))`. It is not an
enablement threshold: the user explicitly requested retaining small positive
improvements rather than disabling the route because they fall below 2%.
Scientific accuracy, complete endpoint behavior, resource/lifetime guards and
matched-work nonregression beyond `max(5%, the same noise gate)` remain blocking gates.
These are descriptive cohort checks, not a significance test. Preserve all
samples, including changing SCF work, timeouts and setup failures. Intrusive
trace captures prove actual resident-route execution but do not supply clean
endpoint timings.

The initial Direct/48 cohort improves warm by 2.045% and moved-warm by 1.868%.
The latter keeps its below-2% label; it is neither relabeled as passing that
benefit gate nor discarded. A transient attempt to raise the exact-J boundary
to 1088 AOs was reverted after the user's clarification. The retained production
source again exactly matches the original 544-AO qualification image.

Direct/96 cold has 30 versus 26 SCF iterations; moved has 17 versus 21.
The latter's complete time increases from 327.889 to 380.302 s, and keeps its
failed descriptive nonregression flag. The former decreases from 516.372 to
456.842 s. Neither single unmatched-work sample establishes a causal proof-route
speedup/regression: the cold SCF count already differs before final validation.
The original blanket same-work/nonregression gate is therefore reported as
work-confounded for these phases, not relabeled as a pass or normalized by
iterations. Warm/moved-warm comparisons retain their matched-work gate.

## Evidence

The final source-qualified campaign retains 192 complete comparison calls, all
passing independent E/F gates. Maximum errors are 4.093e-11 Eh and
1.095e-10 Eh/Bohr. Direct/48 warm/moved-warm reductions are 2.045%/1.868%;
Direct/96 4.101%/4.068%; DF/48 6.439%/6.318%; DF/96 12.417%/12.498%.
Both warm populations have matched SCF iteration/Fock-build counts at every
size. The two confounded Direct/96 cold/moved observations remain separate
from these fixed-work warm performance conclusions.

The final local core is byte-identical to the measured image, native SHA256
`ef73b7bd00652e5850737f03d7dcfdab1f738ba5a8dd6d0d5c5429688f88cd40`,
advertising the frozen 1742-input source identity
`953a3addc555388d8f025f0e0eeb7e02af6b3033775d8ad050bc53e582a21d66`.
All 71 benchmark-consumer hashes match on both hosts. Focused host tests pass
62 cases; node5/1619 passes the two native matrix/producer/lease fixtures and
four resource-schedule guards. Node1/8728 confirms actual unset-selector device
proof on both Direct/48 and DF/48. Main complete endpoint jobs are node5/1620
and node1/8726. All GPU runs use finite Slurm allocations and preserve assigned
device visibility. Native test fixture capability adaptation is retained rather
than silently fixing unrelated master API drift.

## References

- [Resident proof implementation](2026-10-10-ks-resident-final-validation.md).
- [Original default](2026-10-11-default-pbe0-resident-final-validation.md).
- [Rejected degree-seven order-six consumer](../../rejected/2026-10-15-r2scan3c-degree7-order-six.md).
- `benchmarks/results/r2scan3c-final-validation-20261015/` retains source-bound
  endpoint reductions, receipts and the independent rechecker.

## Revisit when

Independent complete endpoints justify changing the AO boundary, adding other
producer classes, or identify a real library/resource/scratch-lifetime cliff.
Do not substitute a smaller basis or bypass final-state gates to obtain a win.

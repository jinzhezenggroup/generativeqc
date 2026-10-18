# Decision: bound Fast anchor defaults and retire idle force-phase storage

Status: implemented
Date: 2026-10-18

## Problem

Review of #2602 exposed two gaps in the
[original promotion](2026-10-18-fast-fixed-anchor-delta.md).
The optional five-matrix/scalar anchor survived until plan destruction, although
the force consumer shares the still-live KS owner. Allocation after mandatory
SCF preparation did not protect mandatory force allocations later in the complete
endpoint. A 768-AO lease consumes 23,592,968 bytes: 65 MiB of initial headroom
can admit it yet no longer admit a subsequent 64 MiB force allocation. This is
an ownership counterexample, not a new GPU memory-pressure measurement.

The retained small endpoint is also adverse, not missing evidence. Water-12
was 3.4189% slower in the experiment and 198.9938% slower in the production
observation (36.0519 versus 12.0577 seconds). Production work stayed at 17 target
Focks and two VV10 sweeps while strict J applications increased from two to four.
Arm ordering does not establish a cause or justify subtracting first-use time.

## Decision

Separate scientific eligibility from automatic profitability admission. With the
environment unset, admit anchors across the continuous 192–768 AO interval,
from the independently qualified water-24 lower endpoint through the existing
Fast shape ceiling. This is not a measured-point whitelist or a guarantee that
every molecule in the interval is profitable. Do not widen the scientific domain.
Explicit `GENERATIVEQC_KS_FAST_ANCHOR_DELTA=1`
retains opt-in for already scientifically eligible smaller owners; `0` disables.
Neither setting bypasses precision, model, spin, Direct, resource-ledger, chunk,
or cold-VV10 eligibility. Water-12 therefore uses ordinary Fast by default.

Compose the already-published retirement fix at `54ef30c14` rather than duplicate
or replace its production implementation and deterministic native controls.
At the existing force-memory handoff, validate the exact final token and reject
capture before any mutation. Only under pressure, drain the producer before
freeing idle anchor storage. Anchor eligibility excludes graph/chunk owners;
ERI retirement invalidates its own graph borrowers. Clear all anchor pointers
and validity flags and subtract the exact state capacity only after successful
free. Recheck headroom, retiring optional ERIs only if still needed, and include
both retirements in the receipt. D/W/XC, the final token and last-good warm seed
do not borrow anchors and remain valid. Repeat handoffs cannot double-retire.
Once retired, anchors stay disabled for this owner's warm/cold replays and
failure recovery, avoiding an unplanned optional reallocation before force work.

## Rejected alternatives and invariants

- Constructor-only OOM fallback cannot protect a later mandatory phase.
- Do not free producer or graph-borrowed storage without draining/invalidation.
- Do not retire caches on energy-only or adequately provisioned force requests.
- Do not retain small-domain automatic admission by calling adverse evidence
  noise or supplementary evidence; a single large-domain win does not cover it.
- Do not limit automatic admission to exactly 768 AOs merely because it was the
  first profitable sample. Additional complete endpoints establish a useful
  continuous interval rather than hard-coding individual benchmark dimensions.
- Do not rewrite measured packets or attribute this repair to historical runs.
- The handoff remains advisory, not a free-memory reservation or a promise that
  arbitrary mandatory force envelopes fit. Scientific thresholds are unchanged.

## Evidence and revisit conditions

The production and experimental packets remain frozen under
`benchmarks/results/fast-anchor-default-20261018/` and
`benchmarks/results/wb97mv-anchor-delta-20261018/`. The 20.4504% production
water-96 reduction is one complete cold observation per arm, not a guarantee.

The [size-domain packet](../../../../benchmarks/results/fast-anchor-domain-20261019/README.md)
adds water-24 (192 AO) complete cold E+analytic-force endpoints. Explicit
off/on/on/off screening takes 67.0003/37.3125/37.3926/44.7662 seconds. The final
continuous-default build's off/auto/auto/off campaign takes
67.6631/37.3343/37.3577/44.7356 seconds. The reverse-order pairs still improve
by 16.47% and 16.49%; first-use time remains included in both initial off arms.
Both cohorts reduce Focks from 18 to 16 and VV10 pair sweeps from three to two.
All eight endpoints pass an independent strictly reconverged GPU4PySCF
full-density, matched-moving-grid oracle at unchanged Fast gates. This selects
a useful lower admission floor, not a fitted optimal molecular crossover.

Two additional water-48 (384 AO) oracle attempts time out at 900 and 1000
seconds; the second requests an independent coarse-grid seed. Neither publishes
a converged reference or reaches production endpoints. They remain incomplete
qualification, not accepted interior measurements or production regressions.
The interval policy interpolates between qualified endpoint sizes without
claiming every interior molecule has been timed.

The final scientific source identity is
`5a860446cf0fc138c706f20f6d492a811dcb0d2c9759432c4914cdcfa4ba240b`;
library SHA256 is
`460a1594ffdd67462c5a2d18b718a4889ca8faafdc0845aed9c019725ad41b51`.
Matching-source native GPU lifecycle/oracle gates pass under Slurm job 10228;
the focused Python suite reports 275 passed, nine opt-in cases skipped. The
final default endpoints run under exclusive finite Slurm job 1147. Published
retirement and retention repairs remain composed, not duplicated.

The host admission test covers both boundaries and unmeasured interior sizes
(including 193, 256, 383, 512 and 767 AOs), so this remains an interval. Native
`generativeqc_ks_cuda_tests --fast-anchor-delta` compares against an independent
CPU composition and exercises small-domain default/off/on, strict exclusions,
roomy/stale/captured/pressured/repeated force handoffs, D/W/token preservation,
warm/cold replay and runtime failure/recovery after retirement.

The earlier [handoff proposal](../../proposed/2026-10-18-fast-anchor-force-memory-handoff.md)
preserves the repair's original host-only validation scope. Additional GPU
lifecycle checks do not retrospectively alter the measured endpoint packets.

Revisit automatic size admission only with profitable complete-endpoint evidence
for additional sizes. Reacquiring retired anchors would require a new bounded
owner-lifetime/resource policy, not just reusing the environment setting.

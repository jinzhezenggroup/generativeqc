# Candidate: shared FP32 semilocal geometry for fitted stationary forces

Status: private diagnostic; no public DF Fast promotion
Date: 2026-10-14

The subsequent public-integration decision is recorded separately in
`../implemented/numerics/2026-10-14-public-df-budget-gradient.md`. The private
diagnostic scope/timings below remain historical, not public endpoint evidence.

## Decision and boundary

The ordinary stationary consumer accepts the same private
`fp32_semilocal_geometry` selection as the composite consumer. It calls the
shared typed source ABI, not a new XC implementation. Prepared identity includes
this arithmetic choice so a retained strict owner cannot silently substitute
for a mixed request or vice versa. Mathematical identity remains unchanged.
Both temporary and retained executions pass the choice to their source owner.
Invalid selection types fail before execution; absent live generation/paired
feature proof fails before mixed submission. Work counters measure enqueued
candidate points, not a claim that every guarded XC point used float.

SCF, AO generation, one-electron/Pulay, three-center integrals, fitted metric
response and global reductions remain strict. Failed numerical candidates are
not published. Positive actual mixed work plus a typed numerical failure permits
one complete strict replay, after draining/discarding all rejected sources.
Driver/allocation/identity errors never retry. The prepared lease is invalidated
rather than terminally closed, so a rebuilt strict owner must pass ordinary
admission again. Temporary candidates use the same retained-owner mechanism
for reliable failure accounting. Rejected work and whole retry time remain in
the report; failed strict replays publish nothing. This is not a per-call
observable audit and cannot grant public admission from an ingredient/name.

## Qualification plan

Independent PySCF DF references share only basis and quadrature inputs, not
native weights, matrices or derivatives. Auxiliary-center and fitted-metric
response belong to that independent oracle. Direct-ERI forces are not a valid
reference for a fitted Hamiltonian. The reference adapter therefore accepts an
explicit auxiliary basis and records whether it used density fitting.

The scheduled matrix covers PBE/PBE0/B3LYP/r2SCAN/M06-2X at 24/96 AOs, using
retained cc-pVDZ-JKFIT auxiliary inputs, unchanged strict convergence controls
and `GridSpec(24,8,16)`. Each case includes strict/mixed, frozen density seeds,
actual moved coordinates and strict replay. Gates are 1e-7 Eh energy, 1e-6
strict and 1e-5 mixed maximum force component, 3e-6 mixed-vs-strict RMS, 2e-8
translation and 1e-7 strict replay. This is a candidate qualification, not a
completed README-grid or performance gate. Public batching, HF DF, corrected
DF, RSH/VV10 DF and broader shapes remain separate unfinished work.

## Initial complete-source evidence

Slurm 8344 passed all ten independent fitted-model cases, including cold,
frozen, explicitly moved and strict replay, before the new numerical retry.
The integrated native library was
`c991c0c6bf4f6871759bf6cacf7a98fdc8070353e6f0921476ae50973d26d240`.
Its full ccache-enabled build also incorporated the composite source ABI,
typed native failure statuses and restored ordinary M06 strict owner. These
ten tests are separate from that library's broader regression and from the
subsequent retry/fault-isolation and performance qualification. Raw receipts:
`qualify-df-candidate.log` and `df-fp32-candidate.xml` under the ignored artifact
root. They do not establish a speedup or public DF Fast admission.

## Retry qualification and rejected fixture

Slurm 8346 passed all ten numerical cases with the retry wrapper, then failed
the new fault-injection test: the ordinary owner drains via `stationary_finish`,
not the composite owner's `stationary_finish_span`, so no error was injected.
The fixture now intercepts any source publication entry, asserting positive
actual candidate work before injection. No numerical gate was relaxed.

Slurm 8350/8351 then exposed a real retry state-machine error: the failed
execution lease was still running when resource invalidation was requested.
The host test had mocked execution without entering a real lease transaction.
It now installs/begins the real lease and reproduced both success/failed-strict
retry failures on the host. Marking the candidate transaction failed before
drain/invalidation preserves the lease's existing no-invalidation-while-running
invariant. The corrected focused host suite passed 46 tests. Slurm 8354 passed
the real pending-stream retry, failed-neighbor isolation and subsequent mixed
replay against the independent DF oracle. The accompanying M06 strict-routing
check also passed, but still describes the separately rejected routing binary.

The earlier all-followup host suite passed 310 tests and pre-commit passed before
the final lease fix; final receipts must distinguish that revision. Complete
DF timing follows the successful retry in Slurm 8354 and remains separate from
numerical qualification. Keep failed XML/logs from 8346/8350/8351 rather than
silently replacing their record with the recovery run.

## Complete diagnostic timings

Slurm 8354 completed the ten-population timing matrix after successful recovery.
It retains construction/SCF/full E+F/export/cleanup singlepoints, separately
charged preparation, frozen post-cold/post-move seeds and explicit movement.
Each mode uses the same strict SCF, fitted Hamiltonian and quadrature. Every
attempt passed its independent energy/force gates; none was dropped. These are
private typed-geometry diagnostics, not public PrecisionCalculator batching.

At 96 AOs the strict/candidate ratios (singlepoint; frozen prepared) were:

| Functional | Complete singlepoint | Frozen prepared |
| --- | ---: | ---: |
| PBE | 1.0064 | 1.0772 |
| PBE0 | 1.0046 | 1.0500 |
| B3LYP | 1.0076 | 1.1143 |
| r2SCAN | 1.0113 | 1.1952 |
| M06-2X | 1.0240 | 1.3196 |

Retain the losing 24-AO populations: PBE0 and B3LYP singlepoints were 0.9933
and 0.9962; r2SCAN prepared cold was 0.9952. All 24-AO singlepoint ratios stayed
within about 0.7% of unity. This demonstrates useful warm-source savings for
the more expensive point models, not uniformly accelerated cold work, a
large-system crossover, all-FP32 DF, or the README dense-grid matrix.

The complete `df-fp32-timing.json`, `qualify-df-recovery.log` and recovery XML
were copied locally. Source and native receipts remain on n1 as
`df-candidate-8354-source.tar.gz` and `composite-df-8354.so`. That binary has the
separately rejected **direct** M06 strict-routing experiment; fitted routes
exclude that controller and this DF comparison does not use the regressed
direct comparator. A following rebuild restores direct native routing and
retests the final lease transition and both ordinary/composite consumers.

## Final integrated source check

Slurm 8358 passed all 11 DF and 5 composite checks against the rebuilt final
library `5b7fc0ef88c97ee56246c91c55428b3cb8df972ca5d68fe576a1ce5792bd9599`.
It also passed six direct M06 checks, including the explicit compatibility
comparison (25 unrelated direct tests were deselected for that focused run).
Both real pending-work fault injections passed. The final host suite passed
310 tests, and all-followup pre-commit/diff checks passed. This confirms the
candidate/recovery implementation, not public DF/WB Fast admission. The current
library/source receipts and final M06 timing remain separate from the earlier
rejected routing experiment.

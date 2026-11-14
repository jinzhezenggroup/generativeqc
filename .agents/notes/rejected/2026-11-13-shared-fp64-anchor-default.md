# Decision: share Fast's optional FP64 anchor default with strict owners

Status: rejected; uniform automatic reuse across strict and relaxed targets
Date: 2026-11-13

The [target-selected policy](../implemented/architecture/2026-11-13-target-selected-precision-presets.md)
supersedes this candidate. The user's clarification is that optimization follows
actual convergence/error targets, not mode labels. Preserve this source-matched
trial as the rejected always-on-at-192 proposal; do not relabel its measurements
as qualification of the later target-selected implementation.

## Problem and scope

The user requested that Standard benefit from reusable Fast engineering, not
that Fast lose its existing optimizations to make the profiles appear equal.
The [retained Fast default](../implemented/performance/2026-11-12-retain-qualified-fast-anchor-default.md)
repairs that regression, but leaves strict FP64 owners opt-in. The
[earlier shared-anchor trial](../implemented/performance/2026-11-05-shared-fp64-anchor-trial.md) established
strict scientific eligibility and historical small Standard benefits, while
its incomplete Fast force packaging prevented common endpoint qualification.

## Candidate

Remove budget-presence arguments from the profitability selector. After the
existing independent scientific admission, unset selects reuse from 192 AOs
for both routes. Explicit zero disables it; one admits eligible smaller owners;
invalid values reject even ineligible owners. Preserve Fast's existing floor,
all complete-model/provider exclusions, optional allocation refusal, charged
five-matrix-plus-scalar storage and force-pressure retirement. Strict owners
remain FP64 and retain their convergence and full-density final-state proofs.
This does not promote FP32 or coarser proposal-source screening into Standard.

The core linear controller is unchanged. Delta actions still apply J/K and
long-range K; they are not omitted applications. Nonlinear XC/VV10 see total
density, full targets refresh strict anchors, and each solve resets validity.
Warm, moved, failed and force-pressure owners cannot borrow stale anchors.

## Qualification contract

Freeze the normal SM120 Release source and use a separate hook-enabled image
for native failure/lifecycle tests. For both policies and 192/384 AOs, measure
three alternating OFF/default pairs of complete constructor, preparation,
synchronized energy/forces, host/density export and teardown, each with three
same-owner warm calls. Timings exclude independent reference work and separate
diagnostic traces/memory sampling. Preserve every failed attempt.

Require independent strict E/max-force/RMS-force gates 1e-8/1e-7/1e-7 and
paired gates 1e-9/1e-7/3e-8. Fast retains 1e-5/1e-4/3e-5 independent and paired
gates. Require warm median ON/OFF <=1.05, at least one Standard cold improvement
above max(2%, twice the sum of relative MADs), and no significant cold regression
in the other sampled domains. Native fixture checks exact optional capacity,
failure invalidation, bounded ledger fallback, force retirement and warm recovery.
Public tests cover default/off/on on both sides of the floor and independently
check moved, moved-warm and fresh-moved observables.

Retain actual operator precision inventories and Fock timelines. The narrow
public J/K diagnostic is unavailable for mixed budget owners; preserve its null
counts rather than interpreting them as zero or inventing a complete census.
Separate JSONL traces prove actual anchor/delta selection and warm isolation.
Sampled NVML process peaks include context and may miss instantaneous peaks;
they do not replace the native owner's exact capacity proof.

## Evidence and decision

The [source-matched trial](../../../benchmarks/results/shared-fp64-anchor-default-trial-20261113/README.md)
has all twelve predetermined OFF/default pairs, each with a cold and three warm
endpoints: 96 endpoints total, all passing independent and paired scientific
gates. The normal Release native source identity is
`3b31877336dd65af68a072e89d993f244138aa6e53dc455b8d33b3cb954697cc`
(2,137 inputs); its library SHA-256 is
`ddfd3e5e3835687da83f011e948acb4250e6ded50213c821ff13dd5e251c1834`.
The retained dirty-source patch reconstructs the measured candidate on
`e41c546c40c1fa04db5e319506aac1f9ed330d50`.

| Profile | AOs | OFF/auto cold medians (s) | Median paired reduction | Warm auto/OFF |
| --- | ---: | ---: | ---: | ---: |
| Standard | 192 | 39.5503 / 39.6257 | +0.0028% | 1.00836 |
| Standard | 384 | 206.2542 / 206.4433 | -0.0185% | 1.00099 |
| Fast | 192 | 38.5134 / 34.9192 | +9.4126% | 0.99981 |
| Fast | 384 | 137.1458 / 119.3346 | +13.0275% | 1.00040 |

Paired reductions are the medians of per-pair changes, not ratios of marginal
medians. Neither Standard shape passes the predeclared 2%-or-noise profit gate.
Do not lower that gate, reuse the older Standard speedup claim, or merge the
candidate as a qualified common default. Fast gains preserve incumbent behavior;
they are not new speedups introduced by this proposal. The production stage is
explicitly not promoted, and this implementation remains a draft.

The 192-AO reference was freshly computed using independent GPU4PySCF. The fresh
384-AO reference attempt hit its finite 1,000-second timeout; the trial reuses the
retained independent reference from 2026-11-05 with the exact same scientific
protocol. This is measurement evidence only, not a production oracle dependency.
The native failure/capacity/force-retirement fixture passed under Slurm job 17331.
The four public selector cases passed in job 17335 before it was stopped; corrected
moved/moved-warm/fresh-moved cases passed separately in jobs 17336 and 17337. The
host selection passed 809 tests with six GPU skips. These are focused checks,
not a claim that the entire repository test suite passed.

Separate traces show Standard auto performs 6 anchor + 21/22 delta builds at
192/384 AOs, while Fast performs 4 + 12. OFF and warm cases show none. Standard
iteration/Fock counts are unchanged (27/27 and 28/28); Fast changes from 18/19
to 16/17. Delta counts therefore do not establish omitted linear applications
or a Standard endpoint gain. Sampled process memory is equal at 192 AOs and
about 6 MiB higher with anchors at 384 AOs, not an exact allocation peak.

Failed deployment attempts, the initial incorrect moved-geometry fixture, finite
timeouts, and incomplete campaigns are retained explicitly. The last Standard
384-AO pair was assigned a fresh finite job before its results were available;
the interrupted attempt is not substituted into the paired samples.

## Revisit when

Identify a concrete Standard workload or work-saving implementation that passes
the unchanged complete-endpoint profitability gate while retaining all strict
scientific, bounded-resource, failure and warm/moved lifetime guarantees. Shared
code is useful, but arithmetic identity alone does not prove a profitable default.

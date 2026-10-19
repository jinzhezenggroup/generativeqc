# Proposal: qualify VV10 grids for FP64 and Fast before refreshing timings

Status: proposed
Date: 2026-10-19

## Objective and acceptance

Determine whether the README grid is sufficient, select empirically converged
FP64 and Fast grids, then refresh benchmarks at those selected contracts.
The [acceptance correction](../implemented/numerics/2026-10-19-vv10-fp64-acceptance-correction.md)
withdraws the earlier coarse-grid recommendation; it does not finish this work.

The README FP64 E+F gates are `1e-8 Eh` / `1e-7 Eh/bohr`. Fast uses its actual
configured budget, by default `1e-5 Eh` / `1e-4 Eh/bohr` / `3e-5 Eh/bohr` RMS.
Arithmetic profiles, fixed-grid engine parity and quadrature convergence are
distinct. A reference-only quadrature pass does not admit native Fast or change
the `PrecisionPolicy` profile named `strict`.

## Study design

1. Hold NLC fixed and refine XC radial/angular resolution independently.
2. Hold XC fixed and refine NLC radial/angular resolution independently.
3. Establish replay consistency and stability of the selected dense anchor in
   both refinement directions. Reserve 10% of each observable budget for that
   empirical refinement allowance, leaving 90% for the candidate comparison.
   These are empirical estimates, not rigorous bounds on the integration tail.
4. Check the combined XC/NLC model, actual native FP64/Fast endpoints, both
   original/moved geometries, Direct/DF, and all original 3/12/48/96-atom cases.
   Implementation and quadrature errors share a total observable budget; do not
   grant each approximation the entire budget independently.
5. Select on complete E+F performance only after accuracy passes. Split grids
   are an option, not an implemented native feature or a promised speedup.

`tools/analyze_vv10_grid_study.py` reads successful hash-linked reference
processes and evaluates both budgets, including every saved force replay and
Fast RMS forces. Missing or unstable anchors cannot admit a candidate, and
self-comparison cannot certify convergence. The result explicitly does not
qualify a native endpoint. Source bytes and scheduler/device metadata remain
attached to each measurement.

## Initial evidence and next decision

The first XC sweep fixes NLC64×16×32 and reaches XC144×48×96 on 3 atoms in
Direct and DF. The old XC48×16×32 differs by about `1.52e-6 Eh` /
`2.20e-5 Eh/bohr`; its XC-only Fast budget passes, not its FP64 budget. The
radial and angular anchor checks still differ by about `8.95e-8` and `7.75e-8
Eh/bohr`, respectively. This is insufficient for the reserved FP64 reference
allowance, so the next sweep extends XC to 240×80×160 rather than assuming
144×48×96 is exact. Twelve-atom and NLC sweeps remain required; no grid is
selected for publication from this initial evidence.

The extended three-atom Direct/DF XC240×80×160 anchors pass the reserved
radial/angular stability checks. In Direct, radial/angle refinement differences
are `5.53e-12 / 5.91e-11 Eh` and `1.79e-10 / 3.21e-9 Eh/bohr`. XC144×40×80
differs by `1.68e-9 Eh` / `6.25e-8 Eh/bohr` and passes the **XC-only** FP64
allowance on these three-atom cases; XC120×48×96 still exceeds its reserved
force allowance. This does not select a shared XC/NLC grid or qualify 12/48/96
atoms. All these comparisons retain the existing `1e-8` molecular VV10 density
domain. The final candidate also needs a screening-control check to distinguish
grid refinement from a possible density-domain accuracy floor.

For larger dense sweeps, `benchmarks.vv10_grid_accuracy_sweep` reuses only its
own independently converged GPU4PySCF densities, then reconverges each grid and
geometry at the same tight tolerances. Its distinct schema/phases explicitly
forbid cold-latency claims. A four-geometry real-GPU replay against the earlier
atomic-guess references agrees within `5.06e-12 Eh` / `2.50e-11 Eh/bohr`.
This reduces redundant SCF work in an accuracy study, never the work counted in
the fresh-process timing population. The twelve-atom refinement uses this
validated accuracy-only path; full timing runs still retain one cold, five warm,
one moved and five moved-warm endpoints.

Raw study: `n2:/data/jzzeng/vv10-precision-study-20261019`; caller snapshot
`9a7d409be`. n2 is an RTX PRO 6000 accuracy population, never a substitute for
the RTX 5090 timing population. The ignored local mirror contains the campaign
plan, full force arrays, process outcomes and SHA-256 receipts. The new grids
do not alter native production defaults.

## Full README rerun and setup repair

Keep all eight methods (`hf`, `pbe`, `pbe0`, `b3lyp`, `r2scan`, `r2scan-3c`,
`m06-2x`, `wb97m-v`), Direct/DF, all four original sizes and GPU4PySCF.
Do not replace the overview with a partial cohort or delete unrelated studies.

The first minimal native package omitted stationary AOT artifacts. Existing
cached packages have different source contracts and cannot be forced into use.
Build the five required restricted s/p/d packages with the verified compiler
cache, validate their exact contracts and binary hashes, and copy them beside a
hard-linked native library in a new real runtime directory. Do not mutate the
old measured build or change its library hash. Repaired measurements occupy
`attempts/aot-fixed`, with a separate package receipt; original failures remain.

`tools/qualify_readme_rerun.py --native-attempt aot-fixed` requires the planned
repair receipt and links the prior attempt. A missing repaired result cannot
fall back to an earlier timing. Its qualification is same-grid FP64 parity only;
Fast policy journals cannot silently enter that timing cohort.

Timing raw root: `n1:/data/jzzeng/readme-full-grid-rerun-20261019`, with the
separate existing n4/n5 GPU4PySCF/96 jobs. A native Fast pilot uses its actual
policy/budget on 3/12 atoms before selecting a Fast grid. Neither a small pilot
nor host regression tests complete the full precision/timing qualification.

The first eight native Fast pilot populations (Direct/DF, 3/12 atoms, shared48×16×32
and shared48×12×24) completed 96 measured E+F calls. Their same-grid independent
energy, maximum-force and RMS-force gates pass; maximum observed arithmetic
errors are `4.27e-12 Eh` / `7.84e-8 Eh/bohr` / `2.96e-8 Eh/bohr` RMS. Preserve
the per-call actual precision/fallback ledger: not every Fast phase executes
the same mixed operators. These observations do not certify grid convergence,
the combined error budget, or the larger benchmark cases.

That older consumer supplied `1e-12` / `1e-10` energy/density tolerances to
**both** modes. These explicit ceilings suppress Fast's normal budget-selected
stationarity controls, so retain it as a **tight-control Fast pilot**, not the
default Fast performance population. The updated consumer leaves Fast's solver
selection to `PrecisionCalculator`, retains the legacy FP64 controls unchanged,
and records requested and initially selected controls explicitly beside each
call's actual refinement/fallback work. The independent reference protocol stays
tight and its accuracy gates are not relaxed. Remeasure the default-control Fast
cohort in a separate attempt before selecting performance results.

The default-control pilot (job 10334, a distinct raw directory) has now completed
the same eight populations and 96 calls. All same-grid Fast gates pass, but
Direct errors reach `2.83e-9 Eh` / `4.45e-5 Eh/bohr` / `1.46e-5 Eh/bohr` RMS,
not FP64 force accuracy. Direct uses mixed stages plus strict refinement; all
DF calls report effective 64-bit execution. Retain those actual execution labels
rather than describing every Fast request as mixed arithmetic or a speedup.
The selected initial native controls are `1e-10` energy / `1e-8` density, versus
the explicitly pinned `1e-12` / `1e-10` FP64 controls. None of these same-grid
passes establishes the total Fast budget after quadrature error is included.

The complete initial twelve-atom XC sweep still shows `7.13e-6 Eh` /
`2.67e-5 Eh/bohr` versus its provisional XC144×48×96 anchor at fixed
NLC64×16×32. Even XC144×40×80 differs by about `1.03e-6 Eh/bohr` on 12 atoms;
its three-atom XC-only pass must not be generalized. The denser twelve-atom
anchor study, 48/96-atom accuracy sweeps and actual default Fast endpoints
remain separate required populations.

The initial three-atom NLC120×32×64 anchor is not FP64-stable under the reserved
allowance: its radial check differs by `4.83e-9 Eh` / `1.11e-8 Eh/bohr` in
Direct. Extend NLC and run explicit density-threshold controls; do not select
the older candidate merely because its difference fits one full observable
budget. The sweep now records a requested screening threshold, and the analyzer
validates nested accuracy controls. Implicit source-bound legacy journals are
not silently interchangeable with explicit protocols or different thresholds.

No final README data or grid recommendation is replaced by these partial
studies. The shared evidence store is at its existing aggregate cap; retain full
raw journals in the declared ignored mirrors until a deliberate replacement or
lossless compaction within this study and the freshly rerun README cohort fits.
Do not delete unrelated studies or raise the cap to publish partial evidence.

## Larger reference convergence and full-Fock control

The first 48-atom accuracy sweep (job 3443) failed its tight SCF solve before
returning any E+F endpoint. It used the split-reference helper's default
incremental Fock policy, unlike the full-density rebuilds in the README
reference benchmark. Preserve that failed journal; do not count it as an
accepted grid or relax its tolerances. The old 96-atom accuracy sweep (3444)
was deliberately cancelled with a saved scheduler receipt to prioritize a
full-Fock control, not declared a measured 96-atom numerical failure.

Accuracy-only sweeps now bind full-density rebuilds **after** optional DF and
mixed-RSH decoration; binding a method to the pre-decoration object could retain
the wrong operator. Record the Fock policy and the last SCF convergence payload,
including unsuccessful solves. A separate source-frozen control first replays
three-atom Direct/DF, then retries the failed larger case. At this point the
repair is an implemented measurement setup, not evidence of a successful
48/96-atom FP64 grid selection. Do not pool old/new Fock-control timings or use
an unverified reference as an independent acceptance oracle.

The full-Fock three-atom replay has completed eight paired original/moved
comparisons (two grids, Direct/DF), differing from the prior reference by at
most `2.65e-12 Eh` / `2.63e-11 Eh/bohr`. Larger convergence is still pending.
The completed extended twelve-atom Direct XC sweep rejects even its
XC240×80×160 anchor: radial and angular checks differ by `2.91e-7` and
`1.20e-7 Eh/bohr`, respectively. Its old XC48×16×32 difference is
`7.17e-6 Eh` / `2.75e-5 Eh/bohr`. The reference-only XC Fast gate passes;
FP64 does not. Extend the separate full-Fock twelve-atom study through
480×160×320, with independent radial/angular checks, rather than labeling a
three-atom pass or an unstable larger anchor as a universal grid choice.

The first default-control Direct/48 Fast population (job 10337, shared
48×16×32) completed all twelve E+F calls. Its same-grid maxima are
`2.35e-9 Eh` / `1.27e-5 Eh/bohr` / `3.96e-6 Eh/bohr` RMS and pass Fast's
arithmetic-only gates. Its quadrature and total endpoint budgets remain
unchecked; preserve this distinction for the larger-size timing refresh.

## Density-screening evidence

At fixed XC240×80×160 / NLC120×32×64 on three-atom Direct, changing the
VV10 density threshold from `1e-8` to `1e-12` shifts energy/maximum force by
`1.05e-8 Eh` / `1.31e-8 Eh/bohr`. The `1e-10` to `1e-12` differences are only
`1.60e-10 Eh` / `1.88e-10 Eh/bohr`. These are controlled finite-grid observations,
not proof of an exact unscreened limit or of larger-system screening error.
Nevertheless, the existing `1e-8` domain difference already consumes the full
FP64 energy budget on this small case. A defensible total-error selection must
qualify screening too, not rely on cancellation between grid and domain errors.

The pending full-Fock twelve-atom ultra-dense reference sweeps therefore use an
explicit `1e-12` density threshold. Native production thresholds are unchanged;
changing their scientific policy requires its own source-matched native and
derivative qualification. The Fast reference sweep retains `1e-8` to study the
current native domain, but still needs the total error budget and larger cases.
An additional lower-threshold check on the final dense reference remains needed.

The repaired full-Fock 48-atom Direct probe has now reconverged both geometries
at both XC48×16×32 and XC64×24×48, with NLC fixed at 48×12×24 and unchanged
SCF tolerances. Their observed XC-only differences reach `3.94e-5 Eh` /
`3.51e-5 Eh/bohr` / `1.95e-5 Eh/bohr` RMS. Even Fast's energy budget is
exceeded by this two-grid difference. It does not establish that XC64 is a
converged anchor or select a final Fast grid; complete the separate 48/96-atom
reference sweeps and the combined/native gates before a timing recommendation.

## Joint endpoint gate and common-anchor controls

The endpoint qualifier `tools.qualify_vv10_endpoint_precision` now compares
every native or GPU4PySCF timing endpoint directly with a joint dense, low-screen
reference. Five reconverged controls share that anchor: XC radial/angular, NLC
radial/angular and a lower density threshold. Sum all observed control norms and
anchor replay drift into one empirical reference allowance, reserve at most 10%
of the total budget for it, and add it to each candidate error. Do not add
independently passing full-budget component gates, reuse different fixed-axis
anchors, or treat this empirical allowance as a rigorous tail bound. Require
tight full-Fock SCF controls and all twelve timing endpoints; native Fast must
use application-selected public controls. GPU4PySCF timing candidates are checked
independently too, without relabeling their arithmetic as native Fast.

On 2026-10-19 at 08:07 +08, finite eight-hour n1 Slurm jobs 10358 (Direct) and
10359 (DF) started new **accuracy-only three-atom** common-anchor controls. Their
experimental anchor is XC240×80×160 / NLC240×40×80 at rho1e-12, with XC192×80×160,
XC240×64×128, NLC192×40×80, NLC240×32×64 controls and a separate rho1e-14 replay.
These dimensions are not a recommendation, and no result is yet accepted.
They use the source-frozen full-Fock reference package and supplemental runtime
data from the n2 study, with matching SHA-256 receipts. Native production
screening and shared-grid ownership are unchanged. Raw data lives under
`n1:/data/jzzeng/vv10-precision-study-20261019/joint-full-fock`; do not pool these
accuracy-only phases with timing populations, or extrapolate them to larger
sizes. Larger full-Fock XC and NLC studies and the all-method timing refresh
remain in progress; a joint gate implementation is not goal completion.

The subsequently collected 48-atom DF full-Fock probe confirms the same XC
problem: XC48×16×32 versus XC64×24×48, at fixed NLC48×12×24, differs by
`3.93866e-5 Eh` / `3.50008e-5 Eh/bohr` / `1.94598e-5 Eh/bohr` RMS across the two
geometries. Its energy difference also exceeds the Fast budget. The three-atom
DF screening control gives `1.05011e-8 Eh` / `1.30898e-8 Eh/bohr` for rho1e-8
versus rho1e-12, and only `1.59318e-10 Eh` / `1.86715e-10 Eh/bohr` for rho1e-10
versus rho1e-12. Neither is a claim about the exact unscreened limit. The old
twelve-atom extended XC sweep has now completed both Direct and DF: both
XC240×80×160 anchors fail the FP64 stability allowance, while their XC-only Fast
checks pass. The all-original-method rerun audit is 139 measured / 1 incomplete /
4 not run, using the explicit `aot-fixed` attempt; this remains a same-grid
timing/oracle audit, not qualification of total quadrature precision.

At 08:17 +08, n1 jobs 10360 (48 atoms, finite 16 hours) and 10361 (96 atoms,
finite 24 hours) also started reference-only full-Fock NLC sweeps, Direct then
DF. XC is fixed at 144×48×96, screening at the current native rho1e-8 domain;
NLC varies over 48×12×24, 48×16×32, 64×16×32, 96×24×48, 96×32×64,
120×24×48 and 120×32×64. The final three provide radial/angular controls if
stable. Save under `n1:/data/jzzeng/vv10-precision-study-20261019/nlc-fast-full-fock`.
They use the same source-frozen package as the joint three-atom study. This
closes a measurement-coverage gap, not the qualification itself: the anchor may
still require refinement, low-screening controls remain separate, and these
accuracy-only runs cannot replace complete cold/warm benchmark timings.

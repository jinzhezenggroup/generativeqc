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

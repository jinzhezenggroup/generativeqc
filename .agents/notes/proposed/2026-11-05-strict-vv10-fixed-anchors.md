# Decision: explicit strict-FP64 provisional VV10 anchors

Status: proposed; default-off draft
Date: 2026-11-05

## Problem

Direct WB97M-V cold execution spends substantial time in repeated full/range
J/K. The existing Fast owner already reconstructs provisional operators from
a fixed FP64 density anchor. Strict callers could reuse that linear structure
without introducing mixed arithmetic, CPU/reference work or new tolerances.

## Decision

Extend the existing explicit `GENERATIVEQC_KS_FAST_ANCHOR_DELTA=1` selector to
strict-FP64, one-spin exact full/range Direct VV10 owners with chunk width one.
The selector remains off by default for strict callers. AUTO alone does not
admit this path, and existing Fast defaults are unchanged. Resource-ledger
owners retain their incumbent path; optional-ledger admission from the research
prototype is deliberately excluded from this patch.

Only provisional J/K uses a density difference from a fixed full-density
anchor. Nonlinear XC always consumes total density. Full targets and final
closure rebuild physical operators. Existing allocation fallback, warm
isolation, failure invalidation and drained force retirement remain mandatory.
The journal records actual admission and validated work, not just a requested
flag. Checkpoint execution identity includes the selector.

## Research evidence, not isolated-PR qualification

Issue #2976 retains source/image/helper identities and failed attempts. Research
source1499c5768e6102ba82d2c8fabb5a092e5d8fbfbca796715046b7146b2ca6fdc5,
normal O3 SM80+SM120 production image, completes A100 LSF225438 and RTX5090
Slurm15871. Complete E+F observations: one cold, median of three own-density
warm calls, identical native settings and strict FP64.

| Device | Cold OFF | Cold ON | Warm OFF | Warm ON |
| --- | ---: | ---: | ---: | ---: |
| A100 |156.998548s|149.112313s|21.602472s|21.621555s|
| RTX5090 |237.314515s|228.518151s|18.552084s|18.710250s|

These are5.023% and3.707% cold reductions in one population, not robust/default
qualification. Warm ratios1.000883 and1.008525 pass the predeclared1.05 paired
guard. Both large diagnostics execute6full anchors/22deltas over28cold
iterations, zero invalidations and no warm anchor work. Main29J/58K applications,
23provisional/5target Focks and4VV10 sweeps remain accounted; density deltas
are not omitted operator applications.

Native-paired E/maximum-force/RMS-force gates1e-9/1e-7/3e-8 and independent
1e-5/1e-4/3e-5 pass every completed observation. A100 has fresh stock GPU4PySCF
with authenticated loaded identities. RTX5090 uses that matching A100 oracle
for science only, never cross-device timing. A separate testsON hook fixture,
Slurm15879, passes independent-oracle, strict/Fast admission, AUTO exclusion,
allocation accounting, failure/recovery, warm isolation, retained final D/W/token,
force retirement, no-double-retirement and ledger-fallback gates.

These measurements are **not this isolated PR source**. The research source
contains other retained force experiments and an unneeded optional-ledger
extension. Exact-PR-source production build, real-device endpoints and
independent gates remain required before readiness or any default promotion.
The A100 WB cold disparity is still61.153091s; its warm-force owner and the
approximately1.576s r2SCAN gap are not solved by this patch.

## Rejected alternatives and provenance traps

- Exact density-Hermite convolution loses complete warm E+F by2.786% on A100
  and1.921% on5090 despite reducing radial groups. It is excluded. The combined
  study's original exact cache-hit/preparation equality gate remains failed;
  physical derivative work and supplemental bounded cache accounting pass.
  Do not relabel that whole study successful or promote from operation counts.
- Ordinary incremental and disabling frozen VV10 were already slower in the
  independently audited A100 study225418. Do not repeat those sweeps.
- V3's changed cuda_ks.cpp did not compile after a copy2 overlay preserved an
  older source mtime. A newly exported source hash can coexist with an old
  object. V5 actually recompiles the changed unit. Preserve fresh mtimes on
  changed overlays and check compile receipts; never override source identities
  or clear the shared cache to mask provenance problems.
- A100 conda DT_RPATH selected old libc under a newer private loader. Bounded
  CPU-only225436/225437 diagnosed and verified child-only private-loader
  `--inhibit-rpath ''` repair. No shared interpreter/runtime change or production
  CPU fallback is part of this patch.

## Revisit

Keep this draft and strict defaults off until exact-source endpoints justify
readiness. Broader ledger, UKS or multi-device admission needs separate resource,
lifecycle and independent scientific qualification.

# Decision: do not blindly enable incremental J/K for default CUDA HF

Status: rejected
Date: 2026-11-05

The subsequent [convergence/composition repair](../implemented/numerics/2026-11-05-incremental-hf-convergence-composition.md)
supersedes the implementation boundaries diagnosed here. Measurements below
remain historical pre-repair evidence; the blind default-toggle rejection is
not rewritten as a performance claim about the repaired source.

The later [96-atom qualification](2026-11-05-hf-incremental-default-promotion.md)
also defers default promotion after repair: Standard cold's roughly 1% benefit
is below the shared performance floor, despite much better moved/strict results.

## Problem

GPU4PySCF uses density-difference J/K in direct HF, but enabling the existing
GenerativeQC opt-in controller is not an isolated replacement of a full-density
contraction. It also changes convergence admission and availability of other
optimized routes. Earlier generic-build incremental measurements cannot establish
the economics of the current production defaults.

## Decision

Retain the existing opt-in policy. This rejects a blind default toggle, not
incremental J/K as an algorithm or its potential use in a qualified domain.
No production code, convergence controls, final-state checks, or defaults change.

## Measured scope and provenance

The diagnostic uses restricted, all-electron RHF/def2-SVP spherical FP64 water
clusters: 12/24/48 atoms, or 96/192/384 AOs. Each arm has four independent
cold/warm/moved sequences in AB/BA/BA/AB order. Cold includes Calculator
construction, preparation and complete host energy/analytic-force return;
process imports and common tiny-water bootstrap are excluded, and cleanup is
reported separately. Warm and moved calls replay each sequence's frozen cold
density, with updates disabled. Both arms use their own native initial guess.

- Source revision: `7618eb29e696daab073c027728e88a9c78a8110c`.
- Native source identity:
  `66c97c2c7753bef777fdc830a5981aed2c5eec085c32a8e7fd039f3eb952ea46`.
- Library SHA256:
  `daecfd107c36840101a7a61d3ffd63ebdcadfac466758ddd795cf9b0b497df34`.
- Release CUDA 12.9.1, official sm120, AOT enabled, fast compilation disabled,
  verified sccache 0.16.0, OpenBLAS CPU provider.
- Slurm n1/node1 RTX 5090 jobs `15956` (Standard) and `15957` (strict), one GPU
  and eight CPU threads; both retain assigned `CUDA_VISIBLE_DEVICES=3`.
- Screening stays `1e-12`; incremental refresh interval stays eight.

The independent tight CPU PySCF arrays and exact original/displaced inputs come
from the retained
`benchmarks/results/incremental-hf-full-energy-refinement-20261015/validation.json.xz`.
The harness records its SHA256 and full fixture values. These are diagnostic
static-geometry measurements, not an HF default-promotion qualification.

## Complete endpoint medians

Seconds, OFF / incremental ON:

| AOs | Standard cold | Standard warm | Standard moved | Strict cold | Strict moved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.362 / 0.919 | 0.117 / 0.117 | 0.483 / 0.751 | 0.452 / 1.032 | 0.761 / 0.819 |
| 192 | 1.527 / 2.204 | 0.251 / 0.252 | 1.305 / 1.776 | 2.337 / 2.395 | 1.988 / 1.828 |
| 384 | 5.045 / 7.180 | 0.734 / 0.733 | 3.882 / 4.896 | 7.468 / 7.488 | 6.368 / 5.101 |

Standard uses the native defaults `1e-9` energy / `1e-5` density. Strict uses
`1e-12` / `1e-9`. Within each campaign both arms receive the same public
controls; the controller's existing stricter physical admission is not bypassed.
Strict warm 96-AO latency improves from 0.162 to 0.117 seconds because its SCF
branch changes from two iterations to one, **not** because it executes delta
builds. Other strict warm medians are effectively unchanged.

All 144 formal E/F endpoints pass their predeclared independent-reference gates.
Standard gates are `2.5e-6 Eh/atom`, maximum force `2.5e-5 Eh/Bohr`, and RMS
force `7.5e-6 Eh/Bohr`; observed maxima are `3.37e-11 Eh` absolute,
`6.34e-7 Eh/Bohr` maximum force and `2.67e-7 Eh/Bohr` RMS force. Strict gates
are `1e-8 Eh` absolute and `1e-7 Eh/Bohr` maximum/RMS force; observed maxima
are `8.19e-12 Eh` and `1.76e-10 Eh/Bohr` maximum force.

## Work and interpretation

At 384 AOs, Standard cold OFF uses ten target SCF iterations/Fock builds. ON
uses sixteen iterations and eighteen target Fock builds: four full SCF, twelve
delta SCF and two final full builds. Both additionally report seven preliminary
initial-guess Fock builds. Standard moved uses seven iterations/nine target
Focks OFF versus fourteen iterations/sixteen target Focks ON. No target retries,
discarded target iterations or warm fallbacks occur in either campaign.

At the same size, strict moved improves about 19.9% despite increasing from
fourteen iterations/fifteen target Focks to sixteen iterations/eighteen target
Focks. This demonstrates a useful delta-work regime, not a default-wide win.
Every incremental warm sample executes one full SCF build, zero deltas and zero
final full builds; its parity cannot be credited to delta contraction savings.
Streaming quartet counters at 192/384 AOs are unavailable, not zero. Do not
infer a complete quartet-work reduction from full/delta build counts alone.

Relevant current ownership boundaries in `src/scf/cuda_rhf.cpp`:

- `qualified_standard_owner` excludes incremental J/K, restoring the strict
  physical-residual ceiling even when public E/D settings are Standard.
- Incremental J/K makes optional ordinary resident-value and preliminary cold
  screening routes ineligible. Eligibility alone does not prove which route
  was exercised in every OFF sample.
- Actual delta work cannot license ordinary converged-Fock reuse; full-energy
  refinement and final physical canonical checks remain required.

These source boundaries and measured work explain why a toggle is not a clean
kernel-only comparison. No isolated ablation apportions the total slowdown
among them; do not claim that any one boundary causes the whole regression.

## Evidence retention and limitations

Full samples, forces, fixtures, harnesses, failed readiness/compile attempts and
compiler-cache statistics are in ignored local
`.artifacts/delta-jk-20261105/` and the owned n1 directory
`/data/jzzeng/qc888-delta-jk-20261105/`. These are local diagnostic artifacts,
not a new permanent benchmark publication or external backup.

Ancillary live CPU audits retain their failures: the initial `1e-13` energy
request does not report convergence for the 384-AO pair, although E/F differences
are small. A subsequent full-Fock original-geometry confirmation converges at
`1e-11` energy / `1e-9` gradient and agrees within `4.10e-12 Eh` and
`2.44e-11 Eh/Bohr`; its moved continuation exceeds the overall audit time limit.
One earlier audit harness API error is also retained. These ancillary attempts
are not relabelled as successful independent-reference qualifications; the
formal GPU comparisons use the original retained tight references.

The first build supervisor times out, but its owned Ninja continuation finishes
before a separate successful incremental build check and any formal GPU run.
The failed readiness job is not a measured endpoint. Cache/source identity is
not overridden, and no other worktree or shared compiler server is stopped.

## Revisit when

An incremental owner can retain independently proved Standard convergence and
physical final-state semantics, compose with applicable cached/resident routes,
and demonstrate complete cold/warm/moved wins with trustworthy operator-work
accounting. Preserve bounded refresh/refinement and numerical gates. Broader
RHF/UHF, basis, size and resource-domain qualification remains necessary before
changing a default.

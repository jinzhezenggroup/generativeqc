# Decision: default eligible J-only residual work to conservative shell-pair tiles

Status: implemented on the promotion branch; qualification pending below
Date: 2026-10-18
Base: master e4aff7810b0462204831eb5168f19b6e217ef520

## Decision and scope

At the user's request, default the existing J-only MD residual owner to the
conservative 8x8 shell-pair tile prefilter. Unset or `=1` selects tiles;
`GENERATIVEQC_MD_J_RESIDUAL_TILES=0` retains the individual-candidate traversal
for paired diagnostics. Validate and freeze the selector during preparation.
K never uses the tile prefilter. Retain existing optional MD admission,
128-MiB cap, resource-failure rollback and normal-J fallback; allocate no new
resident tile queue or density-dependent host inventory.

This supersedes the default policy, not the numerical contract, in the
[initial experiment](2026-10-18-md-j-residual-tile-prefilter.md). Maximum and
density products round upward, while the uniform-eligibility minimum product
rounds downward. Surviving candidates retain every incumbent shell/system,
density and per-orientation AO predicate. FP64 recurrence, density error budget,
XC representation, force/response ownership and independent oracles are unchanged.

Keep event tracing and work census off by default. The earlier ~1.85% prepared
J-only device-time observation is not a complete-endpoint speedup claim or a
promise for other molecules, spin/layout classes or devices.

## Work-census invariant

The geometry producer GPU-compacts active pairs; pair ordering, and therefore
which unrelated pairs share a tile, can differ between freshly prepared owners.
Do not require identical tested/tile-rejected partitions even between explicit
`=1` and unset/default owners. The invariant for each angular class is the full
domain `tested + tile_screened`, admitted shell tasks, contracted primitive
products and uniform density consumers. Within an unchanged owner the census
must replay identically. Zero density must reject all tiled candidates before
individual tests and produce an exactly zero J matrix.

The promotion tests exercise `=0`, `=1` and unset policies against independent
libcint for Cartesian/spherical RKS/UKS nonsymmetric densities. They also verify
plan-frozen selectors, failure propagation, resource lifetime, replay and exact
zero-density behavior. This replaces a scheduling-sensitive counter assertion;
it does not relax a numerical, task-domain or primitive-work gate.

## Complete-endpoint qualification protocol

Use the final same-library source on n1 / RTX 5090 under finite Slurm allocations.
Three pairs each contain two fresh Python processes on the same allocated GPU,
with pair orders `0/default`, `default/0`, `0/default`. Separate pairs may use
different physical GPUs; retain device identities and compare within each pair.
Disable both diagnostic fences/counters. Use 96 atoms / 1088 spherical AOs,
the full method-defined r2SCAN-3c basis and corrections, no density fitting and
the README driver without an intrusive capture wrapper or supplied density.

Each process completes cold, warm, moved-geometry and moved-warm energy/force
endpoints, with one warm replay per geometry. The independent retained oracle's
scientific settings and all selected numerical records are unchanged: project
only `repeats` and its textual `warm_definition` from five replays to one, and
retain one original warm record per geometry. Preserve original/projected hashes
and derivation provenance. No native result is substituted into the reference.

Keep actual native Fock counts, independent energy/force gates, physical residuals,
preparation and complete endpoint times separate. Promotion does not establish a
large or universally robust speedup; report measured paired results with their
limited sample size rather than equating avoided candidates with elapsed time.

## Evidence

The final library SHA-256 is
`5ca1225535852f611b5814d2872455b7c1bfbe8be31543b25e8be76069d8c314`.
Slurm 10102 passes all 41 independent resident CUDA cases, including unset,
explicitly enabled and disabled residual tiles. The related host suite passes
100 tests, with 41 real-GPU cases skipped locally and run separately on n1;
Ruff, clang-format and ownership inventory checks pass. Relative to
master, maintained handwritten-scientific CUDA grows by 139 noncomment lines,
runtime CUDA by zero; the new code is bounded scheduling/rejection, not a second
Coulomb recurrence or a production CPU/reference consumer.

The multi-owner sanitizer replay at 64 GiB host memory reports handled optional
`cudaDeviceSetLimit`/`cudaGetLastError` out-of-memory API errors, not invalid-memory
access reports. It is not a passing sanitizer receipt. An isolated unset/default
UKS spherical partial-screening oracle case passes unsuppressed memcheck in Slurm
10106. Slurm 10108 repeats legacy/default cases in separate processes with work
counting and event tracing enabled, 128 GiB host memory, and unsuppressed API
reporting under all three sanitizers. All six isolated runs pass: memcheck and
synccheck report zero errors for both policies; racecheck reports zero errors,
warnings and hazards. These are scoped independent-oracle cases, not full-endpoint
sanitizer qualification. The changed process scope and host-memory request do not
establish the cause of the earlier optional-stack allocation failure.

The n2 attempt (Slurm 3435) stops before CUDA testing: its running NVIDIA kernel
module is 595.91.07 but installed NVML is 595.99.02. No system driver or device
visibility is changed. Retain that failed environment receipt rather than
claiming larger-memory qualification.

Complete-endpoint pair 3 (Slurm 10105, GPU UUID
`GPU-d05490f6-a867-d1a0-f13f-8bf4b1116387`) completes both fresh processes in
legacy/default order. All independent energy/force gates pass:

| Phase | Legacy complete seconds | Default complete seconds | Legacy/default Focks |
| --- | ---: | ---: | ---: |
| cold | 268.291989 | 266.256284 | 24 / 24 |
| warm | 57.433399 | 56.486135 | 1 / 1 |
| moved | 170.116598 | 176.995186 | 13 / 14 |
| moved-warm | 57.535810 | 56.561501 | 1 / 1 |

The maximum energy error is `2.319211489520967e-11` hartree, maximum force error
`6.936318078069881e-11` hartree/bohr, and maximum physical residual RMS
`1.4895364899767604e-12`. Cold is about 0.76% faster and warm about 1.65% faster,
but moved takes one extra Fock and the sum of all four complete endpoints is
553.377795 seconds legacy versus 556.299106 seconds default (0.53% slower).
This single pair is not evidence of an overall endpoint win.

Pairs 1/2 (Slurm 10104/10103) reach the finite 1000-second first-cold timeout, one
in each schedule, without returning a cold endpoint. They are unresolved runtime
observations, not valid timing samples or evidence of an advantage. Keep the PR
draft until the required endpoint qualification is resolved; do not count initial
intrusive captures, superseded/aborted jobs or incomplete arms as completed
promotion cohorts. Receipts live under the ignored remote task's
`default-qualification/` and are copied into the local ignored artifact tree.

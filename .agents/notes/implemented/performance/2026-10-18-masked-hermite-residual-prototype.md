# Decision: opt-in exact masked-Hermite residual J

Status: implemented (prototype only; not promoted)
Date: 2026-10-18

## Problem

The source-ordered residual J shares primitive Coulomb recurrences, but expands
AO integral components before applying the directional Schwarz mask and density.
Uniform Hermite density contraction cannot simply replace a partially screened
source without changing the Hamiltonian. Tile tuning alone does not remove the
AO-component contraction or orbit-publication work.

## Decision

Generate a small-class consumer in the integral compiler, reusing native E
transforms, Hermite indexing and `fill_coulomb`. Leave native admission and policy
ownership intact. `GENERATIVEQC_MD_J_MASKED_RESIDUAL=1` is frozen at plan creation;
the default is off. Initially support pair-angular `(2,1)` and `(2,2)` only.

For each canonical AO pair, retain separate forward/reverse entries with their
own bounds and alpha-plus-beta density. Sorting finite nonnegative input bounds
makes the original product predicate a suffix for every directed output.
Find cutoffs by multiplying/comparing FP64 values, not dividing the threshold;
reuse suffix density sums and one R contraction per distinct nonempty cutoff.
There is no sparse matrix, molecular-size scratch or new resident allocation.

The important symmetry step is not to assume the incumbent canonical quartet
domain is itself a suffix. Replace orbit scatter with directed consumers over
canonical AO pairs. Distinct shell-pair tasks need both consumers; identical
shell-pair tasks need only one, which already visits all ordered output/input
combinations. Both primitive orientations are still integrated. Reversed
consumers transpose the R contraction while retaining canonical ket parity.

Uniform sources remain on the existing MD route. Invalid/nonfinite/negative
bounds, capacity misses and overly diverse masks fall back before publishing
anything. The small-group admission estimate bounds grouped radial terms only;
it does not account for sorting, suffix construction, sparse E coefficients or
atomic throughput. It cannot be presented as measured performance profitability.
The separate masked kernel specialization keeps incumbent scratch specialization.

## Rejected alternatives

- A single unrestricted density projection changes partially screened J.
- Dividing epsilon by a bound changes FP64 threshold ties and zero/overflow cases.
- Merging reverse AO density first loses nonsymmetric directional-mask semantics.
- Keeping triangular quartet exclusion inside each suffix breaks its proof.
- Default enabling before complete endpoint qualification mistakes algebraic
  equivalence for floating-point/SCF/performance acceptance.

## Evidence

Compiled host tests execute the generated planner, suffix sums, grouped R
contraction and AO projection against an independently enumerated ERI-orbit
sum. They cover duplicate shell/AO pairs, multiple primitives, asymmetric AO
bounds, UKS/nonsymmetric density, neighboring systems, invalid bounds, complex
mask fallback, exact FP64 equality, overflow and underflow. Optional-owner
tests retain allocation-failure/ledger behavior with all three policy settings.

Real-device standalone tests on n1 / RTX 5090 (CUDA 12.9, sccache 0.16.0), through
finite `main` / `gpu:5090:1` Slurm jobs, pass 60 cases against both the incumbent
and an independent closed Hermite derivative sum using hypergeometric Boys
values. Slurm 10164 repeats all 60 under memcheck (zero errors) and racecheck
(zero errors/warnings/hazards), preserving scheduler device visibility `2`.
Both counted and uninstrumented specializations, coarse and complex masks run.
Local focused/adjacent tests pass 1234 cases with 88 opt-in GPU/cold cases
skipped; compiler structure checks report zero dependency errors. The local
CUDA 12.4 compile also succeeds. These are residual-kernel correctness tests,
**not** molecular SCF or complete-J timing evidence.

### Source-matched 96-atom complete J

The performance case is the 96-atom water r2SCAN-3c endpoint with 1088 spherical
AOs and def2-mTZVPP, not the older 768-AO PBE0 case. The source-matched comparison
uses PR #2600 head `aacbccc275274cb70e36b76a646750fe667acf3e`, retaining its
default tiled residual and angular-three reciprocal schedules. Its compact E
layout must be accessed through native `md_j_transform`; a dense pointer stride
is not a compatible consumer. The generated suffix/output helpers accept an
accessor as well as the dense host-fixture representation. An isolated library
recompiles the two affected native translation units, device-links and relinks
against unchanged baseline objects without modifying the baseline build.

Slurm 10172 on n1 / RTX 5090 preserves assigned visibility `3`. Both policies
use the same isolated library, geometry, FP64 threshold `1e-12`, and original
accepted native density (physical residual RMS `1.4893289488046362e-12`).
Density SHA256 is
`a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`;
basis-file SHA256 is
`fb2c5d3a5f1ec149f0bf1d15d1e76cdb8baf042aeaa8be677cf25f33b1b7882b`.
Library SHA256 is
`bb089a0e07e95cd0b6ea65f0a56c6f08e394dcc508de1ac9531666bf5fc6cedd`.
Preparation and warmup are excluded; four samples per policy are interleaved
`0,1,1,0,0,1,1,0`. Each sample measures complete resident J, not only residual J.

| Schedule | Median complete J, CUDA events |
| --- | ---: |
| Incumbent #2600 (`0`) | 3.8710633545 s |
| Masked prototype (`1`) | 4.0808392334 s |

The prototype is **5.42% slower**; wall timing agrees. Maximum absolute error
against the independent frozen GPU4PySCF J oracle is `2.4448427294e-11`;
maximum policy-to-policy difference is `3.6948222260e-13`. Numerical gates pass.
Separate instrumented replays report residual time 2.4342 s versus 2.6363 s;
those diagnostic replays are not the uninstrumented timing samples.

| Class | Residual tasks | Masked tasks | Directed groups | Grouped summands |
| --- | ---: | ---: | ---: | ---: |
| `(2,1)` | 9,929,600 | 9,081,696 | 27,664,960 | 6,093,089,280 |
| `(2,2)` | 4,586,580 | 4,525,808 | 18,003,488 | 8,001,324,800 |

Admitted tasks and primitive roots agree between policies. Candidate probes
can differ because the existing tiled traversal has a dynamic early-exit
boundary; do not mistake candidate probes for retained scientific work. High
masked admission and fewer grouped radial terms do not establish profitability:
sorting, suffix setup, sparse transforms, scratch and publication still cost
real GPU work. These data reject promoting this prototype, not the algebraic
possibility of a better masked-Hermite lowering.

The raw matrices, samples and counters remain on n1 under `j96-10172/` in the
qualification directory below. Local receipts are
`build/masked-hermite/n1/j96-result.json`, `diagnostic-{0,1}.log`,
`pr2600-build-commands.json` and `pr2600-library.sha256`.

Slurm 10201 separately runs the source-matched #2600 normal-backend suite
against the compact-layout library with each masked policy: **114 cases pass
per policy**. This includes independent libcint screened-J gates for Cartesian
and spherical bases, restricted and nonsymmetric unrestricted densities, and
the tiled/high-reciprocal policy combinations. An earlier run of the old-base
test had 27 passes and one stale candidate-count assertion: #2600 conserves
`residual_candidates + residual_screened_candidates`, not each term separately.
The source-matched suite preserves that check; no scientific work-count gate
is removed to obtain the passing result. Retain `normal-qualification-old-test.log`
and `normal-qualification-pr2600.log`.

After the native transform-accessor adaptation, local focused/adjacent tests
pass 752 cases with 92 opt-in cases skipped; compiler structure checks still
report zero dependency errors and focused Ruff checks pass. The launch-error
host double covers both masked policies and propagates every injected reset or
launch failure. A separate overbroad local run could not load a native library
for 42 unrelated Fock tests; its receipt is retained rather than reported as
a passing full-suite run.

### Complete endpoint packaging preflight

A fresh process must find source-matched packaged stationary force artifacts
beside the isolated native library. Slurm 10178 completed cold SCF but returned
status 8 and no forces because these artifacts were absent. Slurm 10179 found
the original artifacts but correctly rejected their stale compiler-closure
identity: adding a module under `integral/` changes the stationary AOT contract
even when the force equations are unchanged. Neither run is a complete endpoint
or a valid timing/accuracy result. Retain both failure records.

Do not edit the manifest or bypass its identity gate. Regenerate and rebuild
the two r2SCAN RKS stationary AOT targets with the actual candidate compiler,
verified compiler cache and original FP64/no-FMA architecture contract. Check
the manifest's compiler-closure identity before starting another 96-atom SCF.

### Completed 96-atom cold endpoint

After rebuilding the packaged force artifacts with sccache 0.16.0, Slurm 10199
runs both policies sequentially in fresh processes on the same assigned n1 /
RTX 5090 GPU (visibility `0`). Both use the same candidate native library,
compiler source closure, stationary force manifest, basis and independent
reference. No preliminary or frozen density seeds the cold SCF. Existing
compilation caches are retained. Profiling and work counters are disabled.

Each timed endpoint includes calculator construction, preparation, converged
exact-direct FP64 SCF and full analytic forces returned as host arrays,
including the method-defined D4 and gCP corrections. Process/import startup,
provenance collection and owner teardown are outside the timer, as in the
original README cold definition. The grid is 48 radial by 16 polar by 32 azimuth,
unpruned, with three Becke iterations. SCF thresholds remain energy `1e-12`,
density `1e-10`, Schwarz `1e-12`, with at most 100 iterations.

| Policy | Prepare | Execute E+F | Complete cold | Iterations / Fock builds |
| --- | ---: | ---: | ---: | ---: |
| Incumbent `0` | 8.63735 s | 254.24104 s | 262.87839 s | 24 / 24 |
| Masked `1` | 8.69769 s | 268.71358 s | 277.41127 s | 25 / 25 |

This **single matched cold pair is 5.53% slower**, not an estimate of a timing
distribution. The extra Fock build is part of observable endpoint work: the
algebraic reordering changes the SCF trajectory even though final accuracy
passes. The separate same-density J experiment establishes a regression
without relying on that trajectory difference. Warm/moved endpoints have not
been measured for this prototype; do not present the cold pair as that larger
qualification matrix.

Both cold endpoints pass independent energy (`1e-8` Ha), force max-component
(`1e-7` Ha/Bohr), convergence, host-force shape/finiteness and physical residual
RMS (`1e-10`) gates. Maximum energy error across policies is
`2.3192114895e-11` Ha; maximum force-component error is `6.8252957756e-11`
Ha/Bohr. Final physical residual RMS values are `1.4894235325e-12` and
`1.4890267721e-12`. Policy-to-policy energy difference is `9.0949470177e-13`
Ha, and maximum force-component difference is `2.7625901566e-11` Ha/Bohr.
Both report `warm_start_used=False`. Full force arrays and physical histories
are retained in remote `cold96-10199/mode-{0,1}.json` and local
`build/masked-hermite/n1/cold96-mode-{0,1}.json`; the launch receipt is
`cold96-final-launch.log`. Compiler commands, cache statistics and stationary
artifact hashes are retained alongside these receipts.

The user's promotion condition is not met. Keep the flag **off by default**;
do not open a default-promotion PR based on either the algebraic work estimate
or the correctness gates. No branch, commit or PR is created for this prototype.

The first coarse-mask-only 30-case sanitizer run (10162) missed a real fallback
scratch-lifetime hazard. Expanded masks exposed racecheck warnings in 10163:
after reading the shared admission flag, some lanes could reuse the overlapping
incumbent scratch before every lane had completed that read. The decision must
be copied into lane-local storage and followed by a warp barrier before either
returning to the incumbent or consuming the masked scratch. Do not remove this
barrier on the assumption that identical predicates imply a lifetime fence.
The failed expanded run remains in `final-qualification.log`; the corrected
60-case evidence is `fixed-qualification.log`. Numeric tests alone had passed
even with that race, so retain complex-mask racecheck coverage.

Raw logs and compiler-cache statistics are retained in ignored local
`build/masked-hermite/` and on n1 in
`/home/jzzeng/generativeqc-qualification/masked-hermite-20261018-49fc/`.
No GitHub or release publication is involved.

## Revisit when

Before promoting or expanding the schedule, collect source-matched molecular
libcint J parity, energy/force acceptance, SCF trajectories, same-density
per-class timing/work and complete endpoint timing on the same GPU. Retain the
existing threshold/density screening and the resource/work fallback. A bitmask
or interval-DAG lowering can replace this sorted small-array schedule only if
it preserves the directed domain and passes those gates. The current prototype's
96-atom numerical gates pass but complete-J and cold-endpoint performance
regress; a revised lowering must demonstrate a measured benefit before further
promotion work.

### Follow-up diagnostic evidence

The upstream dataflow audit and the later sudo-enabled hardware-counter
collection are recorded in
`../../proposed/2026-10-18-residual-j-tiled-density-first.md`. Slurm 10221 removes
the earlier counter-permission limitation without changing driver policy.
That proposal does not implement a replacement kernel or change this
prototype's no-promotion decision.

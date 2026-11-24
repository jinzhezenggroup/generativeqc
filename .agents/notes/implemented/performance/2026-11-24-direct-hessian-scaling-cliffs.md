# Decision: Keep mid-size Direct CUDA Hessians on bounded batched routes

Status: implemented
Date: 2026-11-24

## Problem

The water tetramer's 12-atom Direct FP64 Hessian exposed three cutoffs not hit
by the trimer in #3134. The 160-rotation CPKS operator exceeded the 128 KiB
resident cap, inducing 1,006 physical Fock applications for 36 RHSs. The
one-electron plan exceeded both the 4,096-task dense limit and the useful
capacity of per-task full AO panels. Finally, the grid no longer qualified for
the whole-grid shared-memory Becke kernel. In the Slurm 2433 PBE trace, scalar
nuclear attraction kernels alone took 14.3 s of device time, ordered one-electron
projection took 2.1 s, and partition geometry took 6.3 s over 14,663 launches.
The resident Cartesian AO adjoint was already active.

## Decision

- Allow up to 512 KiB for the optional dense CPKS operator. Above the previous
  ceiling, require no more than six operator-build columns per nuclear RHS and
  an iteration budget exceeding that ratio. This bounds the added upfront work;
  it does not predict convergence. Keep the response and aggregate arena caps,
  the physical residual checks, and the uncached route on refusal.
- Store only the actual shell-pair value/motion block for each parallel
  one-electron tile. The projection already writes every local pair; no
  task-level AO-sized zero fill is needed. The sparse owner lists and the
  ordered scan both fold the same contributions in the original order. With
  that storage reduction, a dense owner can admit up to 16,384 tasks, under the
  same 64 MiB optional-workspace and aggregate caps. Other owners keep their
  task limits. Capture, integral batching, and scalar fallback share the same
  integral and projection algebra.
- Select whole-grid Becke tiles up to 16,384 points and 40 KiB of shared memory
  per block at the default point tile. Keep the existing global-scratch route
  for larger geometries and explicit tiles; do not change partition algebra,
  atomic weights, or the admitted point-pair work.

## Evidence and limits

Local artifacts are in `build/1905-gpu4pyscf/scaling/` (timings, raw matrices,
compiler-cache receipts, hashes, and CUDA trace summaries). Baseline is the
#3134 library with production sources equivalent to `05778122e`. Runs use
node5's RTX 5090, driver 570.86.16, Cartesian STO-3G, FP64, no density fitting,
and an independent GPU4PySCF 1.9.0 analytic moving-grid Hessian on the same
12×6×12 quadrature. B3LYP uses the matched Gaussian/VWN-RPA convention
(`B3LYPG` on the reference side). The 12-atom geometry uses the water dimer
coordinates from `tests/python/test_semilocal_cartesian_hessian_cuda.py` and a
copy shifted by `(1.4, 6.8, -0.6)` Bohr. Timing separates the complete Hessian
after SCF from the full public call including fresh SCF.

Warm timings in seconds (each pair is Hessian / complete endpoint):

| Case | Baseline | Candidate | GPU4PySCF |
| --- | ---: | ---: | ---: |
| 12 atoms, PBE | 32.33 / 32.43 | 5.59 / 5.69 | 2.04 / 3.14 |
| 12 atoms, B3LYP | 32.77 / 32.87 | 5.61 / 5.71 | 2.55 / 4.87 |
| 9 atoms, B3LYP | 1.53 / 1.77 | 1.53 / 1.75 | 2.03 / 3.83 |
| 6 atoms, r²SCAN | 0.50 / 0.66 | 0.49 / 0.66 | 1.19 / 2.79 |

The 12-atom comparison has two warm candidate/reference repeats after a
cold call in Slurm 2435, plus a final-build replay per case in Slurm 2439;
the smaller r²SCAN case is a single warm check per build. At
12 atoms, the standalone CPKS extension reduced physical Fock actions to
232, but warmed PBE still took about 30 s. With all three changes, the 36
directions, 67,541,364 four-center primitive components, 3,556,224 one-electron
primitive components, and 1,296 mixed grid contractions were unchanged. The
candidate ran 36 independent physical residual checks and reported 191,283,444
response workspace bytes versus 178,977,012 at baseline. At 9 atoms the
compact schedule reduced workspace from 195,296,916 to 161,003,156 bytes.
The up-to-12-atom raw matrices agreed within `2e-10 Ha/Bohr²` with baseline and
`3.8e-8 Ha/Bohr²` with GPU4PySCF (gates `2e-8`
and `5e-8`, respectively). Neither precision nor Hessian symmetry treatment
changed.

The final PBE trace reduced total CUDA kernel launches from 872,499 to 70,236
and Becke geometry device time from 6.29 to 0.52 s. The Direct factorized ERI
kernel was then the largest device cost (1.71 s), followed by batched ERI
reduction (0.52 s) and Becke partitioning. Focus the next scaling change on
Direct derivative sources rather than increasing the CPKS cache again.

At 15 atoms (one more water from a dimer shifted by `(-0.5, 13.7, 0.3)` Bohr),
the preliminary 8,192-task/32-KiB build fell back and needed 74.79 / 74.92 s.
The final 16,384-task/40-KiB build needed 14.86 / 14.99 s (two warm runs),
versus 2.74 / 5.03 s for GPU4PySCF (one warm run). Work remained at 45
directions, 168,120,765 four-center and 8,434,125 one-electron primitive
components, 340 induced Fock actions, and 2,025 grid contractions.
Workspace rose from 261,701,328 to 284,731,600 bytes. The new schedule agrees
with the preliminary raw matrix within `3.1e-14 Ha/Bohr²`; both differ from
the analytic comparator by up to `8.1e-8 Ha/Bohr²`, beyond the `5e-8` smaller
cohort gate. This is not a strict cross-engine qualification for 15 atoms.

The 18-atom PBE probe still refuses the Direct canonical source-group work
bound (108,345 shell-quartet sources, limit 100,000). No cap was lifted to make
that case pass. Supporting it competitively requires reducing or restructuring
the exact nuclear derivative work, not increasing the source limit in isolation.
These measurements do not cover the separate 96-atom r²SCAN energy/force case.

## Follow-up: amortized Direct source windows

The dense Hessian already publishes only the contracted frozen Fock and HVP
columns. If the exact sources cannot all be retained, visit each admitted
window once and accumulate into that same private column table. Every column
still visits windows and reduction lanes in its original order. No extra
device arena, component evaluations, post-hoc symmetrization or primitive
screening is introduced. An already-resident schedule keeps its column sweep;
the checked arbitrary-direction action does not use the new traversal. In
the read-only layout, measure the record span up to the first batch descriptor,
not up to raw scratch (which also includes metadata charged separately).
Increasing the optional tail cap did not help at the public 64 MiB integral
budget and was discarded.

The follow-up also keeps structurally absent exchange incidences out of the
Coulomb-only reduction lists, forms the common Becke center-pair denominator
jet once per dispatch in the normal exponent range, and uses the same
compiler AD product with sparse axis dependencies and bounded shell degree
for CUDA full-range s/p ERIs. All accumulation remains FP64. The general
quotient remains available for extreme or non-representable reciprocals.

Same-card RTX 5090 warm timings in seconds, from Slurm 2444, 2447, 2449 and
2451; the candidate reports Hessian-only / complete endpoint with fresh SCF:

| Case | Before follow-up | Candidate | GPU4PySCF |
| --- | ---: | ---: | ---: |
| 12 atoms, PBE | 5.59 / 5.69 | 4.92 / 5.02 | 2.04 / 3.14 |
| 12 atoms, r²SCAN | 5.63 / 5.78 | 4.96 / 5.11 | 2.27 / 3.46 |
| 15 atoms, PBE | 14.86 / 14.99 | 12.99 / 13.12 | 2.73 / 5.06 |
| 15 atoms, B3LYP | 14.85 / 14.97 | 13.09 / 13.20 | 3.34 / 5.94 |
| 15 atoms, r²SCAN | 14.91 / 15.09 | 13.09 / 13.28 | 3.00 / 5.14 |
| 18 atoms, PBE | Not admitted | Not admitted | 3.71 / 12.22 |

The 15-atom PBE directional work still consists of 45 columns, 168,120,765
exact primitive components, 8,434,125 one-electron primitive components,
340 induced Fock actions and 2,025 mixed grid contractions. Workspace is
284,327,888 bytes; this is the response owner's accounting, not a process
peak. The window traversal changes the 15-atom PBE raw matrix by at most
`2.1e-14 Ha/Bohr²` relative to the immediately preceding column-major build.
The budget-fallback test also compares every frozen Fock/HVP column directly
against the old traversal. Slurm 2450 passed 97 tests (2 skipped); 29 host
codegen tests and compiler structure checks passed.

The complete 15-atom PBE trace shows host-to-device copies dropping from
8,509 MB / 24,798 copies to 197 MB / 11,070 copies, while device-to-host
traffic is unchanged. The remaining largest CUDA costs are the factorized
four-center kernels (about 3.4 s across shell degrees), Becke partitioning
(2.28 s), and quartet reduction (1.46 s). These accumulated device times
are attribution, not additive wall-clock savings.

The 15-atom PBE cross-engine maximum remains `8.1e-8 Ha/Bohr²`. A sixth-order
finite difference of newly converged independent CPU PySCF analytic moving-grid
gradients (`fd15.py`, steps 0.01, 0.005 and 0.0025 Bohr) in the highest-error
column put the native column within `2.9e-8 Ha/Bohr²` of this oracle and
GPU4PySCF within `6.4e-8 Ha/Bohr²`. This diagnoses one column, not the full
Hessian; keep the original `5e-8 Ha/Bohr²` cross-engine gate. The remaining
speed gap needs less exact derivative and moving-grid work, not more memory
for the same work or a compiler-flag-only change.

## Follow-up: shared weight rows and relative ERI columns

On the private resident Cartesian route, build all mixed Becke weight rows
for one authenticated right direction together. Prefix/suffix products keep
zero factors valid without division. The rows reuse idle AO scratch; invalid
or extreme-exponent points use the scaled partition kernel, and the ordinary
arbitrary-direction route remains untouched. `action_partition_pairs` still
charges the conservative unbatched all-points bound per completed action;
`cartesian_partition_sweeps` counts actual batched right-direction launches,
not the number of points sent through the scaled fallback.

Visit only atoms incident to a canonical source window. If every source
shares an atom, omit that atom's three right columns and accumulate their
increments as the negative of the other columns for that window. This is
translation invariance of each ERI and its center gradient, not a projection
or symmetrization of the assembled Hessian. An arbitrary direction does not
use this omission. The 15-atom primitive-component count goes from
168,120,765 to 125,719,452, with 45 published directions, 340 induced Fock
actions, 2,025 mixed grid contractions and the same 284,327,888-byte workspace
accounting. The raw matrices differ from the source-major build by at most
`2.5e-14 Ha/Bohr²`. The dedicated grid and J/K probes passed; Slurm 2456
recorded 100 passes and two skips in the CUDA regression set.

Same RTX 5090, Slurm 2454; warm Hessian / complete endpoint seconds:

| Case | Relative-column + shared-weight build | GPU4PySCF 1.9.0 |
| --- | ---: | ---: |
| 12 atoms, PBE | 4.22 / 4.32 | 2.04 / 3.14 |
| 12 atoms, r²SCAN | 4.28 / 4.42 | 2.27 / 3.46 |
| 15 atoms, PBE | 10.06 / 10.18 | 2.73 / 5.06 |
| 15 atoms, B3LYP | 10.19 / 10.31 | 3.34 / 5.94 |
| 15 atoms, r²SCAN | 10.16 / 10.34 | 3.00 / 5.14 |

The whole 15-atom matrix was also compared with independent sixth-order
finite differences of newly converged CPU PySCF moving-grid gradients
(Slurm 1868–1870, `fd15_all.py`). Maximum absolute errors, in Ha/Bohr²:

| Functional | Compared steps (Bohr) | Change on halving | Native vs CPU FD | GPU4PySCF vs CPU FD | Native vs GPU4PySCF |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBE | 0.005, 0.0025 | 1.91e-9 | 3.01e-8 | 5.23e-8 | 8.04e-8 |
| B3LYP | 0.005, 0.0025 | 1.28e-9 | 2.85e-8 | 3.03e-9 | 3.04e-8 |
| r²SCAN | 0.0025, 0.00125 | 4.27e-9 | 3.68e-8 | 7.31e-9 | 3.82e-8 |

The 0.005 Bohr r²SCAN sample is not converged (about `2.1e-6 Ha/Bohr²`
from either analytic result) and is not the reference for acceptance. The
native matrices pass the independent `5e-8 Ha/Bohr²` numerical gate; PBE
still fails the separate analytic cross-engine gate. Neither result
establishes speed parity.

Discarded experiments: sharing primitive geometry across a four-component
CUDA ERI block regressed 15-atom PBE to 15.46 / 15.59 s (Slurm 2457), despite
removing repeated Boys/geometry evaluations. Precomputing the two contracted
AO strides in quartet reduction metadata did not materially change the
endpoint (10.07 / 10.20 s, Slurm 2458). The profile still attributes about
2.9 s to four-center primitive kernels and 1.22 s to four-center publication
reduction. Reassociating the fixed-grid AO contractions with cooperative
reduction only on the long point axis saved about 0.5 s but added 26.9 MB to
the 15-atom response workspace. Neither cooperative reductions on all axes
nor slot reuse without epilogue fusion improved the endpoint (Slurm 2459,
2460 and 2462). Keep the qualified original maps.

## Follow-up: compact Cartesian weight rows beyond 18 centers

Changing the basis composition separates atom-count and shell-count cliffs.
On the same card, an 18-atom hydrogen cluster completed in 3.25 / 3.34 s
warm for PBE (GPU4PySCF 2.09 / 2.60 s). A 24-atom hydrogen cluster fits the
canonical source bound but took 184.30 / 185.51 s cold, because the previous
18-center shared-memory layout forced every mixed weight row through the
scaled path. This is distinct from the 18-atom water case, which still
exceeds the four-center source admission bound.

Reuse one shared derivative plane for the three left axes and overwrite
exclusive prefixes with their final slopes. The static warp scratch fits
under 48 KiB for up to 30 centers; device arena admission is unchanged.
Retain the per-factor range guard and check the product of nonzero factors
before reusing slopes so larger atom counts cannot silently underflow the
normal-domain calculation. A rejected point still uses the scaled path.

Slurm 2461 and 2463: 24-atom PBE now takes 9.54 / 10.91 s cold and
9.33 / 9.40 s warm; raw old/new matrices differ by `1.28e-15 Ha/Bohr²`.
Both routes account for 297,640,692 device bytes, 72 directions, 32,910,219
four-center and 9,704,448 one-electron primitive components, 288 induced
Fock actions and 5,184 mixed grid contractions. The work eliminated is
repeated partitioning within each right-column sweep, not a change in
quadrature, primitive counts, precision, or SCF settings. The 24-atom
weight-row fixture checks against both the scaled route and displaced
independent long-double quadratures.

The first compact implementation still submitted 21 globally backed scaled
tiles per masked left row. The scaled kernel has lane-private scratch, so a
half warp fits larger blocks under the existing 40 KiB bound. Keep its one
full-grid pass when the default tile and a Cartesian acceptance mask permit
it; explicitly small tiles still use the bounded global fallback. This saved
another 0.17 s on 24-atom PBE (Slurm 2466); the GPU geometry suite also covers
an extreme measure that requires the scaled path at 24 centers (Slurm 2467).

The dense RHS geometry uses the same authenticated column on both sides.
Compute its diagonal weight row with the same dual/exclusive-product code
and scaled fallback instead of unconditionally replaying a full unmasked
scaled partition. The acceptance mask borrows the AO scratch only until
partitioning completes, before the normal AO panels are produced on that
stream. The later all-left sweep has a separate cache lifetime and counter.
Arbitrary directions keep the original path. The final build does not add
device arena bytes or change the primitive or response counts above.

Same-card 24-atom warm Hessian / complete endpoint seconds (Slurm 2463,
2468); errors compare the entire raw analytical matrices in Ha/Bohr²:

| Functional | Native | GPU4PySCF 1.9.0 | Max error |
| --- | ---: | ---: | ---: |
| PBE | 7.17 / 7.25 | 3.43 / 3.84 | 3.30e-9 |
| B3LYP | 7.24 / 7.31 | 3.72 / 4.23 | 3.13e-9 |
| r²SCAN | 7.34 / 7.46 | 3.65 / 4.26 | 3.36e-9 |

The 24-atom PBE fresh-process endpoint is 8.63 s versus 185.51 s on the
previous scaled route. Its original unsymmetrized FP64 matrix changes by
`1.39e-15 Ha/Bohr²`. The 24-atom comparison exposes a real grid-scaling win
but does not establish warm parity or qualify 18-atom water. Continue to
prioritize true derivative and response work rather than lifting work caps.

The 24-atom trace attributes only 0.083 s to the remaining scaled partition
kernels, down from 2.64 s in the first compact implementation; much of the
remaining time is distributed across grid response and Direct J/K. The
final 15-atom water matrices still pass the independent CPU gate (up to
`3.68e-8 Ha/Bohr²`), but warm endpoints remain 10.31–10.44 s (Slurm 2469).
Slurm 2470 reports 101 passed, two skipped and ten deselected in the CUDA
regression set, including independent 24-center weights and the diagonal
view's parity with the scaled path.

## Follow-up: contract frozen shell groups before global publication

For batched frozen unit directions, fold s/p shell quartets into their sparse
Fock/HVP owners in the projection block. The global reducer loads one scalar
per owner entry, rather than reading the uncontracted 41-field AO quartet
tensor. A pure-s window needs no shared-memory projection or block barrier.
The same generated J/K/AD maps, local summation order, deterministic FP64
publication and coincident-center accounting still apply. General angular
momenta and public arbitrary directions retain the existing projection;
admitted workspace and conservative work bounds are unchanged.

On node5's RTX 5090, Slurm 2473 measured these warm 15-water-atom timings
(Hessian / complete fresh-SCF endpoint in seconds) before the later AO cache
change:

| Functional | Packed projection | Previous complete | GPU4PySCF complete |
| --- | ---: | ---: | ---: |
| PBE | 9.15 / 9.28 | 10.31 | 5.06 |
| B3LYP | 9.26 / 9.37 | 10.37 | 5.94 |
| r²SCAN | 9.26 / 9.45 | 10.44 | 5.14 |

All three raw FP64 matrices changed by at most `2.36e-14 Ha/Bohr²` and passed
the independent CPU whole-matrix gradient-FD gate (`5e-8 Ha/Bohr²`, observed
maximum `3.68e-8`). Exact work remained 125,719,452 primitive components and
8,434,125 one-electron primitive components over 45 directions. The smaller
publication tensor is not a new primitive screening rule or work-cap override.

The private moving-grid adjoint also reuses its right-AO storage for the left
cotangent once each compiler-generated tile consumes that right tangent.
Output staging is bounded to one tile and may reuse the generic tensor arena.
The copy follows the adjoint on the same stream, and rebinding a direction
replaces the entire panel. A verified public right tangent is never repurposed;
it may still be consumed by the generic mixed evaluator. Enqueue failure during
the swap retires the right cache. This does not raise the cotangent budget.

The matched 28-hydrogen-atom Direct case has 24,192 points. Default 128 MiB
geometry admission leaves the AO jets partially resident and refuses the
dense replay budget for all three functionals (Slurm 2472). An explicit
192 MiB geometry budget makes room for its 177,242,880-byte workspace; this
does not establish performance parity and is not a default-cap proposal.
GPU4PySCF warm complete endpoints were 5.13 / 5.48 / 5.35 s for PBE /
B3LYP / r²SCAN. The 18-water-atom canonical-source refusal is separate.

With the adjoint reuse on 24 hydrogens (Slurm 2478), warm complete
endpoints were 6.87 / 6.95 / 7.06 seconds, and the PBE device receipt dropped
from 297,640,692 to 284,861,172 bytes. Primitive counts, 72 published columns,
288 induced Fock actions and 5,184 mixed grid contractions are unchanged.
Slurm 2477 passed 102 CUDA tests (2 skipped); Slurm 2480 passed all 12
semilocal grid probes, including comparison of the overwritten private AO
buffer's repeated mixed contractions with a freshly rebound checked direction.

The 28-hydrogen explicit-budget trace (Slurm 2479) showed an independent
one-electron cliff, not lost grid-adjoint admission. Five frozen columns
submitted 128,035 scalar tasks, including 120,195 attraction launches that
took 85 percent of the sampled GPU kernel time. Each column has 25,607 tasks,
above the 16,384-task dense limit. Allow up to 32,768 tasks inside the
same 64 MiB optional, integral and aggregate workspace caps, without relaxing
primitive-work checks or removing the scalar fallback.

The revised build completed all three 28-hydrogen FP64 Direct cases in Slurm
2482 with the explicit 192 MiB grid budget. On the same RTX 5090 (reference
Slurm 2472), results were:

| Functional | Native cold complete | Native warm complete | GPU4PySCF warm complete | Raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: | ---: |
| PBE | 14.12 s | 12.72 s | 5.13 s | 3.29e-9 |
| B3LYP | 14.28 s | 12.88 s | 5.48 s | 3.12e-9 |
| r²SCAN | 14.22 s | 12.88 s | 5.35 s | 3.35e-9 |

The earlier build at 192 MiB did not finish its first PBE result before
manual cancellation at about six minutes; do not quote it as a completed
timing. The revised PBE endpoint accounts for 383,011,868 device-workspace
bytes, 84 directions, 59,341,815 exact and 17,781,120 one-electron primitive
components, 364 induced Fock actions, and 7,056 mixed grid contractions.
Slurm 2483's full trace has 84 batched attraction launches taking 0.245 s
in total. The main remaining device costs include Becke rows (1.83 s),
four-center ERI kernels (1.05 s) and exact J/K reduction (0.73 s); the
remaining wall time also includes the many AO/semilocal response launches.
Fixing the scalar cliff does not establish parity. The final grid-reuse
build still passes the 15-water independent gate (`3.68e-8 Ha/Bohr²` maximum)
and reduces its PBE workspace from 284,327,888 to 274,400,208 bytes.

## Follow-up: avoid repeated grid products and unused potential matrices

The compiler fuses the point products from the existing density VJP/JVP, so
8–64 AO grids with at least 4,096 points can reduce each canonical AO pair once
in FP64, then publish both entries. The exact generated contraction is still
used for other shapes. The private stationary mixed contraction needs the
perturbed semilocal grid covector, but not the AO potential matrix built from
it. Skip that matrix only after the response solve; the RHS and public
potential path still build it, and the full nonlocal path is unchanged.

The Cartesian diagonal partition uses the same generated jet arithmetic with
identical left and right motion; it need not construct the all-left slope
planes. For all-left rows, construct each unordered pair's complementary
switches once, then fold the same ordered exclusive products. A cyclic
assignment balances odd and even atom counts. Both changes preserve the
accepted-point guards and scaled fallback and add no device arena bytes.

On the same node5 RTX 5090, Slurm 2489 and 2492 measured the median of two
warm complete endpoints (including fresh SCF):

| Case | Final native | GPU4PySCF 1.9.0 | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| 24 H, PBE | 5.01 s | 3.82 s | 3.29e-9 |
| 28 H, PBE | 9.36 s | 5.14 s | 3.29e-9 |
| 28 H, B3LYP | 9.51 s | 5.48 s | 3.12e-9 |
| 28 H, r²SCAN | 9.51 s | 5.27 s | 3.35e-9 |

All 28-atom runs still require the explicit 192 MiB grid budget. The PBE
endpoint still accounts for the same 383,011,868 device-workspace bytes and
work counts listed above; the B3LYP workspace is 383,138,588 bytes. The
largest raw asymmetry is `2.32e-14 Ha/Bohr²`; the paired switch changes a
raw entry by at most `2.34e-15 Ha/Bohr²` from the preceding build. Slurm 2487
passed 18 directed CUDA checks, 2488 passed 102 CUDA regressions (2 skipped),
and 2490 passed 45 geometry tests (4 skipped) on the final kernel, including
even and odd atom counts against displaced independent quadratures.

The final 15-water-atom endpoints (Slurm 2494; one warm sample per method)
are 7.55 / 7.63 / 7.71 s for PBE / B3LYP / r²SCAN. All three preserve
45 directions, 125,719,452 exact primitive components, 8,434,125 one-electron
primitive components and 2,025 mixed grid contractions. The PBE device
arena is 274,400,208 bytes. The independent sixth-order CPU gradient-FD
whole-matrix errors are `3.01e-8`, `2.84e-8`, and `3.68e-8 Ha/Bohr²`, all
within `5e-8`; raw asymmetry is at most `3.78e-12 Ha/Bohr²`. The PBE
GPU4PySCF disagreement (`8.03e-8 Ha/Bohr²`) remains above that gate and is
not evidence of passing the GPU reference. The 18-water-atom PBE endpoint
still refuses at the canonical J/K source bound in the same job.

The 28-atom trace before pairing switches measured 1.09 s in all-left Becke
and 0.33 s in its standalone diagonal, 1.04 s in factorized four-center
integrals and 0.72 s in their ordered reduction. It contained 165,288 kernel
launches. The usual build is `-O3`; optimization needs fewer visits and
launches, not a new release compiler flag. The fast path still trails the
reference. A default-budget PBE admission probe on node4 (Slurm 1872) rejects
32 H at the canonical J/K source bound and 96 H at XC grid workspace
admission. These are not timings and do not qualify either size. Do not lift
the caps in lieu of a work-bounded schedule.

## Rejected: serialize radial tiles to publish directly into the ERI cache

A prototype gave each component CTA all radial records and folded the
existing FP64 chunk sums into the field-major cache. This removed the raw
scratch handoff and one `accumulate_batches` launch per window/column without
changing the primitive set or the sum order. In Slurm 2496 the 28-H PBE trace
removed about 0.29 s of separate radial reduction, but the integral kernels
increased from about 1.04 to 1.45 s: sequential chunks reduced useful
parallelism. The same-job warm PBE endpoint regressed from 9.36 to 9.43 s.
28-H B3LYP and r²SCAN also regressed, and 15-water-atom PBE/B3LYP/r²SCAN rose
from 7.55/7.63/7.71 to 8.86/8.95/9.00 s (Slurm 2498). The full CPU reference
matrix gate still passed. The prototype was removed; a future direct cache
publisher must keep the radial chunks parallel, not merely remove a launch.

## Follow-up: pure-s ERIs and wider point-density contractions

Keep the pure-s four-center specialization independent of the discarded radial
publisher. On 28 H it alone saves about 0.17 s on the complete PBE endpoint.
For the XC density contraction, generate the per-point algebra from TensorIR
and distribute the canonical AO pairs over a warp. Use one density slot at
8–32 AOs and two at 33–64 AOs, only with at least 4,096 points. The wider
schedule borrows the existing tensor scratch for a transposed density; when
the capacity check fails, use the original point map. Neither optimization
adds device workspace or changes the admitted primitive counts.

Slurm 2503/2504 on node5's RTX 5090 measured complete warm Direct FP64
endpoints with a fresh SCF (one warm sample per row, except two for water PBE):

| Case | Native | GPU4PySCF 1.9.0 |
| --- | ---: | ---: |
| 24 H, PBE | 4.62 s | 3.82 s |
| 28 H, PBE / B3LYP / r²SCAN | 8.24 / 8.38 / 8.39 s | 5.14 / 5.48 / 5.27 s |
| 15 water atoms, PBE / B3LYP / r²SCAN | 6.95 / 7.04 / 7.09 s | 5.06 / 5.94 / 5.14 s |

The 28-H matrix differs from GPU4PySCF by at most `3.36e-9 Ha/Bohr²`.
The water cluster passes the independent sixth-order CPU gradient-FD gate
(`3.68e-8` maximum versus `5e-8 Ha/Bohr²`), but PBE still differs from the GPU
reference by `8.03e-8`. Slurm 1876 passed 16 directed grid checks, including
8/36-AO Libxc comparisons, and Slurm 2505 passed 215 tests (22 skipped) on
the same library. No larger-system admission or precision claim follows from
this. The retained workspaces, mixed contractions and exact primitive work
are unchanged. The trace still spends about 0.92 s on four-center integrals,
0.69 s on their reductions, 0.87 s on all-left Becke sweeps and 0.59 s on the
generated AO adjoint mixed map, with 7,056 separate Cartesian contractions.

## Follow-up: submit a full Cartesian moving-grid column

The resident dense endpoint already builds every left Becke weight row for
the current right unit direction. Submit the scaled masked fallback and
the AO-adjoint contraction in row-indexed grids, then reduce directly into
the private Hessian column. Validate the one explicit right seed; left unit
seeds are implicit in the row indices. Rejected points still run through
the scaled four-jet products and their finite/partition checks. The private
transaction still gates publication of the column and records all left-row
contractions, whether scheduled together or separately.

This schedule borrows the unused tail of the geometry AO panel for the
scaled fallback pair table and the now-idle adjoint builder scratch for
per-row partials. Require full AO-jet residency, the standard block-local
fallback tile and space in both arenas. If either admission fails, keep
the existing per-row path. No new workspace, accuracy relaxation, source
screening or work-cap changes. Slurm 2506 passed the five directed geometry
tests, including a mixed fast/scaled batch and a mismatched right unit seed.

Slurm 2506 ran all three arms on the same RTX 5090. Times are the median
of two warm complete Direct FP64 endpoints with fresh SCFs (seconds):

| System | Functional | Previous | Batched | GPU4PySCF 1.9.0 |
| --- | --- | ---: | ---: | ---: |
| 24 H | PBE | 4.631 | 4.408 | 3.850 |
| 28 H | PBE | 8.228 | 7.852 | 5.092 |
| 28 H | B3LYP | 8.369 | 8.010 | 5.496 |
| 28 H | r²SCAN | 8.391 | 8.020 | 5.292 |
| 15 water atoms | PBE | 6.965 | 6.899 | 5.016 |
| 15 water atoms | B3LYP | 7.056 | 6.985 | 5.791 |
| 15 water atoms | r²SCAN | 7.094 | 7.026 | 5.128 |

Per fixture, the primitive components, induced Fock actions, mixed-grid
contractions and device bytes in `cart-column-matrix-and-timing.log` are
unchanged. For example, 28 H PBE still has 84 directions, 364 induced Fock
actions, 59,341,815 exact and 17,781,120 one-electron primitive components,
7,056 mixed contractions and 383,011,868 bytes. Across these cases the largest
whole-matrix change is `2.09e-14 Ha/Bohr²`. The 28-H GPU reference error is at
most `3.36e-9`; the water-cluster error against independent sixth-order CPU
gradient differences is at most `3.68e-8` versus the `5e-8` gate. Water PBE
still differs from the GPU reference by `8.03e-8`. The smaller endpoint gain
on water and persistent 28-H gap rule out launch batching alone as a parity
solution. Keep the explicit 192 MiB requirement for 28 H and do not treat
32 H, 18 water atoms, or 96 H as completed Hessian timings.

Slurm 2507 passed 215 regressions with 22 skips on the batched library.
Slurm 2508 confirmed the 7,056 semantic contractions were dispatched in 84
kernels, taking 0.203 s total versus 0.394 s for the earlier traced launches.
The larger remaining costs are exact four-center evaluation (0.923 s) plus
its reductions and accumulation (0.976 s combined), Becke all-left and
diagonal rows (0.868 + 0.329 s), and semilocal point evaluation (0.488 s).
An optimization for larger systems needs to reduce exact J/K work or its
data movement, not simply increase the allowed number of canonical sources.

## Follow-up: publish two radial chunks concurrently

A full-range s/p window with 64-record tiles and at most 128 radial records
per source can place both chunks in one CTA. The primitive lanes work in
parallel, then two warps fold their original chunks separately in FP64.
The first warp merges those chunk sums in the original order and writes the
existing field-major resident cache. The compiler still owns the primitive
algebra; no extra screening, precision option, or public batch ABI entered
the path. No intermediate global chunk panel is needed. Other angular or
range kernels, longer radial lists and nonstandard tiles retain the previous
batched/ordered schedules and bounds.

Slurm 2509 ran both binaries on the same RTX 5090. The numbers are warm
complete Direct FP64 endpoints with fresh SCF, the median of two warm PBE
runs and one warm sample for the other methods:

| System | PBE, old → new | B3LYP, old → new | r²SCAN, old → new |
| --- | ---: | ---: | ---: |
| 28 H | 7.861 → 7.692 s | 8.039 → 7.866 s | 8.028 → 7.874 s |
| 15 water atoms | 6.886 → 6.834 s | 7.003 → 6.948 s | 7.045 → 6.994 s |

The 28-H case still performs 59,341,815 exact primitive evaluations and
7,056 mixed grid contractions; the 15-water case still performs 125,719,452
and 2,025 respectively. They use 98,304 and 116,480 fewer arena bytes. The
whole-matrix delta from the previous build is at most `2.09e-14 Ha/Bohr²`;
the independent 15-water gradient-FD error remains at most `3.68e-8`, and the
28-H GPU4PySCF error at most `3.36e-9`. The water PBE cross-engine discrepancy
(`8.03e-8`) still exceeds `5e-8`. Slurm 2509 also passed five focused
J/K probes including signed contractions, s/p and pure-s, and the 256-record
fallback, against independent libcint integrals. Slurm 2511 passed 218
regressions with 22 skips on the same library.

Slurm 2510 traced 1.051 s for the paired integral kernel and no raw-chunk
accumulator, versus 0.923 s plus 0.290 s before pairing. Contrast the rejected
serial producer: this keeps both chunks active while retaining the same fold.
The packed projection's downstream reduction remains at 0.697 s and the
Becke all-left sweep at 0.871 s. The observed win does not establish parity
or lift any large-system refusal. The next J/K experiment should avoid
scanning ordered group metadata for every inactive frozen atom.

## Follow-up: share active owner lists across the three nuclear axes

In the source-major frozen sweep, compact each output owner's ordered
projection addresses once per window and moving atom. The test is exact
center incidence: omit only quartets whose centers are all stationary or
all translate together. The three axes share one bounded index arena and
reduce only the compact addresses in the current projection interval. Each
owner retains its own deterministic FP64 warp fold; deleting zero entries
can change lane assignments, not the contributing quartet set. Translation
anchor increments are still applied directly. If the index arena does not
fit, or all source windows are retained for direction-major evaluation,
keep the old reducer and its original workspace admission.

Slurm 2513 ran paired full Direct FP64 endpoints with fresh SCF on the
same RTX 5090 (two warm samples for PBE, one for B3LYP and r²SCAN):

| System | PBE, old → new | B3LYP, old → new | r²SCAN, old → new |
| --- | ---: | ---: | ---: |
| 28 H | 7.701 → 7.249 s | 7.881 → 7.411 s | 7.875 → 7.425 s |
| 15 water atoms | 6.844 → 6.610 s | 6.957 → 6.706 s | 7.008 → 6.779 s |

The 28-H PBE schedule visits 81 source windows and 1,790 window/atom
pairs. Enumerating their ordered center orbits and Fock/vector incidences
gives 549,078,615 old metadata visits; the compact route makes 183,026,205
metadata visits and 74,972,604 active packed-value visits. On the associated
Slurm 2514 trace, the previous 0.697 s reducer is replaced by 0.194 s of
compaction and 0.026 s of reduction. Exact primitive work, one-electron
work, induced Fock actions, and grid contractions are unchanged. The
reused index arena adds 468,224–788,224 bytes across the six endpoints.

The independent 15-water gradient-FD error remains at most `3.68e-8
Ha/Bohr²`, the 28-H GPU4PySCF error at most `3.36e-9`, and the maximum
whole-matrix change from the prior native build is `1.43e-14`. Water PBE
still differs from GPU4PySCF by `8.03e-8`. Slurm 2513 passed the five direct
J/K checks, including the retained-source budget and 256-record fallback.
Slurm 2515 passed 218 regressions with 22 skips on the final library.
The dominant 28-H costs now include paired integrals (1.052 s), all-left
Becke (0.871 s), and point semilocal evaluation (~0.49 s). Parity and the
larger-system admission gates are still unmet.

## Follow-up: reuse the full Becke column for the CPKS right direction

For an admitted Cartesian column, prepare all left/mixed weight rows before
CPKS and retain them until the stationary contraction. The right weight is
independent of the left direction and is published only by the diagonal row,
including for points rejected by the fast sweep and repaired by the scaled
fallback. With two planes instead of three in every row, the existing
directional AO scratch can hold the column and still supply the first AO
direction. Release the acceptance mask and fallback center coefficients on
the same stream before producing those AO jets in their space. Neither
preparation nor CPKS needs another device arena. The whole column, AO jets
and owner motion stay private behind the same sticky transaction audit.

For each column with `K` nuclear coordinates and `P` quadrature points, this
eliminates a diagonal Becke sweep and reduces stored weight derivatives from
`(3K + 3)P` to `(2K + 1)P`. Directional AO output goes from 12 to 4 planes
and owner motion from 6 to 3. All mixed contractions, induced Fock actions,
and integral components are still performed. If resident jets or the scratch
for the scaled column do not fit, use the diagonal/right and rowwise paths;
an independent VV10 grid retains its existing path.

Slurm 2516 passed 45 geometry checks (4 skips), including compact-column
and diagonal agreement with independently scaled jets, large atomic
measures that force scaled fallback, and invalid Cartesian directions.
Same-GPU warm full endpoints, including fresh SCF (two warm PBE samples,
one for the other methods), changed as follows:

| System | PBE, old → new | B3LYP, old → new | r²SCAN, old → new |
| --- | ---: | ---: | ---: |
| 28 H | 7.229 → 6.868 s | 7.404 → 7.042 s | 7.428 → 7.050 s |
| 15 water atoms | 6.616 → 6.543 s | 6.714 → 6.641 s | 6.756 → 6.695 s |

There was no change in device arena size across these pairs. The 28-H
GPU4PySCF error remains at most `3.36e-9 Ha/Bohr²`; the 15-water
independent gradient-FD error remains at most `3.68e-8`. The largest
whole-matrix change from the previous native library is `1.98e-14`.
The previous water PBE discrepancy versus GPU4PySCF (`8.03e-8`) remains.
Slurm 2517 passed 218 regressions with 22 skips.

PBE0 shows the same pattern, not a functional-specific bottleneck: 28 H
goes from 7.375 to 7.021 s (GPU4PySCF 5.421 s), and 15 water atoms from
6.699 to 6.628 s (GPU4PySCF 5.545 s). Both native PBE0 matrices are
unchanged; their maximum errors versus the independent GPU4PySCF Hessian
are `3.04e-9` and `2.73e-8 Ha/Bohr²`. Each timing is one warm sample;
Slurm 2518 supplied the reference and 15-water pair after 2516's native
28-H pair. On the same candidate, 30 H and 18 water atoms (PBE, 192 MiB
grid budgets) still hit the exact canonical J/K source cap. The 96-H
r²SCAN Hessian still refuses at the XC/VV10 workspace check with 128 MiB.
These are admission results, not completed Hessian timings or r²SCAN
energy/force qualification. The next scaling work must reduce the
four-center derivative work and bound the grid panels; this optimization
does not justify raising either cap.

## Follow-up: factor the two moving-grid AO adjoints together

The private semilocal adjoint previously ran two independent TensorIR
transposes. The lowering now groups the emitted terms by AO source and
density orientation, and requires matching exact rational coefficients
before applying the sum of the two orientations. This is an algebraic
identity, not an assumption of symmetric or positive density. The same
four reference density-times-AO products serve both adjoints. Four right
AO products and four density-response products supply the remaining left
cotangent. Per point/output AO this replaces 80 oriented AO sums with
12 paired sums and 40 lane-local weighted terms, all in FP64.

Use the fused leaf for the existing private single-spin semilocal adjoint
when 8–64 AOs and at least 4,096 points are present. Finish all reads of
each right AO tile before overwriting it with the left cotangent. Keep the
original affine maps for other shapes and the bounded generic contraction
when the adjoint storage does not fit. The leaf itself needs no global
scratch; retain only the later Cartesian row reductions' scratch. No
changes to ordinary energy/force or independent VV10 workflows.

Slurm 2519 compared both libraries on node5's RTX 5090, using complete
Direct FP64 endpoints with fresh SCF. PBE uses two warm samples; the other
functionals use one each:

| System | PBE, old → new | B3LYP, old → new | r²SCAN, old → new | PBE0, old → new |
| --- | ---: | ---: | ---: | ---: |
| 28 H | 6.878 → 5.820 s | 7.039 → 5.986 s | 7.040 → 5.978 s | 7.017 → 5.951 s |
| 15 water atoms | 6.539 → 6.122 s | 6.637 → 6.219 s | 6.693 → 6.281 s | 6.643 → 6.232 s |

The original matrices change by at most `2.54e-14 Ha/Bohr²`; the 28-H
GPU4PySCF and 15-water independent gradient-FD maxima are still
`3.36e-9` and `3.68e-8`. PBE0 compares to GPU4PySCF at `3.04e-9` and
`2.73e-8` respectively. Water PBE's `8.03e-8` cross-engine discrepancy
remains. Device workspace drops 7,374,848 and 9,269,760 bytes; integral
components, physical Fock actions and mixed grid contractions remain
unchanged. Seven targeted tests passed in the same job, including the
8–36 AO grid probes' private adjoint/checked contraction comparison.
Slurm 2520 passed the full 218-test regression with 22 skips. Slurm 2521
traced 0.086 s for the fused adjoint leaf on 28-H PBE. The largest
remaining device costs are exact four-center integration (1.052 s),
the all-left Becke sweep (0.868 s), and the repeated fixed-reference
semilocal point evaluation (0.488 s). Speed parity and the larger molecule
admission gates are still open. Don't substitute the existing SCF Schwarz
mask for a nuclear derivative bound.

The follow-up merges the two coefficients of the same left AO jet inside
the generated leaf. Both AO inputs stay immutable, so the whole panel can
be built in one launch without a left tile or the per-tile device copy.
The older affine consumer is unchanged. Slurm 2522 passed the seven directed
tests and measured PBE `5.835 → 5.747 s` for 28 H and `6.139 → 6.139 s`
for the water cluster (two warm samples per library). The water cluster
saves another 4,587,520 bytes and the largest matrix change is `1.63e-14
Ha/Bohr²`. Thus eliminating the copy is a modest 28-H time improvement
and a workspace improvement for the water cluster, not a stand-alone
solution to the remaining gap.

Slurm 2523 passed 218 regressions with 22 skips on the merged leaf. Its
complete warm times for PBE / B3LYP / r²SCAN / PBE0 are
`5.747 / 5.919 / 5.915 / 5.870 s` (28 H) and
`6.139 / 6.235 / 6.280 / 6.232 s` (15 water atoms). Relative to the
shared-column build the whole-matrix change is at most `2.58e-14 Ha/Bohr²`.
The independent accuracy gates and the pre-existing water PBE comparator
discrepancy are unchanged. Slurm 2524 still refuses 28 H at 128 MiB (AO
jet replay work) and 96 H even at 1,024 MiB (XC workspace/work admission).
For 96 H, 82,944 grid points and 96 AOs, the generated unsplit mixed
work receipt is 69,000,569,856 terms per action, over the 2^34 limit before
byte allocation. The geometry also needs 159,252,480 spatial AO jets,
above its separate 134,217,728 limit, and the unfiltered exact schedule
would have 10,841,496 canonical sources. These are separate constraints,
not just a request for more memory. A scalable route must admit the actual
factored/tiled grid schedule and bound derivative-source work before
materializing dense Cartesian inputs or source windows.

## Follow-up: reuse stationary semilocal point response

At a fixed accepted density and quadrature, the point potential's derivative
is linear in the direction of the physical density/gradient/tau fields. For
restricted LDA/PBE (including scaled exchange), B3LYP and r²SCAN, use four
or five coordinate seeds of the existing point JVP, plus one value probe, to
form an optional FP64 point Jacobian. No point formula or ordinary SCF
energy/force path changes. Moving actions still add the independent measure
motion `dw * Vxc`, and supply `dot(Vxc, dfields)` to the weight cotangent.
Vacuum, extreme values and unrepresentable seeds or actions replay the existing
checked point evaluator. Unit-probe failures must not enter the enclosing
transaction status. Nuclear residency is behind mandatory arena and work
admission; fixed-grid residency is attempted after the enclosing Hessian has
admitted its other resources, including the resident Direct source.

For the bounded Cartesian Becke row kernel, reduce the atomic products across
one warp and reuse its normalized right dual. The left derivative in that
algebra is `(dN - (N/S)*dS)/S`, so each row needs only its left numerator and
denominator tangents. This preserves the scaled fallback, the reference-weight
gate, and the original local pair-product order. It also avoids recomputing
the right weight separately for every left row.

Slurm 2526 measured moving point residency alone and with Becke quotient
reuse; Slurm 2527 added fixed point residency. Times below are medians of
two warm full Direct FP64 calls with fresh SCF on the same node5 RTX 5090.
The final build is Release (`-O3`, sm_120); CUDA and C++ used the verified
`sccache` launcher (receipts alongside the raw matrices).

| System | XC | Prior grid build | Moving cache + Becke | Also cache fixed points |
| --- | --- | ---: | ---: | ---: |
| 28 H | PBE | 5.753 s | 5.643 s | 5.502 s |
| 28 H | B3LYP | 5.906 s | 5.717 s | 5.382 s |
| 28 H | r²SCAN | 5.909 s | 5.727 s | 5.341 s |
| 28 H | PBE0 | 5.862 s | 5.763 s | 5.601 s |
| 15 water atoms | PBE | 6.118 s | 6.116 s | 6.101 s |
| 15 water atoms | B3LYP | 6.222 s | 6.155 s | 6.031 s |
| 15 water atoms | r²SCAN | 6.273 s | 6.197 s | 5.990 s |
| 15 water atoms | PBE0 | 6.237 s | 6.234 s | 6.220 s |

The endpoint keeps the same published directions, induced Fock actions,
integral components, and mixed/first grid contractions. For 28 H these
include 84 directions, 364 induced actions, 59,341,815 four-center primitive
components, and 7,056 mixed contractions. For 15 water atoms the respective
counts are 45, 340, 125,719,452, and 2,025. The two optional point arenas
add 10,063,880–11,999,240 bytes (28 H) or 5,391,368–6,428,168 bytes (water).
Moving-only residency saves up to 3% on 28 H; combined fixed and moving
residency saves up to 10%. Becke quotient reuse has a small marginal effect
in this experiment; it is not the explanation for the main speedup.

The largest full-matrix change from the prior build is `2.62e-14 Ha/Bohr²`.
Against GPU4PySCF the 28-H max error is `3.35e-9 Ha/Bohr²`; for water the
independent gradient finite-difference comparison is within `3.68e-8`, and
PBE0 is within `2.73e-8` of GPU4PySCF. The earlier water-PBE cross-engine
difference (`8.03e-8`) is not resolved by this change. The directed grid
tests exercised signed directions, vacuum rows and nonzero weight motion,
checked the response against the incumbent action, and forced the optional
cache to fall back at the original arena size (Slurm 2526).
Both Slurm 2526 and the final build in 2527 passed 221 regressions, with
22 skips each.

The scale-up target is a different problem from reusing a point JVP: 18 water
atoms and 30 H need derivative-specific exact-source work reduction; 96 H
also needs factored, tile-bounded grid and AO-jet work. Ordinary SCF Schwarz
admission is not an error bound for the frozen derivative or its contraction
with the induced density. A screening proposal must bound first and second
derivative pair products and the sum of discarded weighted contributions,
and verify all Hessian components against the unscreened/independent route.

On node4's RTX 5090 (driver 580.178.04), GPU4PySCF Direct completed these
larger endpoints on the same Cartesian STO-3G and moving quadrature. Each
warm sample here starts with a fresh SCF, without density fitting:

| System | Functional | Full endpoint | Hessian after SCF |
| --- | --- | ---: | ---: |
| 18 water atoms | PBE | 5.084 s | 3.712 s |
| 18 water atoms | PBE0 | 7.943 s | 4.413 s |
| 30 H | B3LYP | 6.290 s | 5.668 s |
| 30 H | r²SCAN | 6.233 s | 5.583 s |
| 96 H | r²SCAN | 109.490 s | 107.416 s |

Slurm 1879 recorded the smaller four warm samples and a 112.192 s cold
96-H sample; Slurm 1880 recorded the comparable 96-H warm sample. These
are reference baselines, not speed ratios: the native large Hessians are
not admitted. Do not transfer those timings to the ordinary energy/force
or density-fitted endpoints.

Slurm 2528 repeated native admission on the final build: 18 water atoms
(PBE/PBE0) and 30 H (B3LYP/r²SCAN) at 256 MiB still hit the exact J/K
canonical source-group cap; 96 H (r²SCAN) at 1,024 MiB still hits XC
workspace/work admission. 28 H (PBE) at 128 MiB still hits AO replay work.
No bounds were loosened to obtain the fixed/moving cache speedups.

## Follow-up: CUDA full-range s/p radial moments

The node5 PBE trace (Slurm 2529) attributed 2.93 s inside the 15-atom
endpoint to four-center primitive kernels. The Release build already used
`-O3`; each full-Coulomb s/p primitive was evaluating a fixed 64-node radial
rule. The CUDA s/p specialization now consumes the compiler's bounded
full-range moment routine: positive series where upward recursion is poorly
conditioned, recurrence in the far field, and the fixed rule on refusal.
The split/LR kernels, other angular classes, and ordinary SCF/force paths
are unchanged.

Same node5 RTX 5090 and Cartesian STO-3G/moving quadrature; seconds are
medians of two warm Direct FP64 full calls, each starting a fresh SCF
(Slurm 2530/2531). Exact primitive components, Fock actions, first/mixed
grid work and device bytes match the 64-node arm for all ten cases.

| System | XC | Fixed rule | Bounded moments | GPU4PySCF | Native / GPU4PySCF, Hessian only |
| --- | --- | ---: | ---: | ---: | ---: |
| 12 water atoms | PBE | 2.663 | 2.234 | 3.084 | 2.130 / 2.036 |
| 15 water atoms | PBE | 6.105 | 4.967 | 5.029 | 4.838 / 2.732 |
| 15 water atoms | B3LYP | 6.026 | 4.885 | 6.131 | 4.768 / 3.346 |
| 15 water atoms | r²SCAN | 5.982 | 4.853 | 5.060 | 4.675 / 2.992 |
| 15 water atoms | PBE0 | 6.200 | 5.067 | 5.720 | 4.951 / 3.330 |
| 24 H | PBE | 3.168 | 2.886 | 3.859 | 2.822 / 3.456 |
| 28 H | PBE | 5.492 | 4.931 | 5.112 | 4.851 / 4.505 |
| 28 H | B3LYP | 5.389 | 4.815 | 5.482 | 4.731 / 4.884 |
| 28 H | r²SCAN | 5.352 | 4.780 | 5.266 | 4.639 / 4.642 |
| 28 H | PBE0 | 5.612 | 5.037 | 5.447 | 4.958 / 4.899 |

The full matrices change by no more than `2.74e-14 Ha/Bohr²` relative to
the fixed rule. The independent 15-atom CPU gradient differences for PBE,
B3LYP and r²SCAN pass `5e-8 Ha/Bohr²` (max `3.68e-8`); PBE0 differs from
GPU4PySCF by `2.73e-8`. For 28 H the maximum difference from the matched
GPU4PySCF reference is `3.36e-9`. The 15-atom PBE discrepancy between the
analytic engines persists (`8.03e-8`); the endpoint speed cannot be used
to waive that separate gate. Nor does it imply a faster Hessian stage:
on the 15-atom waters the native stage is still about 1.4–1.8 times slower.

Slurm 2530 passed 12 independent libcint value/gradient/gradient-difference
JVP cases and 61 native J/K tests (four skips). Slurm 1883 passed 160
geometry, endpoint and moving/fixed grid regressions (18 skips). The
earlier remote checkout failures in 1881/1882 were input resolution, not
numerics: the pinned libxc input is resolved as a checkout asset only when
the remote tree also includes `CMakeLists.txt`.

## Follow-up: traverse the two nonzero Cartesian AO strips

An AO only moves relative to a quadrature point when exactly one of its
center or the point's owner is displaced. For a whole Cartesian column,
store a small center-to-AO incidence table under the live basis lease.
Visit the other-center AOs on this atom's points, and this atom's AOs on
other points. Retain the full weight derivative contraction. The same
jets/cotangents are used; this is not a distance or magnitude cutoff.
The original masked scan is retained for a single row, other domains, and
when the optional incidence table cannot fit. The existing conservative
per-action work admission also remains in force.

For each right column of 28 H (24,192 points and 28 AOs), the number of
AO/point pairs visited by the left sweep falls from 56,899,584 to
3,991,680. The Fock actions, primitive components, and first/mixed grid
contraction counts are unchanged; only the traversal of the same grid
work changes. The packed owner table adds 1,536–3,328 device bytes.
In the Slurm 2532 A/B, 28-H endpoints improved by 2.7–3.0% (PBE/PBE0/r²SCAN)
and 12/15-atom waters by 0.9–1.1% (PBE/B3LYP). Max full-matrix change was
`2.47e-14 Ha/Bohr²`. The separate Slurm 1885 trace sees 4.0 ms total in the
sparse contraction for 15-atom PBE (45 right columns); four-center AD
kernels remain the largest group at about 1.79 s of device time.

The offline Gaussian derivative-envelope census estimates that for an
unweighted dropped budget of `1e-14`, 18 waters/30 H could retain about
87,953/96,844 source groups, and 96 H would still retain about 2.53 million.
These are feasibility estimates only, without outward FP64 bounds, native
contraction weights or solver amplification. They do not authorize skipping
a quartet or raising the production bound.

## Follow-up: pack one-electron task payloads within the existing arena

An isolated diagnostic binary (Slurm 1887) raised only the canonical J/K
source census to 109,000. It did not change the integrals or relax the other
limits. That allowed 30 H to complete a cold r²SCAN call in 244.78 s, with
the warm sample timing out. An initial 15 s trace of the same binary
(Slurm 1890) saw 100,601 scalar attraction launches taking 7.90 device
seconds; CPKS already used a 405,000-byte resident operator. The 28,800
one-electron tasks fit the task limit, but individually 256-byte-padding
every primitive, projection and publication buffer overflowed the caller's
64 MiB integral budget. Pack scalar-double payloads at their ABI alignment
inside the aligned arena. Preserve task order, the final ordered fold,
and the exact previous byte, task and scientific-work limits.

With this packing and the same diagnostic source census (Slurm 1893 on
node4 RTX 5090), 30-H r²SCAN completed in 7.34 s cold / 5.92 s warm, and
B3LYP in 7.43 / 6.10 s. Same-card GPU4PySCF warm full endpoints are 6.23 s
and 6.29 s; its Hessian-only times (5.58 and 5.67 s) remain below native
(5.76 and 6.00 s). One warm sample per arm is not enough to claim a robust
speed win. Native completes all 90 directions with 77,366,583 exact and
23,328,000 one-electron primitive components, 405 induced Fock actions and
8,100 mixed XC contractions. The r²SCAN matrix changes by at most
`4.39e-15 Ha/Bohr²` from the isolated scalar route; both functionals are
within `3.37e-9` of GPU4PySCF. The optional batched arena consumes about
37 MB more than the scalar fallback, without any increase in its allowed
size. Slurm 1892 passed 23 one-electron device regressions (four skips);
the host budget fixture checks admission with the unchanged byte limit.

The production-cap A/B on node5 (Slurm 2533, two warm complete calls per
arm) saves 6.33 MB for 15-water PBE, 9.29 MB for 24-H PBE and 14.58 MB
for 28-H r²SCAN. Their endpoints are 4.91, 2.80 and 4.59 s, versus 4.93,
2.80 and 4.62 s without packing. Semantic counts match, and full-matrix
parity remains within `1.69e-14 Ha/Bohr²`.

18 water atoms remain slower. With the relaxed diagnostic source count,
PBE takes 12.57 s warm vs 5.08 s for GPU4PySCF; payload packing alone
reduces memory by 10.68 MB but does not materially change this number.
A separate diagnostic admission of a 1,036,800-byte FP64 CPKS matrix
(Slurm 1888) reduces induced physical Fock actions from 1,582 to 468 and
PBE to 9.97 s warm, with `2.31e-14 Ha/Bohr²` CPKS parity. PBE0 then takes
10.16 s vs 7.94 s for GPU4PySCF. Independent matrix errors are at most
`3.06e-8` and `2.74e-8`, respectively. Do not ship the diagnostic cap(s):
the 18/30 production J/K source census is unchanged, and the 96-H XC and
AO-jet work bounds need a separate algorithmic change.

## Follow-up: derivative-specific admission and a single J/K plan

After the canonical source-limit refusal, a restricted, full-range Direct
dense sweep may try a certified **derivative-only** source plan. It reserves
its frozen table first. The planner uses the normalized Cartesian s/p shell
pair densities, bounds their first and second derivative Coulomb norms with
outward binary64 bounds, and skips complete eightfold orbits within an
aggregate unweighted derivative budget of `1e-14`. The source census,
workspace and primitive/project/reduction ceilings are unchanged. The bounded
certificate generator does not use the SCF mask, and an arena, topology or
other admission failure does not authorize the retry.

The private table audits zero frozen density response and `|D_μν| ≤ 4`;
its induced contribution also audits `|δD_μν| ≤ 16`. The budget bounds
omitted ERI derivatives **before** contraction, not the final Hessian or
CPKS amplification. The full matrices therefore also need an independent
comparison. Arbitrary directional actions, values, and incomplete frozen
tables are not admitted by this plan. Host libcint gradient/gradient-difference
and aggregate-budget checks passed (10 cases). The 1899 CUDA regression
compared the table against ordinary unscreened J/K, tested a short arena and
out-of-bounds density, and verified refusal of an ordinary action.

The 18-atom PBE trace (1896) still spends 3.24 s of device time in the
four-center primitive kernels. It also shows two roughly one-second
preparation gaps: the canonical schedule was built once to size the arena
and once more to bind it. Carry one owned, single-use host schedule across
the allocation instead. Binding still audits the live reference, workspace
and packed basis. In the paired radial kernel, elide empty runs and trim
inactive leading/trailing blocks without changing any destination cache
slot, primitive work count, or reduction order.
The 1902 trace confirms a single 1.05 s plan gap, and the four-center
kernel count falls from 37,710 to 25,764, with 3.15 s of device time.
Slurm 1901 passed 62 J/K and stationary semilocal regressions (nine skips).

With both changes and capacity-limited FP64 CPKS reuse (up to 1 MiB for the
eligible dense Direct RKS path), the latest node4 RTX 5090 measurements
(Slurm 1898, medians of two fresh-SCF warm runs) are:

| System / XC | Native full | GPU4PySCF full | Native / GPU4PySCF Hessian phase | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: | ---: |
| 18 water atoms / PBE | 7.755 s | 5.084 s | 7.604 / 3.712 s | 3.05e-8 |
| 18 water atoms / PBE0 | 7.839 s | 7.943 s | 7.711 / 4.413 s | 2.74e-8 |
| 30 H / r²SCAN | 5.222 s | 6.233 s | 5.070 / 5.583 s | 3.37e-9 |
| 30 H / B3LYP | 5.274 s | 6.290 s | 5.176 / 5.668 s | 3.14e-9 |

The GPU4PySCF arm has one warm sample per smaller case. PBE0's full-endpoint
near tie is not a robust speed win. 18 water atoms complete all 54 columns,
221,684,769 four-center and 17,146,080 one-electron primitive components,
468 physical Fock actions and 2,916 mixed XC contractions. 30 H completes all
90 columns, 69,394,968 four-center and 23,328,000 one-electron primitive
components, 405 physical Fock actions and 8,100 mixed XC contractions. The
reuse and run trimming leave those work counts unchanged against the prior
screened + CPKS arm (1895), and the largest matrix difference against it is
`2.09e-14 Ha/Bohr²`. Its warm endpoints were 8.75/8.86 s for 18-atom
PBE/PBE0, and 5.70/5.77 s for 30-H r²SCAN/B3LYP. Against the corresponding
unscreened CUDA diagnostics, all four full matrices agree within
`2.11e-14 Ha/Bohr²`.

Slurm 1898 still refused 32 H at the canonical J/K source limit for both
B3LYP and r²SCAN: the host census retains 121,080 of 139,656 source groups,
above the existing 100,000 limit. It retains 96,844 of 108,345 for 30 H.
On the same card, Slurm 1903's 32-H GPU4PySCF warm endpoints take 6.74 s
(r²SCAN) and 6.99 s (B3LYP), including 6.08 and 6.45 s after SCF.
96 H still refuses at XC workspace/work admission for r²SCAN. The
96-H Direct GPU4PySCF reference completed in 109.49 s. Removing host replan
and empty launches does not close that capacity gap or the 18-atom PBE
Hessian-phase gap. Revisit with a work-bounded streaming source schedule and
factored tiled moving-grid/AO actions, not by increasing the diagnostic
source or work limits. No claim is made for ordinary CUDA energy or forces.

## Follow-up: factored semilocal grid work for 65–96 AOs

Lower fixed and moving feature contractions into three 32-AO strips from
the same TensorIR as the two-strip route. For the potential and its moving
tangent, factor the point product by the left AO lane. Stage the unweighted
AO and the weighted right transform once per point tile, then fold each
canonical AO pair in FP64 and publish both symmetric entries. Tile scratch
reuses the bounded tensor arena; a too-small lease keeps the original affine
schedule when that schedule's work budget passes.

Slurm 1909 compared the original affine potential against the factored
potential with identical three-strip features, at 96 AOs, 4,096 points and
restricted r²SCAN. The complete standalone grid probe (including process and
fixture input; not a molecular Hessian) was 6.243 vs 6.014 s, median of three
interleaved samples per arm. Cumulative device kernel time fell from 164.17
to 22.85 ms in that probe, with 660 vs 448 launches and the same 55,500,824-byte
workspace. The moving stage reuses each AO transform across its partner AOs
instead of evaluating the expanded products for every pair. Paired numerical
results agree within `6.94e-18`. The 85-pass, four-skip Slurm 1910 grid suite
also checks signed directions against independent Libxc differences at 68 and
96 AOs, with 4,109 points to cover a short final tile.

When the incumbent affine plan cannot run, an eligible wide semilocal RKS
consumer can now admit a separate factored layout. Charge the generated
feature products, tiled potential work, and required read-only mixed AO
adjoints to the same work/byte ceilings. Do not admit affine replay under
the factored receipt. The 1911 tight-work tests check the required plan
against the unrestricted oracle and refuse sizing without the adjoint budget.
The 96-H r²SCAN Direct call in Slurm 1912 (1,024 MiB shared grid budget)
passed XC admission, then refused at **moving-grid workspace admission**.
It did not produce a Hessian or a timing comparable to GPU4PySCF. To make the
full endpoint fit, tile the AO/adjoint lifetime across the geometry and XC
owners rather than reserving their largest full-point panels concurrently.
32 H still needs a separate bounded J/K source schedule.

## Follow-up: mid-width potentials and sparse one-electron folds

Reuse the staged potential transform above 32 AOs (instead of above 64)
when the existing tensor lease has room for a tile. Preserve the direct
path for one AO strip or insufficient scratch; neither the workspace
ceiling nor the admission accounting changes. Slurm 1913 passed 89 grid
tests (four skips), including signed Libxc fixed/moving differences at
32, 36 and 44 AOs and partial last tiles. Across the 18-atom PBE/PBE0 and
30-H r²SCAN/B3LYP trials, the table counters and workspace allocation did
not change. The 18-atom PBE full warm samples move from 7.91–7.95 to
7.83–7.85 s, and the raw matrix changes by at most `1.51e-14 Ha/Bohr²`.
The PBE0 baseline varied substantially in the same run; do not attribute
the full PBE0 wall-time change to the new potential schedule. GPU4PySCF's
18-atom PBE warm SCF took both 31 and 268 cycles. Its Hessian-only timing
was consistently 3.95–3.96 s, so the native Hessian gap remains despite
the variance in full endpoint times.

The 1902 trace also attributes 0.451 s to one-electron parallel collection.
The atom lists were already sparse, but each Cartesian coordinate was
serial over all its owned integral tasks, with very few active lanes.
With the packed index, assign one block to each coordinate, retain each
task's inner center/AO sums and fold the task partials in FP64. The matrix
panels keep their ordered fold, there is no new global scratch, and the
unindexed route retains the original bounded collection. Tests compare the
reordered vector sums with both the scalar schedule and libcint differences.

Slurm 1915 passed 23 one-electron/stationary tests (nine skips). Its 18-atom
PBE full warm median moves from 7.982 to 7.619 s, and 30-H r²SCAN from
5.512 to 5.226 s after the block fold. The full candidate matrices differ
from the mid-width build by at most `3.64e-14 Ha/Bohr²` across four cases.
The workspace, completed columns, exact and one-electron primitive counts,
physical and cached CPKS Fock actions, and mixed-grid contractions are
unchanged. Slurm 1916 profiles 54 collection passes in the 18-atom PBE
case: 452.58 ms becomes 4.80 ms (4.09 ms for vectors and 0.70 ms for
matrices), including both launches per pass. This is kernel time, not an
end-to-end saving. PBE0 and B3LYP have slow wall-time samples in 1915 and
are not established endpoint wins. For PBE0 the interleaved 1916 rerun
has 9.92/10.50 s baseline and 7.96/8.60 s candidate samples, reversing
the 1915 ordering; preserve all raw samples. The PBE Hessian phase still
takes 7.32–7.58 s natively versus 3.95–3.96 s in GPU4PySCF (1913).

The 96-H geometry shape (STO-3G, 82,944 points, 96 AOs, 288 AO/primitive
incidences) independently exceeds three moving-grid limits. It requires
159,252,480 spatial AO values and 477,757,440 preparation terms, each
against 134,217,728, and 378,224,640 point/Becke pair visits against
16,777,216. Full directional AO storage alone is 729 MiB, before 1,215 MiB
of optional resident spatial jets. Raising just a workspace cap or tiling
storage without bounding repeated partition and primitive work does not
admit this Hessian.

Slurm 1914 changes only functional and system size, keeping the same
Direct FP64 fresh-SCF harness and node4 RTX 5090. PBE, PBE0, B3LYP and
r²SCAN all fail at the J/K canonical group limit at 32 H (256 MiB grid
ceiling), and all fail at moving-grid admission at 96 H (1,024 MiB).
The new 30-H PBE measurements succeed: native 5.59–5.68 s against
GPU4PySCF 1.9.0 at 6.09–6.16 s. For 30-H PBE0 the two native samples
are 6.32 and 7.73 s against 6.60 and 6.67 s: no reliable win. Raw
cross-engine Hessian maxima are `3.31e-9` (PBE) and `3.06e-9` (PBE0)
Ha/Bohr². None of the admission failures are performance measurements.

## Follow-up: bounded host J/K replay and 32-lane Becke columns

The 100,000 canonical source limit still bounds the retained host schedule.
Only the certificate-bound private dense route may instead visit a bounded
candidate set above that limit. The same 200,000 candidate cap used to build
the certificate includes omitted sources; primitive, projection, reduction
and per-window device storage caps are unchanged. The sizing pass reduces
window sizes and semantic work into a receipt rather than saving expanded
orbits. Execution rebuilds one window at a time and keeps the previous host
upload alive until its stream event. This trades one additional host planning
pass for a bound of two expanded windows plus the certificate, preserves
ordered reductions, and cannot serve arbitrary HVPs or values. Slurm 1917
passed 62 J/K tests (four skips). Its directed case checks exact numerical
equality with retained certified windows, agreement with unscreened derivatives,
identical counted work, a one-byte-short arena, candidate/primitive refusal,
and invalid frozen or response densities. The 32-H r²SCAN acquisition with
the original over-30-atom geometry schedule did not return a matrix before
the diagnostic was stopped after 8 min 21 s; that is not a measured endpoint.

For 31–32 atoms, a warp now retains one switch jet per unordered Becke pair
and reconstructs its generated complement when reading the opposite orientation.
Reuse those shared bytes for all left rows after the suffix warp barrier.
The full 32-lane column uses at most 35,856 bytes of static shared memory
in this layout. Smaller columns retain direct switch lookup; exceptional
points still use scaled mixed jets. At 96 H both the AO and partition work
limits remain; the warp-sized column is not a general large-grid solution.

The frozen one-electron table also needed to recognize 34,816 tasks at 32 H.
A stack sample (Slurm 2535) identified loss of table residency at the former
32,768-task cutoff, not time spent in the new J/K replay. The default dense
task limit now inherits the existing owner job limit; an explicit limit and
the actual byte and primitive ceilings still apply. Set the geometry row
capacity and the XC reduction/AO-owner capacities from a single warp-size
constant so these 32-atom Hessians consume the columns rather than silently
falling back to separate diagonal and rowwise work.

Same node5 RTX 5090, Cartesian STO-3G, Direct FP64, no DF, moving grid and
fresh SCF per call. Seconds below are medians of two warm samples. The first
build had the 32-atom geometry producer but not the matching XC column
consumer (Slurm 2536); Slurm 2537 qualifies the shared capacity.

| 32 H XC | Rowwise full | Batched full | GPU4PySCF full | Batched / GPU4PySCF Hessian phase | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| r²SCAN | 7.222 | 6.002 | 6.711 | 5.848 / 5.975 | 3.37e-9 |
| PBE | 7.507 | 6.295 | 6.554 | 6.208 / 5.916 | 3.30e-9 |
| PBE0 | 7.603 | 6.386 | 6.901 | 6.300 / 6.345 | 3.06e-9 |
| B3LYP | 7.252 | 6.061 | 6.979 | 5.969 / 6.408 | 3.14e-9 |

Both native builds complete 96 columns, 86,280,552 exact and 30,081,024
one-electron primitive components, 448 physical Fock actions, 192 right grid
directions and 9,216 mixed contractions in all four cases. The column
scratch costs 13,312 additional bytes. Slurm 1921 passed 47 geometry and
stationary CUDA tests (nine skips) and traced 96 sparse adjoint contractions
in 23.91 ms instead of 9,216 row launches. It no longer launches the separate
Becke diagonal kernel; computing the batched Becke rows themselves still
takes 1.32 s. This closes the complete endpoint gap for this system, but PBE
still has a slower Hessian phase and the 18-atom water gap remains separate.

## Follow-up: smaller s/p jets and translation-aligned source windows

The Release SM120 binaries already compile with `-O3`. In the four-center
kernel, the s/p Cartesian degree bounds the *sum* of the three axis degrees.
Reserve three constant coefficients plus that degree, rather than reserving
the maximum degree independently on each axis. Generate the product of the
two pair jets with their disjoint center supports directly in the compiler;
it no longer constructs two expanded four-center jets. The primitive
definition and FP64 contraction order are unchanged. A 96-thread producer
trial (instead of 128 for 81 records) was rejected: Slurm 1922 measures
3.1605 vs 3.1607 s inside the factorized kernels, and Slurm 2538 shows no
endpoint win at 18 water atoms or 32 H.

On the private dense schedule, stop source windows at a first-atom boundary
even when the current window is not full. The existing translation proof
then applies to both windows, omitting the anchor's three evaluations in
each and reducing them from the other atoms' contributions. This can add one
window per boundary. It makes no change to the public arbitrary-direction
schedule or limits, and it never approximates an active derivative.

Slurm 2540 (node5 RTX 5090, warm complete Direct FP64 timings) reduces
four-center primitive components from 221,684,769 to 209,894,895 at 18 water
atoms and from 86,280,552 to 80,753,517 at 32 H. One-electron, grid and physical
Fock work remain unchanged. The combined arm takes 6.94 and 7.08 s for
18-water PBE and PBE0, respectively, vs 7.06 and 7.17 s in the previous
same-card acquisition (2538/2539). The largest water matrix change is
`1.87e-14 Ha/Bohr²`. The 32-H complete endpoints are 6.06, 6.34, 6.43 and
6.10 s for r²SCAN, PBE, PBE0 and B3LYP respectively, within 0.07 s of the
previous native schedule and still below the matched GPU4PySCF full calls.
The lower work count is not a 32-H wall-time win: Slurm 2541 shows more
small batches and about 0.78 s (9,528 launches) inside the 32-H factorized
kernels, vs 0.76 s (8,865 launches) in 1921 on a different card of the same
model. Slurm 2540 passed 12 independent libcint primitive-jet cases and six
frozen J/K probes, including narrow/wide certified windows, retained/replay
parity and bounded refusal.

At 15 water atoms, tightening both SCF arms' density convergence from
`1e-9` to `2e-11` (energy tolerance `2e-14`) does not resolve the PBE
discrepancy. Slurm 1923 records `8.03e-8 Ha/Bohr²` cross-engine at all three
settings, with native and GPU4PySCF at `3.01e-8` and `5.23e-8` from the
independent CPU-DFT gradient difference, respectively. Retain both gates.
The same job adds 96-H GPU4PySCF Direct warm references: 111.55 s PBE,
125.30 s PBE0 and 117.40 s B3LYP, alongside the earlier 109.49 s r²SCAN.
These are not native timings. A host-only prototype of block pruning
(`block-envelope96.cpp`) using the existing derivative envelope and a
uniform budget share per canonical candidate still retains 2.84 million of
10.84 million groups and visits 6.83 million tree nodes. It cannot replace
the current certificate and planner without a stronger bound or a more
efficient source traversal, even after the moving-grid ceiling is addressed.

## Follow-up: specialize the checked restricted PBE response

Slurm 2541 found that PBE's checked point evaluator still accounts for
373 ms over 652 launches at 32 H, even with the retained regular-row point
Jacobian. The general kernel also carries the unrestricted and B3LYP states
and consumes 255 registers and a 2,848-byte SM120 stack per thread. Select a
restricted PBE kernel at dispatch, retaining the same evaluator, scaled
PBE/PBE0 parameters, signed directions, replay mask and failure policy.
No density cutoff or numerical contract changes. The specialization uses
210 registers and 432 bytes of stack. Slurm 2542 measures 129 ms for the
same 652 launches.

On the same node5 RTX 5090 (Slurm 2542), complete Direct fresh-SCF warm
medians move from 6.350 to 6.087 s at 32-H PBE, from 6.435 to 6.197 s
at 32-H PBE0, and from 6.949 to 6.798 s at 18-water PBE. Hessian-phase
medians are 5.999, 6.111 and 6.658 s, respectively, after specialization.
The maximum raw matrix difference is `1.78e-14 Ha/Bohr²`, and workspace,
primitive components, Fock actions and grid work counts are unchanged.
Slurm 2542 passed 27 selected grid tests against Libxc and the checked
actions. All four 32-H cross-engine comparisons remain within
`3.37e-9 Ha/Bohr²`. PBE's Hessian phase is now 5.999 vs 5.916 s, a small
remaining deficit even though the full fresh-SCF call is faster.

The node5 18-water references in 2543 use the same Direct FP64 harness.
The native timings combine the relevant medians from 2540 (PBE0), 2542
(PBE), and 2543 (r²SCAN/B3LYP). The two warm repeats are fresh SCFs:

| 18 water atoms XC | Native / GPU4PySCF complete (s) | Native / GPU4PySCF Hessian phase (s) | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE | 6.80 / 11.99 | 6.66 / 3.72 | 3.04e-8 |
| PBE0 | 7.08 / 6.50 | 6.96 / 4.43 | 2.73e-8 |
| r²SCAN | 6.84 / 5.23 | 6.63 / 3.87 | 3.70e-8 |
| B3LYP | 6.89 / 8.28 | 6.75 / 4.43 | 2.90e-8 |

GPU4PySCF PBE used 268 SCF cycles in both warm samples, compared with 31
in one earlier run; the PBE full-endpoint ranking is therefore not robust
across these acquisitions. PBE0 used 12 and 25 cycles. The Hessian phase
is slower natively for all four water cases, independent of this variance.
Slurm 1924 also tightened the 15-water PBE reference CPKS tolerance down
to `1e-14`; the independent analytic cross-engine error remained
`8.03e-8 Ha/Bohr²`. Neither SCF nor CPKS tolerance explains it.

## Follow-up: contract Hermite derivatives before lifting Coulomb incidence

For the full-range CUDA s/p primitive, the four product-center scales are
constant across its Hermite terms. Keep coefficient derivatives in the
one-axis/four-center basis and sum Coulomb derivatives in the three
separation coordinates before lifting them to the centers. Scalar compiler
AD still supplies the coefficient weights, Coulomb partials, and both JVP
product rules. A zero-degree axis has a unit coefficient and needs no
coefficient derivative. This changes FP64 summation association, not the
retained primitive set or radial reduction. It does not modify ordinary
SCF/force kernels or the general-angular/range-separated Hessian path.

Slurm 2545 (same node5 RTX 5090, Direct Cartesian STO-3G, FP64, no DF,
fresh SCF and matched moving quadrature) recorded the following warm
medians from two full calls per arm:

| 18-atom water XC | Complete before / after (s) | Hessian before / after (s) |
| --- | ---: | ---: |
| PBE | 6.81 / 6.38 | 6.67 / 6.23 |
| PBE0 | 6.93 / 6.50 | 6.80 / 6.37 |
| B3LYP | 6.87 / 6.45 | 6.74 / 6.31 |
| r²SCAN | 6.80 / 6.39 | 6.60 / 6.18 |

All four completed 54 columns with 209,894,895 four-center and 17,146,080
one-electron primitive components, 468 physical Fock actions and 2,916 mixed
grid contractions per call, before and after. The maximum raw matrix change
was `2.16e-14 Ha/Bohr²`. The 32-H PBE control was effectively unchanged
(6.094 / 6.086 s complete), consistent with the single Hermite term for
four s shells. The 28 independent libcint primitive checks and six
frozen-J/K tests in Slurm 2545 passed. The 18-atom Hessian phase is still
slower than the matched GPU4PySCF reference (3.72–4.43 s from Slurm 2543).

The 21-atom PBE endpoint extends the benchmark geometry by the first water
of an additional dimer at shift `(0.7, 20.4, -0.5)` Bohr, for 49 Cartesian
AOs and 18,144 points. The baseline on the same card (Slurm 2544) completed
63 columns in 13.24 s (13.08 s Hessian), versus 6.26 s (4.67 s Hessian)
for GPU4PySCF. It used 331,493,067 four-center and 31,311,441 one-electron
primitive components, 3,969 mixed XC contractions and 1,846 physical Fock
actions. The analytic baseline cross-engine error is `7.02e-8 Ha/Bohr²`;
changing the Hermite lowering leaves this discrepancy essentially unchanged.
Keep the `5e-8` gate rather than treating that run as qualified parity.

## Follow-up: admit the 21-atom restricted Direct CPKS cache

The 49-AO water cluster has 490 occupied-virtual rotations for 63 nuclear
RHSs. Its operator already passes the existing eight-build-actions-per-RHS
amortization test, but the 1 MiB limit rejected its 1,920,800-byte FP64
matrix. Allow up to 2 MiB for eligible dense Direct RKS (no DF, ECP,
correction or nonlocal grid). The other routes keep their smaller limit.
The operator still competes for response and aggregate memory budgets;
allocation failure or admission refusal leaves physical GMRES, and each
cached solve is checked with the physical operator. This is not an increase
in the 96-H moving-grid or source-work limits.

Same-card Slurm 2546, two fresh-SCF warm calls per arm, with the split-axis
integrals on both arms:

| 21-atom water XC | Complete 1 MiB / 2 MiB / GPU4PySCF (s) | Hessian 1 MiB / 2 MiB / GPU4PySCF (s) | Physical Fock actions before / after |
| --- | ---: | ---: | ---: |
| PBE | 12.60 / 10.08 / 6.26 | 12.44 / 9.92 / 4.67 | 1,846 / 616 |
| r²SCAN | 12.26 / 10.05 / 6.33 | 12.03 / 9.84 / 4.88 | 1,754 / 616 |

The PBE reference comes from Slurm 2544 on the same device. Both native
methods keep all 63 columns and physical CPKS checks, the frozen primitive
census, and the grid contractions. The cached matvec replaces 1,720 PBE and
1,628 r²SCAN actions; the reported physical counts include the operator
build. No forced numerical tolerance or SCF setting was changed. The
largest cache-on/off raw Hessian difference is `2.43e-14 Ha/Bohr²`, while
cross-engine differences are `7.02e-8` (PBE) and `5.43e-8` (r²SCAN).
The independent CPU PySCF PBE whole-matrix oracle reconverges each displaced
SCF and differences the analytic moving-grid gradient with a sixth-order
stencil. Halving the displacement from 0.005 to 0.0025 Bohr changes the
oracle by `2.17e-9 Ha/Bohr²`. At the finer step the candidate and GPU4PySCF
errors are `2.98e-8` and `5.99e-8 Ha/Bohr²`, respectively. The candidate
passes this independent PBE gate, not the cross-engine comparison. Further
work is still needed for speed and cross-engine accuracy parity.

Slurm 2547 passed the three focused final-binary cases: signed r²SCAN
HVP/dense agreement and LDA/PBE public live-peak bounds. Slurm 2546
completed both paired benchmarks and matrix comparisons, but hit its time
limit in the subsequent full stationary pytest invocation; it is not a
full-suite pass.

## Follow-up: retain four radial moments for wide s-only moving grids

For a contracted constant Cartesian AO, retain the four derivatives along
the half-squared-radius coordinate. Scalar compiler AD generates both the
primitive moments and the pullback to the ordinary order-three Cartesian
jet. The arena stores four planes instead of twenty; mixed motion, the
right CPKS direction and the Cartesian XC adjoint reconstruct the needed
spatial entries. A private view explicitly identifies its representation
and the authenticated point coordinates. The expanded accessor must refuse
the compact representation. Host topology selects only wide, regular pure
s bases; the device checks the packed constant monomials and incidence
bound. Smaller and general-angular paths retain their existing kernels;
memory-constrained compact plans still charge bounded AO replay.

At the actual 96-H shape, 82,944 points and 96 AOs, the AO cache drops from
1,215 to 243 MiB and its preparation bound from 477,757,440 to 95,551,488
primitive terms. The host admission probe isolates that change: the compact
plan fits a 1 GiB geometry-only lease when the Becke ceiling is overridden
for **sizing only**. With the production ceiling it still correctly refuses
378,224,640 point/pair visits versus 16,777,216 allowed. The 12-plane
direction scratch is still 729 MiB, and the shared XC lease is separate
work. None of these numbers qualify a 96-H Hessian or its performance.

Slurm 2549 checks the independent polynomial Gaussian oracle through third
order (including coincidence in the AO-only test), resident and replayed
signed mixed directions, false packed topology, and short-arena refusal.
It passed six focused, 47 geometry, and 24 native semilocal XC cases.
The 4,098-point, eight-AO Cartesian case consumes the compact geometry in
the fused PBE XC adjoint and compares every left contraction for two right
columns to expanded AO motion. The next scaling step must stream point
and row tiles with shared right-direction Becke work, not re-evaluate the
same point/pair graph for each left row or just raise the ceiling.

## Follow-up: share the wide Cartesian Becke sweep across left rows

For 33–96 atoms, one block per point evaluates each unordered Becke pair
once per right column. It packs the complementary switch duals, forms
exclusive prefix/suffix products, then reuses the double buffer for the
three axis folds. Two exactly zero right duals make every exclusive
left-factor contribution zero; neither this rule nor the fast path divides
by a switch. Small columns retain the warp path. Budget up to 32 MiB for
reused point scratch, separate from the full left/right output panels. A
short tile which cannot borrow the center coefficients reserves them
separately. If the optional arena does not fit, the column is not admitted.

Out-of-domain points (e.g. extended measures) use the existing scaled
quotient with compact right-column center coefficients. A device queue
bounds repairs by point/pair visits across all left rows. If it overflows,
the caller's private transaction fails before any Hessian is published.
The bound is not a measurement of repair visits. For a 34-atom grid with
29,376 points, a normal wide sweep would visit 16,479,936 point pairs per
right column instead of 1,680,953,472 for independent scaled rows, plus
at most 16,777,216 visits for repairs. These are semantic work bounds,
not wall timings or proof that a full 34-atom call selected the path.

This does not change the 16,777,216 preparation-pair limit, the full
directional AO and weight scratch, or the frozen J/K and one-electron
admission. In particular, it does not admit the 82,944-point 96-H
endpoint. A synthetic 96-center column hit the repair limit in Slurm
2551; the derivative fixture can reserve all 96 points explicitly without
raising the production limit. The 65-center fixture tests both a zero
budget and an overflowing one-point queue with forced extended measures.

Slurm 2552 supplements the 21-atom water data (same RTX 5090, FP64 Direct,
no DF, matched moving quadrature, fresh SCF every call; two warm samples):

| XC | Complete native / GPU4PySCF (s) | Hessian native / GPU4PySCF (s) | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE0 | 10.23 / 7.87 | 10.09 / 5.55 | 3.80e-8 |
| B3LYP | 10.20 / 7.16 | 10.03 / 5.53 | 5.07e-8 |

Both native calls completed 63 columns, 616 physical Fock actions,
331,493,067 four-center and 31,311,441 one-electron primitive components,
and 3,969 mixed grid contractions. B3LYP misses the unchanged `5e-8`
gate. These are performance baselines for the smaller path, not gains
attributed to the wide Becke sweep. A first CPU B3LYP gradient-difference
diagnostic did not agree with either analytic engine (roughly `2.8e-2`
on the selected column), so is not used as an oracle until the CPU
configuration and hybrid gradient are reconciled.

## Follow-up: remove two 34-H dense-response cliffs

The 34-H PBE run with 29,376 points fits the geometry preparation work
ceiling, but the first 64 MiB integral-budget attempt did not finish in
six minutes. Slurm 2557's eight-second trace spent 81.3% of the recorded
kernel time in the scalar nuclear attraction leaf, with 76,987 such
launches. Its repeated one-electron job descriptors stored six full-width
integers per Cartesian component, even though the admitted s/p/d/f powers
are in `[0,3]`. Pack each shell's three powers in a byte without changing
component, AO, or summation order. The 41,616-job STO-3G 34-H shape can
now keep the one-electron batch inside the existing 64 MiB caller budget.
Slurm 2559 passed 23 native one-electron cases, including the independent
libcint displacement oracle and bounded schedule fallbacks. The host
test also checks 34-center batch admission within that same budget.

The faster acquisition exposed a different failure: the dense audit
rejected the raw matrix, not a failed grid repair or response iteration.
The zero-based `(101,98)` and `(98,101)` entries differed by `5.87e-5`
in Slurm 2561; raising the J/K cap for an unscreened diagnostic did not fix
the dense route. The separately admitted scalar HVP and selected dense
columns with scalar geometry agreed with each other and with GPU4PySCF
within `3.27e-9 Ha/Bohr²` on that entry. The SCF energies agreed within
`2.65e-9 Ha`. No invalid dense matrix was published and the diagnostic
capacity override and printouts were removed.

The sparse XC consumer had been widened to 96 atoms, but its exact AO
incidence initializer still populated only one center per lane of a
single 32-thread block. Rows for centers 32 and later therefore used
uninitialized incidence. Stride through all centers in the initializer.
The regression uses the full 34-H moving quadrature and checks both
directions of the terminal dimer against scalar partition derivatives and
expanded AO-motion contraction, rather than testing only right seeds in
the first block of centers. Slurm 2556 separately confirms that 96 H
still fails moving-grid admission with a 1 GiB grid budget.

Slurm 2567 passed all five focused wide-grid cases and obtained the first
complete native Direct 34-H comparison for all four methods. The same
RTX 5090 and matched quadrature were used, with two warm fresh-SCF runs per
arm. No raw matrix was symmetrized:

| XC | Complete native / GPU4PySCF (s) | Hessian native / GPU4PySCF (s) | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE | 8.03 / 7.42 | 7.93 / 6.75 | 3.30e-9 |
| PBE0 | 8.16 / 7.84 | 8.06 / 7.27 | 3.06e-9 |
| B3LYP | 8.03 / 7.91 | 7.93 / 7.29 | 3.14e-9 |
| r²SCAN | 7.94 / 7.60 | 7.77 / 6.98 | 3.37e-9 |

Each completed 102 columns, 95,744,916 four-center and 38,203,488
one-electron primitive components, 493 physical Fock actions, and 10,404
mixed XC contractions. Slurm 2568 profiled the complete 34-H and 21-water
PBE endpoints on this binary. The wide Becke column took 2.49 s (37.5% of
GPU kernel time) over 11,730 launches at 34 H. The principal 21-water cost
was different: 4.15 s (52.3%) in 42,069 factorized four-center launches.
Optimizing the wide Becke column alone cannot close the water-cluster gap.

## Follow-up: keep medium-width Becke product planes in shared memory

Use the same wide Cartesian row algorithm and exception-repair queue for
33–36 centers, but keep its two product planes block-local. The full
static allocation fits within 48 KiB. The path no longer needs a global
scratch tile for the two planes; it can dispatch all 29,376 grid points of
34 H together. Above 36 centers retain the bounded global tile. This does
not change the candidate-pair census, partition preparation limit,
precision, or either repair refusal gate.

Same-card Slurm 2569 retained-binary comparison with Slurm 2567 (FP64
Direct, matched moving quadrature, fresh SCF per call, two warm samples):

| XC | Complete native / GPU4PySCF (s) | Hessian native / GPU4PySCF (s) | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE | 7.25 / 7.42 | 7.15 / 6.75 | 3.30e-9 |
| PBE0 | 7.35 / 7.84 | 7.25 / 7.27 | 3.06e-9 |
| B3LYP | 7.28 / 7.91 | 7.18 / 7.29 | 3.14e-9 |
| r²SCAN | 7.21 / 7.60 | 7.05 / 6.98 | 3.37e-9 |

The same-run PBE baseline was 8.00 s complete (7.90 s Hessian). The
34-H Becke sweep dropped from 2.49 to 1.74 kernel-seconds and from
11,730 to 102 launches, with 9,469,952 fewer global device workspace
bytes. All 102 columns and the primitive, Fock and XC counts above
are unchanged; no changed matrix entry exceeded `3.06e-15 Ha/Bohr²`
against the old storage layout. Seven focused GPU cases passed, with
independent displaced partition weights at the 36/37-center boundary
and separate scaled-fallback and overflow checks at larger sizes.
These are specific geometric/functional results, not a global default
performance promotion. PBE and r²SCAN still trail in the Hessian phase.
Slurm 2576 ran the full grid-geometry device selection on the retained
binary: 58 passed and four skipped. The deselected host generator check
passed separately in the checkout. The final cached rebuild reproduced
the measured library and both probe hashes exactly.

Do not retain the experimental early exit of the exact-zero prefix scan
or compaction of the live product columns. Slurm 2570 and 2574 measured
1.73 s in the Becke kernels for both; full PBE was 7.23 s after early
exit and 7.27 s after compaction, compared with respective same-run
baselines of 7.28 s and 7.22 s. The marginal, inconsistent speedup is
not worth the more complex kernel. The Slurm 2572 grid test run on that
discarded build had 58 passes and four skips, but missed the deterministic
generator test because its deployment contained no `tools/` tree in the
test root. That test passed separately in the checkout; the full
deployment run is not an all-pass result.

For 21-water B3LYP, spelling the CPU functional explicitly as `B3LYPG`
reproduces the prior `2.8e-2` gradient-difference discrepancy (Slurm
2571). Independently differencing reconverged total energies on the
disputed diagonal gives `0.531596516` and `0.531596500 Ha/Bohr²` at
0.02 and 0.01 Bohr (Slurm 2573), consistent with both analytic values
at those steps. The 0.005-Bohr estimate moves by `6.7e-8`, already
larger than the analytic discrepancy. This rejects the large gradient
discrepancy on that diagonal but cannot adjudicate the `5e-8` gate.

Slurm 2575 also varied only the 21-water PBE integral budget across 64,
128 and 256 MiB. Complete medians were 10.080, 10.079 and 10.088 s,
respectively (two warm calls each); the Hessian phase was 9.93 s for
all three. Primitive and Fock counts and live workspace bytes were
unchanged, and the maximum raw matrix change was `2.36e-14 Ha/Bohr²`.
Increasing the integral budget is not a performance remedy for this shape.
The next target is reusing direction-independent work in the four-center
schedule; degree-one and degree-two full-range s/p kernels account for
2.56 s of the 4.15 s measured there, rather than a generic compiler
optimization switch.

## Follow-up: reuse full-range s/p quartet geometry within a dense window

The source-major frozen sweep can prepare direction-independent Gaussian
geometry and Boys moments for a canonical primitive record once. Reuse
those FP64 values across its Cartesian components and nuclear columns,
while leaving directional jets and radial folds unchanged. Bind on the
same stream as compact records, only for the existing full-range s/p
paired path and for windows with at least two active atoms in the sweep.
Reserve at most 32 MiB of spare workspace, without splitting windows or
reducing any mandatory allocation. Non-fitting windows and the ordinary
signed HVP continue to use the inline geometry binder.

The 16 MiB experiment (Slurm 2578) served just 1,005 of 42,069
four-center launches; complete 21-water PBE took 10.038 vs 10.065 s.
With the 32 MiB bound, Slurm 2580 served 42,066 of 42,069 launches.
Geometry preparation took 13.5 ms and the integral kernels took 3.28 s
instead of the prior 4.15 s. Same-job complete PBE was 9.185 vs 10.045 s;
the Hessian stage was 9.034 vs 9.894 s. Live workspace grew by
33,177,600 bytes, inside the existing 64 MiB integral budget. The raw
matrix changed by `1.65e-14 Ha/Bohr²`; 63 columns, 331,493,067 four-center
and 31,311,441 one-electron primitive components, 616 physical Fock
actions, and 3,969 XC contractions are unchanged. Seven selected native
J/K checks passed. The separate retained-window Coulomb fixture failed
on both the 16 MiB experiment and the older split-axis probe (Slurm 2579);
it was excluded from the 32 MiB focused run, not counted as a pass.
After tightening admission to reserve only a window that actually fits
and adding preparation/reuse counters, Slurm 2582 reproduced the complete
PBE speedup (9.185 vs 10.053 s) and unchanged live extent. The raw change
was `2.31e-14 Ha/Bohr²`, and the same seven focused checks passed.

Same-device Slurm 2581 extended qualification to the other three
21-water functionals and two 34-H controls. Two warm calls per arm;
earlier GPU4PySCF measurements use the same Direct FP64 quadrature.

| Shape and XC | Complete native / GPU4PySCF (s) | Hessian native / GPU4PySCF (s) | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| 21 water, PBE | 9.18 / 6.26 | 9.03 / 4.67 | 7.02e-8 |
| 21 water, PBE0 | 9.47 / 7.87 | 9.33 / 5.55 | 3.80e-8 |
| 21 water, B3LYP | 9.43 / 7.16 | 9.26 / 5.53 | 5.07e-8 |
| 21 water, r²SCAN | 9.19 / 6.33 | 8.96 / 4.88 | 5.43e-8 |
| 34 H, PBE | 7.02 / 7.42 | 6.92 / 6.75 | 3.30e-9 |
| 34 H, r²SCAN | 6.93 / 7.60 | 6.77 / 6.98 | 3.37e-9 |

The 21-water matrices agree with their prior native versions within
`2.45e-14 Ha/Bohr²`; the optimization does not resolve the independent
cross-engine discrepancies. The r²SCAN CPU gradient difference on
column 56 in Slurm 2577 also did not reproduce either analytic result
and is not used as an oracle. The 96-H admission thresholds are unchanged.

## Follow-up: submit only active Cartesian writers in source-major J/K

Trimming empty tails from a homogeneous batch run still launches interior
holes. Instead, build a bounded list of actually active component writers
for each frozen atom and bucket it by total shell angular momentum. The
three Cartesian axes share this list. Each indexed block writes its own
canonical cache component, so dispatch order does not affect the FP64
radial, projection, or translation folds. Require the prepared full-range
s/p window; all other dispatches keep the per-tile path. Cap the reused
device slot at 4 MiB and retained host indices and spans at 24 MiB. A
certified replay keeps only the slot extent, regenerates its index with
each window, and releases it after the existing upload event. Lack of
space or a failed optional host allocation leaves per-tile dispatch.

Slurm 2583 and 2584 left the 21-water endpoint on per-tile dispatch at
both 64 and 128 MiB because the index was available only for retained
windows. At the original 64 MiB integral budget, Slurm 2585 hits indexed
dispatch in certified replay. In a complete 21-water PBE execution,
four-center launches drop from 42,069 to 20,424 and blocks from 14,666,739
to 4,092,507. Four-center CUDA time drops from 3.264 to 1.755 s; 257,280
additional live bytes fit within the existing budget. Both arms evaluate
331,493,067 four-center and 31,311,441 one-electron primitive components,
616 physical Fock actions and 3,969 mixed XC contractions for 63 columns.
The same-job two-warm-run medians are 9.174 -> 7.762 s complete and
9.025 -> 7.613 s Hessian, versus the earlier Direct GPU4PySCF 6.26 and
4.67 s, respectively. The largest raw change is `1.55e-14 Ha/Bohr²`.
Eight focused native probes pass, including indexed retained/replay
parity and an independent per-column exact J/K comparison on the screened
four-center fixture. The retained-window Coulomb failure from Slurm 2579
remains outside this focused pass.

Slurm 2587 uses the same complete endpoint and two warm samples per arm
for the other 21-water functionals. Four-center and one-electron primitive
components, Fock actions, and XC contractions match for both arms of
every functional. No matrix changes by more than `2.45e-14 Ha/Bohr²`
from the prepared-geometry control.

| XC | Complete control -> indexed / GPU4PySCF (s) | Hessian control -> indexed / GPU4PySCF (s) | Max raw cross-engine error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE | 9.17 -> 7.76 / 6.26 | 9.03 -> 7.61 / 4.67 | 7.02e-8 |
| PBE0 | 9.38 -> 8.02 / 7.87 | 9.25 -> 7.88 / 5.55 | 3.80e-8 |
| B3LYP | 9.33 -> 8.02 / 7.16 | 9.17 -> 7.86 / 5.53 | 5.07e-8 |
| r²SCAN | 9.14 -> 7.71 / 6.33 | 8.92 -> 7.50 / 4.88 | 5.43e-8 |

Only PBE0 meets the independent cross-engine gate at 21 water atoms.
No water functional has caught up on the isolated Hessian stage.

Slurm 2586 checks the two hybrid controls at 34 H. Both arms include
fresh SCF; timings are two-warm-run medians. The cross-engine error gate
remains `5e-8 Ha/Bohr²`.

| XC | Complete native / GPU4PySCF (s) | Hessian native / GPU4PySCF (s) | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE0 | 7.06 / 7.86 | 6.98 / 7.28 | 3.06e-9 |
| B3LYP | 6.97 / 7.94 | 6.88 / 7.30 | 3.14e-9 |

The 36-H r²SCAN run still refuses during moving-grid preparation: even the
1 GiB grid lease cannot help the point/pair work cap (19,595,520 visits
versus 16,777,216 allowed). GPU4PySCF completes the same 36-H r²SCAN case
in 8.76 s (7.84 s Hessian). Do not extrapolate the native 34-H timings
past this boundary or treat capacity increases as an optimization. The
96-H Hessian still requires work on both moving-grid and integral admission.

Separately, Slurm 2588 runs the ordinary 96-atom r²SCAN-3c Direct FP64
energy and analytic forces endpoint without invoking Hessian or HVP.
Against the #3134 baseline, the indexed worktree has identical total
energies, at most `7.94e-12 Ha/Bohr` change in one force component, the
same SCF domain, and the same Coulomb/exchange work (11 Coulomb builds
cold, one warm, no exchange). One cold and one warm sample per arm give
28.729 -> 28.720 s and 10.267 -> 10.274 s, respectively. This is an
ordinary endpoint regression check, not a new timing population or a
96-atom Hessian result.

## Follow-up: authenticate stationary partition weights without point jets

Hessian geometry preparation verifies the reference quadrature but does
not consume any of the directional outputs. For up to 96 atoms, assign
one block to each grid point, evaluate each complementary Becke switch
once, and fold the per-center scalar products in the original scaled
order. Keep the same extended-exponent norm gate, center pair geometry,
dimensionless weight tolerance, and finite-value checks. Only stationary
verification uses this route; actions and their scaled repair queue are
unchanged. Shared scratch is bounded below 40 KiB per block and the
existing global workspace and point-pair census are unchanged. Above
the shape limit the general zero-direction verifier remains available.

In Slurm 2589, 34-H PBE preparation visits the same 16,479,936 point
pairs but replaces 29 directional-kernel launches (90.36 ms) with a
single scalar verification launch (4.81 ms). The 102 repair-queue
launches outside preparation still cost 1.35 ms in aggregate. The
same-run warm complete median is 6.956 -> 6.840 s and Hessian is
6.866 -> 6.750 s, versus GPU4PySCF 7.421 and 6.753 s, respectively
(Slurm 2567). Treat the Hessian as tied, not faster. For 21-water PBE
the complete medians are 7.705 -> 7.700 s: this is not a material
improvement at that shape. Semantic work counters do not change and
the largest raw matrix changes are `1.67e-15` (34 H) and `1.55e-14`
(21 water) `Ha/Bohr²`. The geometry selection passed 58 tests, skipped
four; the separate build-only generator test passed in the checkout.

36-H scaling requires a separate remedy. A diagnostic binary (not a
default policy change) sets the two known census bounds to 19,595,520
moving-grid pair visits and 222,111 canonical J/K candidates only for the
36-H workload. The first version gated the override on a radial AO cache
not enabled at 36 AOs, so its refusal was not diagnostic. The corrected
binary enters the endpoint in Slurm 2591 but times out at four minutes
without even one complete r²SCAN matrix. Slurm 2592 profiles its first
25 seconds: the scalar nuclear-attraction leaf consumes 15.70 s (79% of
recorded kernel time) in 197,651 launches. Corresponding generic
projection also consumes 1.40 s in 210,610 launches. This is the same
batch-residency cliff as the previous 34-H failure, not a moving-grid
performance measurement or evidence for increasing either default cap.

## Follow-up: share immutable center positions across one-electron records

Keep only the two primitive exponents and the contracted scale per
batched one-electron record. The generated launcher reconstructs its
ordinary operand record from the authenticated AoBasis and the same center
incidence; the primitive, FP64 radial order, and ordered projection do
not change. Reserve the packed batch only when its descriptors also fit.
Try the complete-record plan otherwise and retain the scalar workspace
and failure gates. No integral or global workspace ceiling changes.

The host scheduling regression admits the 36-H batch at the 64 MiB
integral cap with the packed binding and not with the old record layout.
Slurm 2593 passes 23 native one-electron tests (four skipped), including
independent primitive/value checks. The 34-H and 21-water PBE complete
matrices differ from the scalar-preparation control by `1.23e-15` and
`2.23e-14 Ha/Bohr²`, respectively; 21-water live device storage decreases
from 519,216,476 to 501,488,220 bytes. The local timing log was truncated
on a full filesystem; the probe JUnit and full raw matrix pairs are in
the deployment. Do not treat the truncated record as a full timing A/B.

At 36 H, rerun the same diagnostic bounds as in Slurm 2591. The
bounded Slurm 2595 trace now reaches all 108 nuclear-attraction batch
launches in 0.66 kernel-seconds, versus 197,651 scalar launches in just
the first 25 seconds of Slurm 2592. But the capped complete run (Slurm
2594) still returns no matrix in four minutes. The new leading kernel
is the general point-wise partition path (4,359 launches, 15.78 seconds,
83.8% of the 25-second trace); there is no Cartesian whole-column
launch. The two full XC adjoint AO panels alone require 71,663,616 bytes
at 36 H, beyond the existing 64 MiB optional adjoint budget. At 34 H the
same pair occupies 63,922,176 bytes. Stationary weight authentication
itself takes 5.50 ms at 36 H. Keeping just one live adjoint AO panel or
reusing idle owned storage is the next candidate; releasing the workspace
cap alone would neither prove complete 36-H performance nor scale to 96 H.

## Follow-up: reuse private right AO tiles for the mixed XC cotangent

Keep the read-only fused adjoint when both outputs fit. Under a tight
adjoint lease, a bound private transaction stores its second output in the
right AO panel after the last read of each point tile. All orbitals at the
same point can read that point's right AO values; direct in-place writes
would race with other blocks. Stage the four FP64 planes separately and
commit each tile on the authenticated stream before advancing. Reuse idle
tensor scratch or a bounded optional slot, with a 64-point minimum tile
and both the enclosing and optional byte ceilings enforced. Count the
read and write visits for the copy. Repeated contractions use the
retained outputs; a new right direction reloads the original tangent.
If a launch or copy fails, invalidate the right direction. The checked
expanded path and the factored checked plan do not repurpose the panel.
No ordinary CUDA energy/force leaf or floating-point tolerance changes.

On the 36-H diagnostic shape (31,104 points, 36 AOs) the alternate plan
recovers the full Cartesian sweep under the original 64 MiB adjoint cap.
These diagnostic runs still temporarily use the two shape-specific
*work* ceilings from Slurm 2595. They are not evidence of default
admission. The control with those same bounds and two mandatory
resident outputs returned no matrix in four minutes (Slurm 2594).
Use `tiled-xc-adjoint-36-run.sh` with the matched GPU4PySCF Direct
reference; compare raw matrices at `5e-8 Ha/Bohr²`. The native runs
complete all 108 right directions and 11,664 XC mixed contractions,
with 116,924,553 exact four-center and 47,869,056 one-electron primitive
components for each functional. The diagnostic library and the ordinary
build are separately hashed; both production work ceilings are restored.

Slurm 2598–2599 compares Cartesian STO-3G, FP64, Direct (no DF), and
the same moving quadrature, with fresh SCF per sample. Two warm-sample
medians on the same RTX 5090:

| XC | Complete native / GPU4PySCF (s) | Hessian native / GPU4PySCF (s) | Max raw error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE | 8.24 / 8.71 | 8.14 / 7.73 | 3.30e-9 |
| PBE0 | 8.47 / 8.98 | 8.37 / 8.32 | 3.06e-9 |
| B3LYP | 8.39 / 9.17 | 8.29 / 8.37 | 3.14e-9 |
| r²SCAN | 8.22 / 8.73 | 8.05 / 7.85 | 3.37e-9 |

This closes the diagnostic scaling cliff, not the default admission gap.
The Hessian stage itself is near parity, not a universal improvement;
the water-cluster performance and precision gaps and 96-H Hessian
admission are still open.

Slurm 2600 passes 16 focused tests on the restored normal build. Its
tight-lease fixture cannot store both outputs and checks nonzero signed
AO and density directions against the expanded forward contraction,
including the last partial tile, cache replay and rebinding. The 34-H
and 21-water PBE controls change by only `1.28e-15` and `2.53e-14`
Ha/Bohr² from the compact one-electron control. Slurm 2601 traces a
complete diagnostic 36-H r²SCAN call: the 108 Becke column kernels
take 2.08 s (32.6% of GPU kernel time), while the 864 fused adjoint
tiles take 0.208 kernel-seconds. Prioritize the Becke columns for this
shape, not retuning the compiler flags on the AO adjoint or increasing
the work ceilings. The 21-water trace still ranks the four-center
factorized integral kernel first.

Slurm 2602 repeats the ordinary 96-atom r²SCAN-3c Direct FP64 energy
and force comparison on the final normal build, without Hessian or HVP.
Against the pre-optimization baseline the maximum energy and force
component changes are `1.37e-12 Ha` and `9.23e-12 Ha/Bohr`; SCF
iterations, Fock builds, functional domain and Coulomb/exchange work
are unchanged. One cold/warm pair per arm gives 28.71/10.27 s versus
28.68/10.25 s. These are regression samples, not a speedup claim.

## Follow-up: reduce wide Becke rows over incident center pairs

For wide Cartesian columns, keep the ordered exclusive products and
right duals. At a fixed grid point a left center moves only its incident
switches. Each incident switch feeds two complementary center products;
sum their signed exclusive slopes for the denominator and select the
grid owner's slope for the numerator. Accumulate all three left axes in
registers, project each fixed-point row through the same right dual
quotient, and reconstruct the moving owner's row by translation. This
replaces three dense atom-product derivative tables and their reductions
with one local result vector per atom. Exact-zero inactive columns must
publish zero slopes before their neighbors read them. Retain the same
pair census, FP64 exponent/finite gates, bounded scaled repairs and narrow
path. The addition order changes; the dense matrix is still audited raw.

Slurm 2603, 36 H Direct FP64 with the same two diagnostic census bounds
for both arms, uses a cold call plus two warm fresh-SCF calls. PBE complete
/ Hessian medians change from 8.23 / 8.13 s to 8.04 / 7.94 s; r²SCAN
from 8.27 / 8.09 s to 8.02 / 7.85 s. Both arms complete the same 108
directions, 540 physical Fock actions, 11,664 XC contractions, 116,924,553
four-center and 47,869,056 one-electron primitive components. Max raw
same-engine changes are `1.61e-15` and `2.78e-15 Ha/Bohr²`; independent
Direct GPU4PySCF errors remain below `3.4e-9`. The 108 r²SCAN Becke
column kernels take 1.842 s versus 2.076 s in Slurm 2601. This brings
r²SCAN to isolated-Hessian parity for the diagnostic shape; PBE still
trails 7.73 s from GPU4PySCF. Default admission ceilings are unchanged.
Do not retain the extra stationary-center branch and zero-slope skip
(Slurm 2604): the complete Hessian does not benefit over the simpler
incident-pair reduction and measured Becke time is 1.863 s rather than
1.842 s. Its independent 36-center adjusted-radii oracle is retained.

On the restored normal build, Slurm 2605 passes 59 geometry cases
(four skips); the deselected build-only determinism test passes locally.
Two warm fresh-SCF samples at 34 H give PBE 6.68 / 6.58 s complete /
Hessian versus GPU4PySCF 7.43 / 6.77 s, and r²SCAN 6.61 / 6.45 s
versus 7.68 / 7.04 s, both with raw error below `3.4e-9 Ha/Bohr²`.
The narrow 21-water PBE control changes by `2.04e-14 Ha/Bohr²` and
retains the same work counts; its single warm Hessian is 7.54 s, still
well behind the previously measured 4.67 s reference. Neither the
21-water accuracy gate nor the 96-H Hessian admission gap is resolved.

## Follow-up: compact the indexed low-degree four-center jet

The indexed window header now also bounds the radial population per active
degree and atom. Launch only the warps needed by that population and size
shared memory accordingly. Both 64-record chunks retain their original FP64
addition order. The authenticated prepared binder needs only the stored
contraction multiplier, not an expanded record with replicated coordinates.
Slurm 2606 holds all semantic work and scratch budgets fixed: warm 21-water
PBE complete time changes from 7.727 to 7.634 s and Hessian time from 7.572
to 7.479 s, with at most `1.00e-14 Ha/Bohr²` raw matrix drift. Nine focused
J/K cases pass, including an 81-record signed contraction across chunks.

For one or two p shells on the indexed prepared full-range path, specialize
the Hermite product in a permuted Cartesian frame. Generate from the same
pair expansion and Coulomb DAG. The four layouts distinguish a single p,
two same-axis factors in one pair, two same-axis factors in different pairs,
and two distinct axes. Differentiate at most two pair shifts and three
product-separation coordinates, then apply their affine incidence back to
the four physical centers once. The Gaussian prefactor keeps its existing
jet. The signed ket inverse applies to Hermite orders, not the same-pair
covariance. The ordinary/scalar, non-indexed, range-separated, and other
angular paths do not use this specialization.

Same-device PBE traces from Slurm 2606 and 2608 retain 6,702/5,745 indexed
degree-one/two launches and move from 236/236 to 164/168 registers per
thread. Those degrees take 0.453/0.583 s before and 0.268/0.330 s after.
All four-center launches together take 1.644 -> 1.209 s; the complete
endpoint improves less than this isolated kernel total. The compiler uses
the same `-O3`, FP64, no-FMA policy on both sides. This is removal of
runtime-indexed jet work, not a compiler-flag change.

Slurm 2608: same RTX 5090, matched Direct/no-DF FP64 moving grid, two warm
fresh-SCF samples. From radial-lane control to compact jet:

| XC | Complete control -> compact / GPU4PySCF (s) | Hessian control -> compact / GPU4PySCF (s) | Max raw cross-engine error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE | 7.61 -> 7.40 / 6.33 | 7.46 -> 7.24 / 4.69 | 7.02e-8 |
| PBE0 | 7.88 -> 7.63 / 7.84 | 7.74 -> 7.49 / 5.55 | 3.80e-8 |
| B3LYP | 7.83 -> 7.58 / 8.22 | 7.66 -> 7.41 / 5.55 | 5.07e-8 |
| r²SCAN | 7.64 -> 7.39 / 6.35 | 7.41 -> 7.16 / 4.91 | 5.43e-8 |

The two native arms agree within `2.98e-14 Ha/Bohr²` over all measured
repeat pairs. All 63 columns, the one- and four-center primitive census,
Fock actions, and XC contractions are unchanged within each functional.
The independent PySCF PBE sixth-order gradient difference remains
`2.98e-8 Ha/Bohr²` from the candidate; the separate cross-engine gate
remains `5e-8 Ha/Bohr²` and only PBE0 meets it. Timing differences in
reference SCF between jobs are not evidence of a Hessian win. Slurm 2607
passes 11 selected J/K tests (including indexed table replay and the
four-distinct-center independent comparison). The subsequent timing CLI
invocation in 2607 had a missing atom argument; the corrected timings are
from 2608, not 2607. Keep the pre-existing retained-window Coulomb failure
as a separate test result, not part of the 11 passes.

## Follow-up: spend the fixed CPKS tile lease on points rather than planes

A fixed-grid potential transform uses four AO planes; the moving transform
uses eight. The private fixed-grid CPKS response can therefore cover up to
twice as many points per tile in the same scratch lease. Keep the public
checked and moving-geometry schedules unchanged. The private result still
audits nonfinite data, including an invalid direction followed by replay.
The first experiment widened the nuclear grid owner's fixed tile instead:
Slurm 2610 cut only 18 of 49,196 tile pairs in the 21-water PBE trace and
changed every median by less than 0.05 s. Do not retain that schedule; the
induced CPKS operator uses the separate private fixed-grid action.

On Slurm 2614 the private specialization reduces the 21-water PBE stage
and reduction launches from 49,196 each to 25,788 each, and their combined
kernel time from 0.616 to 0.453 s. The live workspace, physical Fock action
counts, nuclear columns, XC contractions, and one-/four-center primitive
counts do not change. The two native arms differ by at most
`3.27e-14 Ha/Bohr²` over all repeats. Four grid tests pass, including a
36-AO, 4,096-point checked-versus-private replay after a device audit error.

This is an additional same-card A/B against the compact jet. The reference
uses Slurm 2608's samples on the same GPU with the same settings. Two warm
fresh-SCF medians per arm:

| XC | Complete control -> private tile / GPU4PySCF (s) | Hessian control -> private tile / GPU4PySCF (s) | Max raw cross-engine error (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| PBE | 7.39 -> 7.18 / 6.33 | 7.24 -> 7.03 / 4.69 | 7.02e-8 |
| PBE0 | 7.66 -> 7.39 / 7.84 | 7.52 -> 7.25 / 5.55 | 3.80e-8 |
| B3LYP | 7.62 -> 7.35 / 8.22 | 7.44 -> 7.18 / 5.55 | 5.07e-8 |
| r²SCAN | 7.42 -> 7.14 / 6.35 | 7.19 -> 6.92 / 4.91 | 5.43e-8 |

The 21-water unrelaxed cross-engine criteria are not met; even the two
complete endpoints with faster elapsed times still have slower Hessian
stages. PBE remains `2.98e-8 Ha/Bohr²` from the independent CPU FD oracle.

The 34-H r²SCAN case on Slurm 2614 is 6.48 s complete / 6.32 s Hessian,
versus 7.66 / 7.00 s for the same-device reference from Slurm 2610. The
raw repeat-pair error is `3.37e-9 Ha/Bohr²`. The 36-H ordinary admission
bound is unchanged. Slurm 2615 independently compares the final binary's
ordinary 96-atom r²SCAN-3c Direct FP64 energy and analytic force against
#3134, without Hessian or HVP. The maximum difference is `4.55e-13 Ha`
in energy and `8.06e-12 Ha/Bohr` in force; SCF domain, iterations, Fock
builds and J/K work match. The single cold/warm samples are 28.74/10.26 s
for the baseline and 28.71/10.26 s for the candidate, not a speedup claim.

The larger 24-water control (56 AOs, 20,736 points) refuses during exact
J/K before a Hessian is returned for both r²SCAN and PBE0 (Slurm 2611 and
2613). Forty STO-3G shells imply 820 shell pairs and 336,610 canonical
quartets, above the certified census's 200,000 candidate ceiling. This is
the same class of admission issue as 36 H, not a slow timed native
endpoint. GPU4PySCF PBE0 completes in 10.90 s (6.21 s Hessian), median of
two warm samples. The r²SCAN reference has one warm 9.57 s / 6.04 s
sample and one 400-cycle SCF nonconvergence; do not report a paired median.
Prioritize reducing the candidate enumeration itself rather than enlarging
the census or treating a refusal as a performance sample.

## Follow-up: screen canonical sources before decoding a window

For eligible frozen full-range J/K sources, rank conservative Coulomb
curvature envelopes and charge each omission to the affected physical
right atoms using outward-rounded arithmetic. The analytic monomial norm
bound applies to normalized radial records; absent a certificate, retain
the source. Record the mask in a bounded bitmap and replay survivors in
canonical order; only retained quartets are decoded. The per-right-atom
omission budget is `1e-14`, and the independent 200,000 retained-source
limit is unchanged.

Slurm 2616 passes 10 focused real-GPU J/K tests. On 21 water atoms, the
PBE Hessian agrees with the private CPKS tile control to `3.55e-14 Ha/Bohr²`;
the exact primitive-component count drops from 331,493,067 to 319,681,809,
with all 63 directions, Fock actions and mixed grid contractions preserved.
The envelope's 13 host tests pass. The 24-water PBE0/r²SCAN attempts still
reject at J/K admission: offline sizing with the physical atom map finds
201,028 retained sources out of 336,610, after 172,708 screening checks.
Counting shells as separate atoms yields an inapplicable lower estimate.

The 36-H r²SCAN default grid still rejects at moving-grid admission. At a
matched coarser 25,920-point grid (`36 × 10 × 6 × 12`), the same screened
native binary and GPU4PySCF complete: warm medians are 7.47 vs 8.71 s end
to end and 7.31 vs 7.84 s for Hessian, with `3.80e-9 Ha/Bohr²` maximum raw
matrix difference. The old timing probe's reference point-count label was
hardcoded to 12 radial points (31,104 total); it did not describe this run.
Coarsening is a benchmark control, not a new production default. Neither
this run nor the earlier 96-atom ordinary E/F comparison qualifies a
96-atom Hessian. Evidence: `n5-sparse-certified-sources.log`,
`monomial-topology-sizing.log`, and the `sparse-certified-sources/` deployment
with separate candidate and private-tile binary hashes. These jobs used
builds atop #3134, before updating the branch to include #3135.

## Follow-up: charge only the pairs moved by a frozen right atom

When the right atom belongs to only one AO pair, the derivative of the
other pair in that column is identically zero. The mixed product rule
needs only that pair's second-order term and one cross term; the shared
atom still needs both second-order and both cross terms. Charge each
column's own bound and sort by the maximum relevant bound. The monotone
candidate-search rank condition still holds because every bound term it
uses affects at least one atom. The tolerance and both work limits remain
unchanged.

The offline Cartesian STO-3G census with physical atom ownership keeps
199,513 of 336,610 24-water sources, down from 201,028 with the coarser
column bound, fitting the 200,000 retained-source gate. The rank check
count stays at 172,708 and the maximum aggregate charge is below `1e-14`.
This certifies one planner gate, not a full Hessian. Fourteen host tests
pass, including independent libcint gradient finite differences with an
atom shared across the bra and ket and a per-atom ledger reconstruction.
Slurm 2617 passes 10 focused real-GPU J/K tests on the integrated #3135
build. With the complete Python deployment, Slurm 2619 completes both
24-water endpoints and all 72 columns, with maximum raw GPU4PySCF errors
`2.73e-8` (PBE0) and `3.79e-8 Ha/Bohr²` (r²SCAN). The r²SCAN comparator
has only one converged warm reference, not a reference median. The
initial endpoint launch omitted runtime data files; only the later
fully provisioned job supplies this comparison. Evidence:
`atom-columns-sizing.log`, `atom-columns-host-pytest.log`, and
`n5-atom-columns-master-complete.log`.

For the 36-H default grid, a host census found essentially no switches
clamped at the coordinate endpoints (19,595,520 of 19,595,520 interior).
Do not expect an exact clamp-skip bitmap to solve this grid-work ceiling.
The same is true for 96 H (378,224,640 pair visits). An offline run with
12 million allowed rank checks, solely to size the 96-H integral problem,
checks 9,375,944 candidates and still retains 2,129,205 of 10,841,496
canonical sources at `1e-14`. Neither cap should be enlarged to fit that
census. A reusable geometry and a more selective integral contraction
need their own work receipts and independent tests. Slurm 2619 confirms
default 36-H moving-grid refusal on the integrated binary.

## Follow-up: bound the additional repeated-RHS CPKS lease

With exact J/K admitted on 24 water atoms, the 640-rotation CPKS
operator needs 3,276,800 FP64 bytes and fails both the 2 MiB size gate
and the eight-rotations-per-RHS work gate. In the new 2–4 MiB tier for
eligible dense Direct RKS, require at least one nuclear RHS per ten
rotations. Smaller caches retain their existing thresholds, as do other
backends. Keep the aggregate and response caps, the physical GMRES
fallback and the physical residual check for every cached column.

Slurm 2620, same RTX 5090 and two warm trials, holds the job geometry,
grid, and frozen integrals fixed. PBE0 Hessian/endpoint times move from
15.50/15.68 to 11.43/11.61 s; r²SCAN from 14.17/14.41 to 10.83/11.07 s.
Physical Fock actions drop from 2,104 and 2,046 respectively to 784,
including the 640 build columns and checks. The 72 published columns,
5,184 XC contractions and 490,356,747 exact primitive components do not
change. Native A/B raw differences are at most `3.31e-14 Ha/Bohr²` and
cross-engine differences remain below `5e-8` for both methods. The
reference Hessian is still faster (6.21 s for PBE0; one valid 6.04 s warm
sample for r²SCAN). The 24-atom workaround is not a parity claim.

Slurm 2621 finds no benefit from 128 versus 64 MiB of integral workspace
on this path. The 24-water trace points first to four-center derivative
replay and the physical induced J/K operator, then to fixed-grid features
and AO pair reductions. Slurm 2623 doubles the canonical window from 1,024
to 2,048 sources without changing the primitive census; it regresses
21-water PBE Hessian time from 6.87 to 8.11 s and 24-water PBE0 from
11.40 to 13.53 s. Keep 1,024. Slurm 2624 bounds a same-address graph of
the private fixed-grid tile launches; 21-water PBE is unchanged and
24-water PBE0 improves by only 0.065 s. Sixteen grid audit tests pass
in Slurm 2625, but the endpoint result does not justify another retained
graph. Changing submissions alone does not eliminate the repeated Fock
or derivative work.

Slurm 2622 verifies ordinary 96-atom r²SCAN-3c Direct on the screened
plus larger-cache binary against #3134, without Hessian/HVP. Total energy
and maximum force differences are at most `4.55e-13 Ha` and
`1.05e-11 Ha/Bohr`; iterations, Fock builds, domain and J/K work match.
The single warm observations are 10.24 and 10.27 s, not a speedup claim.

Slurm 2626 requalifies the compiled change after the draft's formatting
and test-fixture sync. On the same RTX 5090, ten focused GPU J/K cases
pass, and each Direct FP64 24-water endpoint publishes all 72 columns
with 784 physical Fock actions and 490,356,747 exact primitive components.
With two warm repeats per engine, PBE0 takes 11.40/11.59 s for the
Hessian/complete endpoint versus GPU4PySCF's 6.23/8.55 s. The r²SCAN
times are 10.82/11.07 s versus 6.05/9.71 s. Maximum raw matrix differences
over cold and warm repeats are `2.73e-8` and `3.79e-8 Ha/Bohr²`,
respectively. The source-matched deployment path, binary hashes and
qualification log are `synced-cpks-n5.path`, `synced-cpks.sha256`, and
`n5-synced-cpks.log` in the local evidence directory.

## References

- #3134 and [the prior launch-cliff note](2026-11-22-cuda-direct-hessian-launch-cliffs.md).
- Local `build/1905-gpu4pyscf/scaling/`: `probe.py`, `run.sh`, matching raw
  matrices, compiler-cache receipts and CUDA traces.
- Slurm 2544–2546: `split-axis-probe.py`, `split-axis-run.sh`,
  `cpks-2mib-run.sh`, `n5-split-axis-21-baseline.log`, `n5-split-axis.log`,
  and `n5-cpks-2mib.log` in the same evidence directory. Independent CPU
  gradient differences: `fd_larger_water.py`, `fd21.log` and
  `fd21-all-pbe-rks-*.npy`.
- Compact grid evidence: `radial-grid-run.sh`, `n5-radial-grid-retry.log`,
  matching probe hashes and `radial-grid-*.xml` in the same directory.
- Wide-grid and 21-atom hybrid evidence: `becke-tiled-run.sh`,
  `n5-becke-tiled*.log`, matching probe hashes, and warm raw matrices in
  the deployment's `becke-tiled/` directory.
- Wide-grid incidence and complete profiling: `h34-owner-run.sh`,
  `h34-owner-profile.sh`, `n5-h34-owner*.log`, matching binary/probe hashes,
  raw matrices and traces under the deployment's `h34-owner/` directory.
- Medium-width storage experiments: `becke-block-local-run.sh`,
  `becke-early-zero-run.sh`, `becke-live-rows-timing.sh`, their `n5-*`
  logs, hashes, cache receipts, and matching deployment directories.
- Indexed J/K: `jk-active-replay-run.sh`, `jk-active-replay-hydrogen.sh`,
  `jk-active-replay-water.sh`, `jk-active-replay-ordinary.sh`, their `n5-*`
  logs, and the `jk-active-replay/` deployment with binary hashes, raw
  matrices, ordinary endpoint receipts, probe JUnit report and Nsight
  Systems trace.
- Stationary partition validation: `grid-primal-weights-run.sh`,
  `n5-grid-primal-weights.log`, matching native hashes, probe JUnit
  and `*-profile-h34.sqlite` in the `grid-primal-weights/` deployment.
- 36-H admission diagnostic: `grid-primal-36-diagnostic-run.sh`,
  `grid-primal-36-diagnostic-profile.sh`, `n5-grid-primal-36-diagnostic-v2.log`,
  and the bounded trace under `grid-primal-36-diagnostic-v2/`.
- Packed one-electron records: `core-compact-records-run.sh`,
  `core-compact-36-build.sh`, native probe JUnit and matrix pairs under
  `core-compact-records/`, plus jobs 2594/2595 and the bounded trace in
  `core-compact-36-diagnostic/`.
  CPU hybrid diagnostics: `fd21-b3lypg-explicit.log`,
  `fd21-energy-diagonal.py`, and `fd21-b3lypg-energy-diagonal.log`.
- 21-water integral-budget sweep: `water21-integral-budget-run.sh` and
  `n5-water21-integral-budget.log`.
- Prepared-geometry experiments: `jk-prepared-geometry*-run.sh`,
  `jk-prepared-geometry-functionals.sh`, `n5-jk-prepared-geometry*.log`,
  binary/cache receipts and matching deployment directories. CPU r²SCAN
  check: `fd21-r2scan-explicit.log`.

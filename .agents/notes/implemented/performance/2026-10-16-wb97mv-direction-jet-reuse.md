# Decision: retain bounded last-direction native integral jets

Status: implemented
Date: 2026-10-16

## Problem

The grid-aligned complete water Hessian took 15.4907 s versus GPU4PySCF's
10.8317 s in Slurm 3241. Slurm 3244 profiled the actual complete native endpoint:
434,900 kernel launches and 13.945 s total GPU kernel duration, including
7.2366 s (51.9%) in 46,512 factorized integral launches. This profile is diagnostic,
not clean timing. CUDA API durations include overlapping/backpressure time and
must not be added to the kernel duration or interpreted as pure copy bandwidth.

Each nuclear direction evaluates the exact nuclear J/K source twice: once with
frozen density for the response RHS, then with relaxed density for assembly.
The primitive jets are identical across those calls, but the old source rebuilt
them. Water performed 7,001,316 primitive components across eighteen source builds
and nine Hessian columns. Its 108 induced actions are separate work, not eighteen
response solves. The response remains scalar-column in this change.

## Decision

Retain the complete ordered-group 25-field integral jets for the last right
direction when both a separate 64 MiB cap and the total arena budget admit it.
The basis is immutable under a live owner/stream/device lease. Compare the full
direction on device, not a pointer or hash. Density and density response are not
cache keys: every action performs fresh compiler-generated projection and
contraction. An independently prepared streaming owner provides a bitwise parity
control. Reserve mandatory DF/ECP consumers first, then optional exact storage.

The pool stores one canonical source per ERI eightfold shell-symmetry class.
Choose the lexicographically first shell tuple, guaranteeing that the source is
built earlier in the same ordered stream. Projection remaps Cartesian strides
and both gradient/HVP center slots; permuting scalar values alone is incorrect.
Method/radial parts remain separate keys. Streaming still evaluates all ordered
groups. Conservative ordered work limits are unchanged, but receipts report the
selected unique build work. A first direction-only diagnostic (Slurm 3255) took
11.6843 s versus 15.5402 s old native and 11.2834 s GPU4PySCF; it motivated this
additional symmetry reuse, not a standalone accepted speed claim.

The original bounded shell-quartet cache remains the fallback and minimum
admission. A zero resident cap disables reuse. Reprepare, failed construction or
failed publication cannot mark a new pool valid. Input and live-owner checks
still run on cache hits. Only existing scalar admission/status data reaches the
host; no numerical panels are downloaded and no Python/JIT enters production.

Fuse the four incidence reductions, atom vectors and energy into one CTA per
ordered group. Barriers retain incidence order even when targets overlap. Each
output still has one deterministic owner; no floating-point atomics, changed
precision, screening or scientific expression is introduced. Canonical primitive
accumulation may differ from ordered accumulation at FP64 roundoff; public
incidence summation order is unchanged.

## Rejected alternatives

- Basis-only residency: incorrect because twelve fields are gradient JVPs and
  depend on the right direction.
- Direction-pointer identity: incorrect when a caller overwrites an allocation,
  and misses reuse across equal values in different allocations.
- Reusing projected results: incorrect when D/dD changes after response.
- Unbounded full molecular residency: changes admission/scaling; keep explicit
  memory and streaming limits instead.
- Hiding convergence checks, weakening matrix gates or patching reference
  kernels: none addresses the redundant native work.

## Qualification and consequences

The component probe checks equal directions in distinct allocations, changed
D/dD, in-place direction changes, restore, explicit disable and exact-minimum
budget fallback. Independent displaced libcint values qualify RKS/UKS and
Cartesian/spherical s/p/d components. Molecular tests compare complete resident
and streaming Hessians at `3e-12` absolute/relative tolerance for direct and
coupled DF/ECP RKS/UKS owners. Component resident/streaming gates use the same
`3e-12` tolerances to allow symmetry-related primitive summation roundoff; reuse
of identical direction values under one fixed reference is bitwise.
Full endpoint timings, raw matrices and executed work must accompany any speed
claim; counters or this profile alone are insufficient. Optional storage trades
bounded memory for avoided recomputation; large closures still stream.

Revisit batching directions or retaining direction-independent derivative
ingredients only with a new work/storage plan and complete scientific gates.

## Source-matched complete endpoint evidence

After PR #2527 merged, rebuild both merged master
`f37f96a1c3fee0da04ea0cc2c490f04140d65ae6` and integrated candidate
`5aa703721fdce6a220838676af615d574f0fed05` with the same full default AOT
inventory, CUDA 12.9/sm120 and verified ccache launchers. This avoids attributing
concurrent master changes to the optimization. The candidate compiled identity
matches a clean Git archive; native linkage contains no Python runtime.

Slurm 3265 reserves node2 exclusively and exposes one RTX PRO 6000 Blackwell to
the nested benchmark step. One first-observed and five primed, interleaved fresh
processes per engine/case produce 36 complete startup/SCF/Hessian/JSON/teardown
endpoints. Keep first-observed samples; exclude only those from the medians.
The native numerical child cannot discover Python or compilers, and PTX JIT is
disabled. Canonical WB97M-V, self-consistent VV10 and moving primary/NLC/CPHF
8/4/8 grids are matched. This is a coarse-grid small-system latency diagnostic,
not chemical grid convergence or a large-system scaling claim.

| Case | Baseline native | Optimized native | GPU4PySCF |
| --- | ---: | ---: | ---: |
| H2 | 1.76805 s | 1.66781 s | 5.42347 s |
| Water | 15.49040 s | 8.67943 s | 10.93261 s |

Water uses 44.0% less complete endpoint time than baseline native and 20.6% less
than GPU4PySCF. Its primitive component count is 7,001,316 versus 793,152;
nine source builds and nine resident hits retain 217,856 optional bytes.
Nine Hessian columns, 20 SCF iterations, 108 induced actions, 39,690 one-electron
components and 81 grid mixed contractions are unchanged. Owner-accounted storage
changes from 4,771,912 to 5,049,928 bytes, not total process/device memory.

Every baseline/candidate raw matrix pair passes `3e-12`, with water maximum
`6.8834e-15`; every candidate/GPU4PySCF pair passes `3e-6`, with water maximum
`3.6033e-8`. Every final candidate matrix also passes the unchanged full-matrix
independent-energy gate (`8e-7` absolute, `1e-7` relative). The latter reuses
unchanged values-only reference stencils from Slurm 3240/source17 and does not
claim those reference energies were recomputed on the final source.

The retained historical source17/source616 profiles (Slurm 3244/3262) explain
the removed work: 434,900 to 196,813 launches, 13.945 to 7.183 s total kernel
duration, and 46,512 to 4,644 factorized integral launches. These are explicitly
pre-master diagnostic executions, not final-source timings. Grid AD pairs remain
35,316 launches/about 2.14 s and motivate future work rather than an unmeasured
claim here. The authenticated raw measurements, gates, build receipts and
reproduction scripts are in
`benchmarks/results/wb97mv-hessian-direction-reuse-20261016/`.

Slurm 3266 qualifies the final source-matched binaries on RTX PRO 6000:
50 independent component/lifecycle cases, four resident/streamed full-Hessian
pairs, four actual C++ CLI/API cases (including coupled UKS/DF/ECP spherical
f/g), and three additional independent molecular HVP cases. Retain the four
raw parity pairs as well as their unchanged `3e-12` absolute/relative gates.
The entire 35-case stationary suite is not claimed as rerun. Slurm 3267 passes
memcheck, initcheck, racecheck and synccheck without errors or race warnings.
The separate host regression passes 833 tests with 12 opt-in skips; these are
not substituted for real-GPU gates. The evidence tests independently recompute
the raw matrix checks, work invariants and all timing ratios without requiring
that a future timing run be favorable.

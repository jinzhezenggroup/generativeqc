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

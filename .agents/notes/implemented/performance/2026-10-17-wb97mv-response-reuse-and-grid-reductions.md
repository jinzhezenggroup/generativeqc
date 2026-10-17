# Bounded WB97M-V response reuse and grid reductions

The subsequent capacity investigation and bounded-source scheduling decision
are recorded in `2026-10-17-wb97mv-bounded-source-tiles.md`. They supersede the
old capacity boundary, not the source-matched small-system measurements here.

## Problem and acceptance boundary

PR #2590's complete CLI endpoint advantage did not establish a Hessian-phase
advantage over GPU4PySCF. The water-dimer Hessian remained slower. This change
targets response/integral work and grid execution rather than startup, SCF
tolerances, quadrature or accuracy gates. Compare native preparation **plus**
Hessian execution with the reference Hessian, separately from complete endpoints.

Keep compiler AD, native CUDA numerical execution, UKS coupling, direct/DF,
effective-charge/residual ECP and Cartesian/spherical contracts. No displaced
SCF, CPU/PySCF numerical fallback, runtime Python or runtime compiler is added.
The comparison remains a deliberately coarse STO-3G diagnostic-grid experiment,
not chemical convergence or a general large-molecule performance promotion.

## Repeated-RHS CPKS representation

An optional 128 KiB operator stores physical unit-rotation columns in FP64.
It borrows the exact immutable frame/provider/device/stream; independent live
authentication remains mandatory. Allocate it only after mandatory arenas, under
the existing response and aggregate budgets, before optional source retention.
Zero capacity, larger rotation spaces and optional allocation failure keep
physical GMRES. A standalone HVP does not pay dense construction cost.

GMRES controls and its unpreconditioned tolerance are unchanged. A cached
operator can never alone certify a physical solve. Re-evaluate the compiler's
stationarity RHS using the final physical Fock of the complete metric-plus-
rotation density motion, and subtract the physical gap times the solution.
This is the true orbital stationarity residual; it reuses the physical Fock
already required by the weighted-density tangent. A deliberately nonlinear
provider must fail this gate, even if its sampled matrix solve converges.
For the dimer, expensive induced Fock actions fall from 406 to 76: 40 physical
columns and two physical actions for each of 18 nuclear directions. Every
direction still has an independent physical residual check.

## Independent primitive chains

Replay the existing scalar AD callbacks on at most 32 internal streams, with
distinct record/raw/projected/candidate slots and at most 2048 tasks. Optional
storage is capped at 64 MiB **inside** the original one-electron arena cap.
An ordered gather preserves job/radial/component/pair/center accumulation order,
including separate coincident-center additions. Do not replace this fold with
floating atomics or reassociated atom subtotals. Input directions and weights
are staged; graphs retain no caller pointer. Scalar CUDA remains the bounded
fallback. No recurrence mathematics or callback ABI changes.

## Affine scheduling and the failed prefetch alternative

Fuse only single-consumer, same-output-coordinate definitions; retain every
intermediate finite check and original contraction loops. Output definitions
and shared definitions remain materialized. Preserve duplicate operands in
nonfused contractions. Workspace admission and semantic term counts retain the
original complete program, including primal validation.

One-lane scalar mixed contractions spent about 0.5 s loading/summing only
13,824 elements per contraction. Cooperative shared-memory **prefetch with the
same single-lane sum did not remove this bottleneck**. Retain its negative
profile/pilot instead of repeating that approach. The selected schedule uses
a fixed 128-lane FP64 tree. This deliberately reassociates reductions and must
pass independent numerical gates; it is not a bitwise-equivalence claim.

Fusing all potential contractions into one per-output serial kernel exposed
another low-occupancy cliff. Single-axis AO-potential reductions instead use one
cooperative block per output; these children are excluded from epilogue fusion.
Mixed geometry/feature contraction graphs stage their inputs into bounded
workspace and replay only arithmetic. Authentication, input validation and
transactional publication stay outside capture. Instantiate graphs in prepare,
not during an action; resource failure retains ordinary precompiled launches.

## Qualification and evidence

Keep all intermediate and failed harness attempts. Early remote runs omitted
the compiler's pinned Libxc input and checkout marker; their failures are not
numerical rejection and do not count as passing qualification. A subsequent
deployment must contain the actual compiler inputs and root marker.

Numerical gates cover independent libcint one-electron derivatives, displaced-
SCF coupled CPKS, independent Libxc/VV10 moving-grid values, RKS/UKS direct and
DF/ECP/spherical complete-endpoint fallback parity, and complete matrices against
GPU4PySCF. Use finite Slurm jobs and preserve assigned visibility. The original
180 s endpoint deadline and 24 GiB sampled total-device watchdog remain unchanged.
Historical measurement JSON may be losslessly compressed with original Git-blob
and decoded-byte receipts to retain the strict 64 MiB evidence budget; do not
drop negative samples or enlarge that budget to make a result fit.

## Source-matched observations

The [retained campaign](../../../../benchmarks/results/wb97mv-hessian-response-reuse-20261017/README.md)
binds the final rebuild to clean source `51a87a04fdb8ef8e17bd8f6b848f2fcdc77f60af`,
rebased onto merged PR #2590 (`743c06bf80db5f04b43d78cb6b3a3ea423577c73`).
Job 9950 on n1's Slurm-assigned RTX 5090 interleaves first observed and three
artifact-primed fresh-process runs per engine/case. Compare the median of each
native preparation-plus-Hessian sum, not the sum of independent medians:

| Case | Native prepare + Hessian (s) | GPU4PySCF Hessian (s) | Reference / native |
|---|---:|---:|---:|
| water-dimer | 2.643700 | 2.873015 | 1.087x |
| formaldehyde | 1.102402 | 7.359160 | 6.676x |
| water | 0.446459 | 2.449993 | 5.488x |
| H2 | 0.128532 | 0.456874 | 3.555x |

All complete paired matrices pass the original 3e-6 absolute gate (maximum
3.652e-8 hartree/Bohr²); all four previous-native matrices for each case also
pass the original 3e-12 absolute/relative parity gate. Keep the modest dimer
margin visible; this is not evidence of a universal or statistically general
advantage. The grids remain coarse, spherical STO-3G diagnostics. UKS/DF/ECP
qualification does not turn these direct-RKS timings into UKS/DF/ECP rankings.

Total-device sampled native memory is about 16.8 GiB versus about 2.7 GiB for
the reference, inside the unchanged 24 GiB limit but not a memory-efficiency win.
Final native-only ethane/methanol endpoints complete in about 8.9/5.3 seconds;
their independent paired reference attempts failed SCF under unchanged controls,
so no larger-case speed ranking is justified. Formal broad performance promotion
remains unavailable without converged-grid/scaling evidence, clean compilation
cost and an isolated exact GPU peak. Preserve the final qualification, sanitizer,
fallback parity, timeout and failed-harness receipts alongside these observations.

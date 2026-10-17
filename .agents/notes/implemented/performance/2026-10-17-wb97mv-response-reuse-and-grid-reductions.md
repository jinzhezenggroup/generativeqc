# Bounded WB97M-V response reuse and grid reductions

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

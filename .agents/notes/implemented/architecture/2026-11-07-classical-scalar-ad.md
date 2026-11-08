# Decision: share compiler-AD classical scalar derivatives

Status: implemented source migration; full-method qualification and retirement incomplete
Date: 2026-11-07

For the subsequent ownership census and source-migration closure, see
`2026-11-08-derivative-retirement-closure.md`. Earlier qualification states
below describe their original experiments, not the final campaign.

## Problem

Nuclear repulsion had three derivative authorities: host analytic gradients,
CUDA runtime dual numbers, and a handwritten restricted-Hessian block. D3/D4
also repeated Gaussian-normalization, switching and pair derivative formulas
around otherwise reusable native traversal. Moving handwritten derivative text
into a generator would not remove the independent mathematical maintenance.

## Decision

- Define the nuclear pair energy once and derive its gradient and directional
  Hessian through the existing scalar Graph. Native host/CUDA consumers keep
  pair traversal, ionic charges, accumulation and force/gradient signs.
- Move existing HVP consumers and Hessian AOT inventory to that owner directly;
  do not retain an import facade or compatibility alias.
- Generate Gaussian, normalization, smooth-switch, coordination, BJ damping
  and D4 charge-scaling responses from primal scalar programs. ATM now also
  derives physical-sign edge/C6 responses (including the GFN2 absolute-product
  energy branch), D3 zero damping derives regular and independently scaled
  overflow responses, and gCP differentiates its actual Slater recurrence and
  damped pair primal. Keep the native
  schedulers and table/resource owners, rather than promoting a different
  GeometryIR scheduler just to remove the local formulas.
- Reuse IntegralIR's neutral scalar Graph/emitter through narrowly enumerated
  structure-audit clients. This permits neither integral recurrences nor method
  policy in the generic compiler services, and introduces no second AD algebra.
- Emit derivative-free scalar specializations for energy-only D4 calls. Retain
  existing normalization fallback, charge saturation and finite-input behavior.
- EEQ retains native assembly, LU solves and transactional publication. Its
  compiler-owned RHS is differentiated before solving; the exact regularized
  derivative of `kcnchi*CN/sqrt(CN+1e-14)` replaces the former approximation
  `0.5*kcnchi/sqrt(CN+1e-14)`. Cache that per-atom response rather than repeating
  its evaluation in every matrix/scatter pair. Independent charge/force fixtures
  and energy finite differences must continue to qualify this numerical change.
- The gCP near-equal-exponent branch differentiates the complete finite B series,
  including its small-argument limit, not a constant table substituted for that
  series. Keep the existing `abs(za-zb)<0.1` admission and pinned shell products.

## Scope and remaining work

This is **not** completion of the repository-wide handwritten-derivative
retirement request. At minimum the following owners still need migration and
their own independent acceptance gates:

- dynamic CPU integral and CUDA Dual/Boys scalar migration is now recorded in
  `2026-11-07-integral-ao-scalar-ad.md`; generic recurrence traversal remains
  production and needs complete endpoint/fallback qualification;
- VV10/rVV10 local-scale, pair and row-composition partials still include
  manually maintained formulas, even inside compiler emission;
- ECP quadrature's Newton slope still uses an explicitly supplied identity;
- full XC point consumer/molecular performance qualification (the Graph-derived
  scaled-point arithmetic and exchange migration is recorded separately);
- remaining GFN2 periodic response and method contractions
  require a complete ownership audit rather than filename-based deletion.

Independent reference implementations and pinned upstream fixtures must not be
deleted to make the source census look complete. No full-retirement PR is ready
until remaining owners and real-device gates are resolved.

## Evidence

- Compiled-host scalar tests compare generated responses with independent
  energy finite differences, including charge saturation, ionic charges,
  additive gradients, translation invariance and invalid geometry/shape.
- Existing D3 ATM/zero and D4 fixed-charge/EEQ native test binaries execute
  against pinned upstream fixtures and complete finite-difference gradients.
- The initial focused compiler/native-host/Hessian suite passed 104 tests, including
  nine complete native BJ fixtures. The separate provenance, source-registry
  and D4 manifest suite has 27 passes and two failures because this checkout
  lacks the registered upstream GFN1 source cache. An explicit cache sync times
  out after 120 seconds; no source-integrity check is bypassed or removed.
- The changed CPU integral translation unit compiles through the owned finite
  cache runner. Nuclear, D3 and D4 CUDA translation units compile for `sm_80`
  with CUDA 12.4; this is not RTX 5090/`sm_120` qualification or device execution.
- The compiler structure audit accepts only the named scalar-client edges.
- CPU CMake source generation is checked with an explicit verified sccache
  launcher. Cache statistics and local validation logs belong in ignored local
  artifacts, not in tracked benchmark publication history.
- The first local Slurm probe exposed only drained `node3`. That is not a global
  GPU-unavailability conclusion: SSH aliases subsequently exposed allowed nodes
  n1/n2/n4/n5. Real qualification now runs on n1/node1 through finite `srun`
  allocations (`main`, `gpu:5090:1`), CUDA 12.9.1, sm120, driver 580.95.05.
  Node3 is never used and scheduler device visibility is preserved.
- The expanded focused host/compiler/Hessian suite passes 171 tests; the
  standalone CPU Release CMake tier passes all five classical oracle tests.
  The n1 actual D3 ATM/zero, D4 fixed-charge/EEQ and gCP CUDA test binaries pass
  their independent numerical, ragged, lifetime and failure-isolation gates.
  The gCP complete MB16 gradient finite-difference error is about `3.2e-12`;
  fixed-charge D4 independent energy/gradient/charge errors are below `3e-19`.

## Performance qualification and rejected first emission

The first generated emission regressed D3 correction endpoints by roughly
1.6--1.8x. Do not accept generated ownership as sufficient performance evidence.
The diagnosed causes were generic libm integer powers, independently emitted
lazy branches recomputing the same exponential, and CPU out-of-line hot scalar
helpers. Fixes retain Graph differentiation: lower small integer powers,
parameterize dynamic positive powers logarithmically before binding the native
`pow` primal, coalesce simultaneously demanded branches without evaluating
inactive arms, and use a CPU-specific hot-inline policy without forcing CUDA
inlining. Normalized quotient JVPs are factored without squaring tiny norms.

`benchmarks/classical_derivative_endpoints.cpp` measures real D3/D4/gCP plans:
construction+execution+destruction, unchanged replay and changed geometry,
including synchronized host publication. It records pair/triple work counts,
EEQ dimensions, 7 samples of 3 repeats, and checksums. Three interleaved
master/candidate rounds use 4/24/64 atoms and 8/8/2 systems. Master is exactly
`b40b57be5`, with the identical benchmark driver. This is correction-endpoint
qualification, **not** complete SCF/force qualification.

The preserved `candidate-coalesced` stage has CUDA warm candidate/master ratios
of about 0.84--0.90 (BJ/ATM), 0.97--0.98 (zero), 0.81--0.88 (D4), and 0.64--0.68
(gCP). CPU D4 is about 0.95--0.96 and gCP about 0.48, but CPU D3 zero remains
about 1.01--1.02 and the smallest BJ/ATM case about 1.02. These residuals are
not silently declared a passing no-regression gate. A subsequent reciprocal-
power lowering is being measured separately; do not overwrite earlier evidence.

Ignored logs, source snapshots, compiler-cache statistics and JSONL live under
`/home/jzzeng/generativeqc-qualification/classical-ad-20261107-b40b57be5` on n1,
with fetched copies in `.artifacts/classical-derivatives/performance`. Each stage
uses a new directory; executables are never relinked while timed. Build workers
use verified sccache 0.16.0 with owned finite compiler processes. No cache is
cleared and no release/asset publishing is involved.

## Rejected alternatives

- Keep runtime dual-number nuclear derivatives: retains a separate execution
  authority when all required primal/derivative roots already fit a scalar DAG.
- Replace all native execution with a new generated scheduler: changes resource
  and endpoint behavior unnecessarily and bypasses existing promotion gates.
- Delete fallback/oracle implementations before replacement qualification:
  violates numerical independence and bounded complete-endpoint contracts.

## Revisit when

Complete the remaining ownership audit, migrate each scientific owner from its
primal authority, and qualify changed CPU/CUDA complete endpoints, including
changed geometry and bounded fallback/failure behavior, on allowed Slurm GPUs.

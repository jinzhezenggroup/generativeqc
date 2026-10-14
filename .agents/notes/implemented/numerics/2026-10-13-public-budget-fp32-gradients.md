# Decision: application-budget FP32 stationary gradients

Status: implemented
Date: 2026-10-13

## Problem

The broad candidate narrowed much more than an ERI recurrence, but remained a
private force endpoint. Its native FP64 comparator used serial point workers;
speedups against that comparator did not prove a public Fast improvement.
Merely connecting the private controller would make some public workloads
slower. The user clarified that the objective is useful FP32 within Fast's
application allowance, not uniform FP32 at the expense of correctness.

## Decision

Generalize the shared retained/tiled cooperative Becke traversal to typed
arithmetic. Both native strict and mixed owners use the same bounded cooperative
schedule when actual device limits allow it, otherwise retaining serial workers.
Reuse the existing charged double scratch arena for float point-local sums.
Recover translation from the same rounded atom contributions before widening;
do not independently round the cancelling atom/grid channels. This is a schedule
of the existing emitted equations, not a second Becke or derivative definition.

Public Fast admits the existing complete native PBE0/RKS gradient for exact,
all-electron full-range models with 24–384 AOs through d shells and 2–48 atoms,
partition iterations three, and allowances at least 1e-6 Eh energy, 1e-5 Eh/bohr
maximum component and 3e-6 Eh/bohr RMS if requested. The linked binding must match
the full scientific model, not just a functional label. The same immutable
admission serves preparation resource metadata and execution. Unrelated methods,
UKS, fitted models, corrections and unsupported budgets retain their old paths.

Ordinary FP64 promotes the PBE0/RKS d-shell native consumer through 32 atoms. A
preliminary public comparison against the incumbent Python force executor is
retained only as diagnostic evidence: most of its small-system gain was executor
overhead, not FP32. Above the retained-pair regime the incumbent phased strict
consumer remains faster than native inline cooperative FP64 (the independent
48-atom comparison is retained). Keep that faster strict path; do not force a
slower identical scheduler just to inflate Fast ratios. Final timings compare
the best qualified public strict path, not an intentionally slow baseline.

The gradient request enables weighted orders 0–5, float grid jets/projection/
features, interior PBE X/C, AO pullbacks and Becke local sums. Global reductions,
one-electron/Pulay, Boys, normalization, higher orders and unsafe tails remain
FP64. The separate SCF warmup now also narrows PBE exchange, using the already
shared full-X/C point evaluator; its domain remains 24–256 AOs. Exact SCF J/K and
strict final convergence remain unchanged. A 384-AO Fast force therefore need not
use an approximate SCF trajectory.

## Publication and fallback invariants

- The resident final density and stationary weights must authenticate the exact
  current FP64 final-state token before and after the complete force operation.
- Optional float grid storage never displaces the strict bounded owner; failed
  allocation retains FP64 grid production and reports zero mixed grid points.
  Public preparation reserves the float panels **and a second contraction
  provider** in the cached-consumer tier. The strict minimum remains unchanged.
  The v1 receipt's cached stationary-consumer cost includes this optional producer;
  the emitted stationary arena does not grow. An initial public test caught a
  zero-headroom bug: passing only strict cached capacity made every float grid
  request fall back. Shared `cuda_grid_fp32_storage` now owns both the planner
  formula and actual allocator sizing; never fix this by bypassing the ledger or
  reducing the test's required mixed-work count.
- A nonfinite weighted task retries in FP64 before scatter. A numerical geometry
  failure retries the entire gradient once at the same token. Driver, stale-token
  and resource failures must not masquerade as precision retries.
- No clipping, oracle density, CPU reference or modified quadrature enters the
  production force. Independent references belong only to qualification/audits.
- The private versioned five-word gradient report travels with the completed
  result, separately from the stable SCF census ABI. Getters preserve context
  locking, bounds and unavailable status. Energy-only replay cannot inherit it.
- Report attempted mixed grid/geometry points and whole-gradient retries. Clean
  calls do not collect intrusive weighted task counters: unknown retry counts
  stay unknown, never zero. Returned double arrays are not an arithmetic census.

## Budget semantics

An allowance selects independently qualified arithmetic, not a rigorous bound
for every arbitrary molecular geometry. `qualified_mixed_gradient` must not be
reported as strict forces or `observed_met`. Explicit `TargetAccuracy` still
requires a complete independent FP64 endpoint and, after rejection, a third
tighter verification before returning FP64. Include all of this cost in timing.
No Hessian/TS/IRC/NVE qualification follows from single-point force agreement.

## Rejected alternatives

Do not present the private serial-FP64 ratios as public speedups, withhold the
cooperative schedule from strict users, or make J/K FP32 by toggling the narrower
recurrence-only J policy. Do not widen the SCF warmup just because the gradient
domain is wider. Do not infer per-call observable errors from qualification gates
or SCF residuals.

## Evidence and reproduction

Executable host tests cover double/float retained and tiled cooperative paths,
AO maps, empty/tail workers, translation cancellation, guard regions, sticky
failure, phased strict launches and typed creation cleanup. The serial typed
science has independent Decimal Becke/Libxc checks; scheduled complete molecular
tests compare against independent PySCF moving-grid gradients. Public tests cover
budget admission, explicit target rejection/verified fallback, retained geometry,
seeds, provenance lifetime and transport bounds.

The source-matched public comparison, semantic work and numerical gates are in
`benchmarks/results/pbe0-budget-fast-v2-20261013/`. Raw successful and failed runs,
build/cache logs and source snapshots remain in the ignored campaign of the same
name. Use `benchmarks.pbe0_fast` for public timings, not the private gradient probe.

This supersedes the unpromoted-candidate conclusion in
`2026-10-13-broad-fp32-stationary-candidate.md`; its earlier measurements remain
historical evidence, not measurements of this implementation.

## Master integration

The milestone PR integrates master `f2d59d9f3c8f622954ca34b90ac72f8de7dea595`.
Its automatic Modified Huckel seed remains identical for strict and Fast;
MINAO compatibility and explicit choices retain their own resource contracts.
The shared typed point algebra keeps master's subnormal PBE value/response
repair rather than restoring the earlier restricted-only ratio switch.
Historical campaign timings and source/library hashes are unchanged; they do
not measure this integrated tree or the newer automatic seed. Integration
qualification is recorded separately from the original numerical campaign.

## Revisit when

Expand numerical/size/spin domains only with independent complete-endpoint
acceptance, conditioning/geometry coverage and resource bounds. Future SCF J/K
lowering must include contraction/accumulation and resident-cache precision
identity, not just recurrences. Per-geometry audit avoidance and NVE require
separate error, continuity and energy-drift evidence.

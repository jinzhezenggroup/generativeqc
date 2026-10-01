# Decision: retain Becke center geometry between stationary point contractions

Status: implemented
Date: 2026-10-01

## Problem

The shared compiler-emitted Becke reverse composition calculated the same robust
center-center norm and its unit-vector pullback in both pair passes of every
point. Its local ratio AD also repeated geometry-only reciprocal/power nodes.
Local scalar CSE cannot move those operations across point invocations. The
bounded stationary lane schedule from #1672 fixes concurrency, but does not remove
this source-work amplification.

## Decision

Retain canonical triangular `(a > b)` records containing the existing scaled AD
norm `[R, dR/dx, dR/dy, dR/dz]` and the two geometry-only nodes cut from the existing
ratio AD graph. The cut retains `1/R` and `-pow(R, -2)` with the same arithmetic
order. It does not replace the latter with `-(1/R)^2`. Both the direct and prepared
routes enter one shared pointwise traversal. All local scalar mathematics remains
compiler-owned and the direct helper remains available to other consumers.

CPU's stateless contraction prepares once per ABI call. The stationary CUDA owner
prepares on device once per successful reset and shares the table across every
point/tile/lane until the next reset. Each independently owned composite source
arena retains its own geometry; this change does not add a cross-owner lease.
The native value-quadrature inverse-distance buffer is not borrowed: it is private
to that owner, uses `hypot`, and has coincident/tolerance sentinel semantics that
differ from the derivative's strict nonsmooth-geometry rejection.

## Resource and lifetime invariants

- Storage is exactly `48 * natom * (natom - 1) / 2` bytes, at most 390,144 bytes
  under the stationary owner's 128-atom cap
- CUDA chooses existing point lanes first. Only remaining budget admits the table;
  it cannot steal lanes or make a formerly admitted minimum budget fail
- CPU budgets include the optional table beyond existing staging/scratch. Tight
  budgets or table allocation failure keep the direct O(natom) scratch route
- A typed CUDA allocation failure destroys the partially prepared owner before
  retrying the admitted direct arena once. Other runtime errors do not retry
- The compiler plan is an upper bound after an OOM downgrade. Native metrics are
  authoritative for actual owned bytes/cache retention
- Both reset entry points drain pending geometry before uploading new centers,
  prepare/validate on the owner's stream, and synchronize before any borrowed
  stream can consume the records. A failed reset poisons the owner until a valid
  reset. No geometry pointer or identity is borrowed from another owner
- Preparation adds no H2D/D2H traffic and no extra kernel launch: it extends the
  existing center-validation kernel. Center coordinates retain the existing upload
- Pair traversal order, clipping, zero-product branches, point-center collision
  rejection, and per-point arithmetic are unchanged. Geometry-only ratio overflow
  is not rejected early, so saturated/zero-seed behavior follows the original path

## Work accounting and evidence

For `P = natom*(natom-1)/2` and `G` valid points, a direct reset/call evaluates
`P + 2*G*P` center norms; the prepared route evaluates `P`. Both still evaluate
`G*natom` point-center norms and perform the same two Becke pair passes. The
existing pair-visit admission bound is intentionally unchanged. Native counters
separately expose center-distance evaluations and cache preparations; source work
is not a measurement of device throughput or complete-endpoint speedup.

Host executable tests compile the actual emitted helper and ratio cut, compare
cached/direct output bits for iterations 1–5, exercise ragged/empty tiles and
changed geometry, and count norm calls. The integrated generated geometry-kernel
host test uses actual Becke code with AO/XC stubs to check the pointer/indexing
seam. Existing independent Decimal/finite-difference grid tests remain the
scientific oracle. Resource tests compile actual native allocation/create code,
including exact cache threshold, arena end/error boundary, post-allocation typed
OOM cleanup, and non-allocation failure behavior.

The opt-in NVIDIA test extends #1672's lane/replay matrix with cached and direct
owners, both reset entry points, exact work counters and unchanged upload bytes.
Before promotion, run the exact integrated head through the repository's Slurm
stationary validation runner, including independent full-force oracles and
cold/warm/changed-geometry endpoint measurements. A fresh CPU build passed 57/57
native tests; 13 full-force independent analytic/finite-difference and late-state
checks passed. The broad host suite passed 676 tests with 49 CUDA/opt-in skips
before the final allocator regression was added. No NVIDIA runtime result or
speedup is established by host execution.

## Rejected alternatives and follow-ups

- Reimplementing norm/ratio derivatives by hand would create a second scientific
  owner and could alter underflow, signed-zero or overflow behavior
- Retaining an uncharged O(natom²) table or reducing point lanes to fit it would
  conceal the resource tradeoff; optional bounded fallback is explicit instead
- Reusing point-dependent pair values, logs and normalized products from the
  forward pass is a separate slice with a stronger lifetime/storage contract
- Locality screening is outside this exact optimization and requires an explicit
  force-error bound before any approximate production path is considered

References: #1479, #1672; `test_becke_center_geometry.py`,
`test_grid_native.py`, `test_stationary_geometry_resources.py`,
`test_stationary_geometry_kernel_host.py`, `test_dft_complete_cuda.py`

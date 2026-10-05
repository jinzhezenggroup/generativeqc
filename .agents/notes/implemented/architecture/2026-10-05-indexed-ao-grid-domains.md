# Decision: preserve independent indexed AO axes across grid consumers

Status: implemented compiler representation and ordinary-force binding; full P0-C qualification pending
Date: 2026-10-05

## Problem

Issue #1893 requires compiler-visible local AO/grid domains, not only a runtime
active-map switch or a census of admitted work. TensorIR's existing runtime
indexed selection zips coordinate vectors into one leading domain. Expressing
`D[I,I]` that way requires AO-square coordinate tables; sequential one-axis
gathers can instead materialize a local-by-global intermediate.

## Decision

Add generic Cartesian selection/scatter primitives that preserve independent
local axes and share one immutable int64 map on both AO dimensions. Their
runtime values are not static source identities. Gather materializes the local
result directly; scatter is its exact transpose, including repeated indices.
Reference and generated AD, serialization, generated CUDA, and precision
lowering retain that contract. Scientific traffic excludes integer map values
from floating-point work. FP32 scatter requires explicit reduction qualification.

`dft.indexed_layout.AoGridBlockLayout` is data-only: shape, point interval,
basis identity, evaluated jet order, map capability, and optional binding epochs.
It owns neither a second map array nor device storage. Evaluating second jets
does not certify a map discovered using first jets. Unknown capability remains
unknown; dense identity domains need no sparse-map qualification.

`method.indexed_grid.AoGridBlockProgram` owns graph composition for density
gather, local projection, and potential scatter. The method composition layer
may now consume DFT's data contract, while DFT still cannot import TensorIR and
TensorIR still cannot import DFT. The explicit method-to-DFT dependency is
reflected in the structure checker and current architecture documentation.

The existing resident AO cache's `select_block` returns its original immutable
array plus this descriptor under the same grid lock. No numerical cache or
retained index copy is added. Typed CUDA admission checks basis, dimensions,
evaluated order, derivative capability, and supplied epochs before enqueue.
The ordinary stationary consumer requires first jets for rho-only ingredients
and second jets when first spatial ingredients contribute. A lease exposes
the producer's same descriptor and still expires at its existing lifetime gate.

## Evidence

The host campaign passes 306 tests and skips five explicit device opt-ins. New
tests cover empty/identity/subset/repeated maps, independently sized axes,
independent NumPy projection/scatter oracles, reference/generated AD, replay,
shared map accounting, invalid coordinates, int64 precision controls, dense
fallback, derivative capabilities, epoch admission, and lease expiration.
Compiler ownership checks cover 443 modules with zero dependency errors.
Focused lint/format and diff checks pass.

The frozen source for device qualification is
`81a92efbc9c1999a3444d0dee2ca0a9ace67a4e5dd774089b8c8f6c6b874d7f0`.
n1 Slurm 6046 failed before device execution because the freeze command omitted
the required source-root argument and wrote an empty expected identity. After
confirming that exact handle was FAILED, correct only the manifest and submit
6049 for finite RTX 5090 generated/resident correctness, memcheck, and racecheck
qualification. Retain the failed harness receipt; do not count it as a numerical
failure or as completed GPU coverage.

Slurm 6049 completed with exit 0 on RTX 5090. Three generated/resident consumer
tests pass, covering local density gather, potential scatter, and projection,
including changed/repeated maps, invalid-index rejection, and same-owner
recovery. Memcheck repeats all three with zero errors; racecheck repeats all
three with zero hazards, errors, or warnings. The verified source matches the
frozen identity. These are standalone TensorIR runtime gates, not qualification
of the newly typed native AO/force endpoint or an endpoint speedup. Raw receipts
are retained under `.artifacts/n1-layout-6049/`.

## Remaining Scope

The [consumer-binding follow-up](2026-10-05-indexed-ao-grid-consumer-binding.md)
extends this foundation to spatial and resident nonlocal consumers, removes the
owned restricted-spin route's duplicate density gather, and retains subsequent
qualification and compatibility evidence. The frozen receipts below remain
historical foundation evidence rather than measurements of that later candidate.

This is not a promoted performance optimization or completion of P0-C. Still
required: integrate SCF map producers with the same typed contract; qualify all
RKS/UKS PBE/PBE0 fixed-density E/V and force consumers; compare native/reference
local domains on identical geometries; separately attribute atom gather and
owner reduction; remove demonstrated duplicate execution; retain profitable
dense/tiny/high-occupancy routes; and demonstrate clean complete 48/96-atom E+F
improvement without prohibited cold/moved trajectory changes. Generated generic
scatter is correctness infrastructure, not a claimed profitable default provider.

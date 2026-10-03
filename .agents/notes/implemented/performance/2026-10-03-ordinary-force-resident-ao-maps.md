# Decision: consume local resident AO maps in ordinary stationary forces

Status: implemented; experimental opt-in, no default promotion
Date: 2026-10-03

## Problem

Resident molecular-grid storage avoided point transfers but still contracted
the full density matrix on every force tile. Locality cannot reduce this
`G*M^2` work until selected AO maps reach the feature and geometry consumers.
Increasing the tile size changes scheduling, not this dense work dimension.

## Decision

Use the shared AO-only resident discovery and bounded map-cache owners. The
ordinary caller supplies the exact current geometry/grid domain and passes
selected indices into `feature_task_device_points`. No separate PBE0 producer,
density gather, geometry scatter, or cache implementation is introduced.

The opt-in `resident_ao_cutoff` remains `None` by default. Budget/capability
misses keep full AO membership. Cache reserve fits inside the existing total
host allowance after every already-required owner, including prepared TensorIR
storage; it does not enlarge the host cap. Full-capacity device arenas remain
charged because discovery and dense fallback require them.

## Invariants

- Current native snapshots validate their token before cache lookup; pointer
  equality alone never proves scientific identity or continued ownership.
- Maps cover the configured force derivative order. Ordinary GGA uses order
  two; an SCF order-one map cannot be substituted.
- Domain and grid-generation changes drop old storage before replacement.
  This also handles center rebinding that preserves an initial basis identity
  and failed-execution refresh at an otherwise unchanged geometry.
- AO-only maps can survive density changes, but density subblocks still bind
  the exact final SCF density for every force endpoint.
- Empty maps do not skip point tiles, partition pairs, or geometry consumers.
- Cutoffs are explicit sampled-jet heuristics, not certified force bounds.
  Keep the original independent energy/force gates and every measured repeat.

## Rejected alternatives

Per-tile CPU bounding boxes are not used: sampled outer atom-centered radial
tiles have boxes enclosing the entire molecule despite vanishing sampled AOs.
Pointer-keyed maps without geometry/generation binding are unsafe under moved
geometry and allocation reuse. Charging mean active AO count as device capacity
is also invalid; this first consumer retains full-capacity owners instead.

## Evidence

Caller host tests cover optional-budget clipping, unchanged dense resource
fixtures, default-off forwarding, cache lifetime, stale snapshots, and the
actual production loop with empty/noncontiguous maps and deferred failures.
Native job 5564 passes 77 host/GPU/caller checks and 22 memcheck checks with no
errors. Jobs 5565/5566/5567 retain all 156 complete energy/force calls, including
the zero-budget control. Every same-geometry reference-repeat pair passes the
unchanged gates (maximum `1.069e-10 Eh` / `3.322e-11 Eh/Bohr`). Full evidence,
exact source/binary identities and executable verifier are retained under
`benchmarks/results/pbe0-resident-active-ao-20261003/`.

At unchanged tile/budgets, warm 24/48/96 improves by 4.79%/13.12%/18.60%.
The 96-atom complete endpoint is 77.143137 to 62.791405 seconds; force GM² is
5.8249% of dense, mean/max active AOs are 164.516/553 of 768. Allocated capacity
is still dense. First-use discovery itself takes 21.638 seconds, while every
warm replay has zero discoveries; changed geometries rediscover all maps.
This establishes useful work reduction with amortized AO-only metadata, not a
universal asymptotic claim or closure of the overall reference gap.

At 24 atoms moved time worsens from 27.224856 to 27.670800 seconds with equal
SCF iteration counts; small-system moved regressions also remain. All variable
iterations and ordered/shared-cache cold observations are retained without
normalization. The later master/dependency integration is a different source
identity and does not relabel these frozen measurements.

Capacity qualification continues to bind the full-AO default formulas. It also
checks optional-reserve clipping after prepared tensor ownership, charging
before artifact lookup, current geometry/token binding, and prepared request
policy. New mutation controls ensure that moving these checks into helpers
does not escape the fail-closed audit.

## Revisit when

Promote no default until same-binary complete endpoints validate numerical gates,
discovery cost, warmed reuse, and moved-geometry invalidation. Reducing allocated
AO capacity requires a separate complete-owner plan based on maximum admitted
map size, not the point-weighted average. Shared tile planning and SCF local-AO
integration remain distinct changes.

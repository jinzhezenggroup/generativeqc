# Decision: reuse metric derivatives across auxiliary pair orientations

Status: implemented
Date: 2026-10-25

## Problem and decision

The generic metric force consumer evaluates the same physical derivative for
both `(P,Q)` and `(Q,P)`. A complete contiguous metric weight panel now visits
`Naux*(Naux+1)/2` unordered pairs and uses the unchanged generated derivative
policy once for each. It scatters both original weights separately. This
preserves arbitrary nonsymmetric weights and avoids summing them before
multiplication, where two individually valid weights could overflow.

The route requires the full zero-offset panel, contiguous row-major indexing,
cooperative mapping and a single gradient sink. Partial, strided, serial and
distributed-sink inputs retain the existing bounded traversal. Explicit
`GENERATIVEQC_DF_METRIC_DERIVATIVE_PAIRS=0` rolls back; invalid values reject.
No weight tensor, source integral, precision, recurrence or scratch allocation
is added. Dense weight storage and mathematical contributions are unchanged.

## Evidence

The [source-bound capsule](../../../../benchmarks/results/df-metric-pairs-20261025/README.md)
retains three alternating rounds for PBE48/96 and r2SCAN-3c96, both profiles:
36 fresh populations / 144 complete E+F calls. Standard cold medians change
6.137/16.337/34.654 to 6.093/16.120/34.065 s; corresponding warm medians change
1.880/6.070/11.066 to 1.828/5.877/10.554 s. Small cold differences remain
scoped observations, not a universal speed ranking.

Diagnostic work confirms 3,268,864 -> 1,635,336 tasks for PBE48,
13,075,456 -> 6,539,536 for PBE96, and 46,457,856 -> 23,232,336 for r2SCAN-3c96.
Warm metric derivative time is approximately halved. Intrusive trace timing is
kept separate from complete endpoint measurements.

Qualification retains independent arbitrary/skew/one-sided/diagonal weights,
full and split panels, Cartesian/spherical and mixed representations, auxiliary
g, shared and moving centers, range-separated exchange, finite differences and
resource/lifetime controls. Thirty fresh independent original/moved/property
checks include same-owner explicit rollback at both geometries.

Four original broad-suite failures reproduce on the unchanged control. One
assumed 4 MiB must fail although current RHF fits; it is now an independently
checked near-floor endpoint and the 1-MiB rejection remains. Three assumed
auxiliary class 003 cannot use Rys although the current production manifest
qualifies it; per-class manifest checks and independent forces remain intact.

## Other investigated schedules

The independent force-grid trial shows PBE benefits from 512 rather than 256
points per tile, but r2SCAN-3c96 refuses 512 because its additional host workspace
exceeds the budget. Tile 128 regresses all three. Keep the default unchanged
here; a future selector must use actual resource feasibility and preserve the
256-point fallback. All completed results and the refusal remain retained.

## Revisit when

Larger panels and density-fitted meta-GGA endpoints still spend substantial time
in three-center derivatives and grid geometry response. Extend reuse to partial
panels only if both weight orientations have a bounded authenticated lifetime.
Do not silently assume symmetric adjoints, double diagonal weights, relax
resource caps or replace the generated recurrence with native mathematics.

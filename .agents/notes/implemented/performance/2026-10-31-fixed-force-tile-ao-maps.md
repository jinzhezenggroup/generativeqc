# Decision: explicit fitted force tiles retain exact-jet AO maps

Status: implemented
Date: 2026-10-31

## Problem

The fitted exact-jet profile accepted only fixed 256-point tiles, whereas its
budget-auto counterpart accepted any capacity-admitted tile. The new public
`KsOptions.force_tile_points=1024` therefore disabled the map in the generated
Python stationary force consumer, even when automatic selection used the same
1024-point tile. Fast r2SCAN exposed this loss; Standard's native force controller
has separate map admission and was unaffected.

## Decision

Remove the fitted fixed profile's tile-size equality predicate. Both explicit
and automatic requests already pass the complete force-capacity planner. Keep
the exact-jet producer, derivative order, work crossover, `1e-16` cutoff, optional
64-MiB storage allowance, 0.8 occupancy bound, and resource/capability checks.
The actual tile remains part of map lifetime identity. Direct sampled and
pre-AO profiles retain their existing fixed-tile qualification.

No GPU name or architecture allowlist selects this path. Optional allocation,
capacity and occupancy misses still execute the dense derivative domain.
This repairs an explicit option; it does not promote a larger default tile.

## Evidence

The [retained campaign](../../../../benchmarks/results/independent-force-tiles-20261031/README.md)
contains same-binary ABBA ablations at fixed 1024 points for Fast r2SCAN/DF96.
The diagnostic changes only this profile predicate in memory. Each population
includes a complete cold call and three complete warm calls; all twelve pass the
unchanged independent oracle gates. Six-sample warm medians are:

| Device | Dense control (s) | Exact-jet map (s) |
| --- | ---: | ---: |
| A100 | 6.75148 | 5.86981 |
| RTX 5090 | 8.61721 | 5.19095 |
| RTX PRO 6000 | 8.01960 | 4.59324 |

Selected point×AO² work is 67,377,718,336 versus 636,028,452,864 dense work on
each device. Both arms use nine cold and one warm SCF iterations with identical
Fock counts. Maximum paired force differences are below `6.5e-9` Eh/bohr.
These static-geometry timings are an option regression diagnosis, not evidence
that Fast universally outperforms Standard or GPU4PySCF.

Regression coverage exercises explicit and automatic tiles, RKS/UKS, AO-jet
orders one/two, independently converged forces, moved geometry, map reuse,
zero-cache and occupancy fallbacks. Small-fixture test admission changes the
work/occupancy threshold only to exercise the consumer, not to promote a
small-system performance policy.

## Rejected alternatives and remaining limits

Do not rewrite the request as budget-auto to evade the predicate: that would
obscure the caller's explicit bound. Do not apply this qualification to the
different Direct producers or widen mixed-precision force admission.

Large Fast DF calculations can still use the generated force consumer while
Standard uses the faster native FP64 controller. Sharing an authenticated
DF-XC budget capability with native FP64 force admission is a separate change;
removing mixed-gradient shape guards without that contract is unsafe.

Revisit the producer crossover with complete endpoint evidence on additional
workloads/devices while retaining the same scientific and resource gates.

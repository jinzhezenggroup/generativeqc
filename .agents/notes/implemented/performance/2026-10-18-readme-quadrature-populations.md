# Decision: separate grid populations and quadrature convergence

Status: implemented
Date: 2026-10-18

## Problem

The README's shared `(48,16,32)` quadrature has 24,576 points per atom. Its
VV10 pair-domain size makes unlabelled single-grid timing comparisons easy to
misinterpret. Matching two engines on a smaller grid does not qualify that grid's
quadrature accuracy. Native KS currently has one semilocal/nonlocal grid owner.

## Decision

Expose explicit grid sweeps through the existing complete-endpoint runner, with
grid-local references, logs and outcomes. Preserve single-grid CLI compatibility,
but label every new plot and record pre-screening point/pair bounds separately
from actual SCF/force work. Compare independent reference forces and energies
between grids at both original and moved geometries, with explicit tolerances.
Do not modify production quadrature defaults from one water-cluster study.

## Rejected alternatives

- Changing only GPU4PySCF's NLC grid would compare different numerical models.
- Shrinking all dimensions without force-convergence checks is unsafe: the
  initial n1 measurements of `(24,8,16)` and `(32,12,24)` fail the exploratory
  force gate against a denser three-atom grid despite converging SCF.
- A same-grid `1e-8 Eh`/`1e-7 Eh/bohr` engine-parity pass cannot substitute for
  cross-grid convergence. The latter tolerances must not relax engine parity.

## Reference compatibility

Dense quadrature produced empty AO tiles in GPU4PySCF 1.8.1. Its SCF iterator
skipped those tiles, causing an NLC density/coordinate shape assertion. Its full
grid-response routine constructs a new NumInt and also needs the zero-AO adapter;
adapting only the SCF instance left a zero-size `take_last2d` kernel failure.
Scope the existing exact-zero iterator compatibility to newly constructed
response integrators, restoring the class in a finally block. Do not replace
any reference integral, XC, VV10, partition response or derivative mathematics.

## Evidence and revisit conditions

Host tests exercise forwarding, isolation, invalid dimensions, missing geometry
oracles, model mismatch and cleanup after a response exception. Real-GPU
qualification uses finite Slurm allocations on supported nodes. The benchmark
report retains measured work and outcomes, including rejected grids. Separate
semilocal and VV10 grids require a production owner/ABI/analytic-response change,
not another reference-only benchmark override.

## Publication storage (2026-10-19)

The first PR check exposed the repository's aggregate evidence budget. Store
timings, independent force/energy arrays, convergence observations, build
settings and raw receipt hashes together in one compressed JSON record, with
one canonical SVG in `assets/`. Remove only optional iteration trajectories
and subcomponent timing profiles from the reduction, not complete endpoint
samples, convergence gates, SCF builds, nonlocal scopes, or semantic work and
traffic counters. Full journals remain locally and on n1 under their original
hashes. A replay regression still checks all 96 native calls against independent
reference arrays. The superseded README DF SVG is losslessly compressed using
its existing stored/expanded-identity verifier, matching the earlier Direct
figure's storage policy. Do not raise the budget or delete other studies'
observations to fit this publication.

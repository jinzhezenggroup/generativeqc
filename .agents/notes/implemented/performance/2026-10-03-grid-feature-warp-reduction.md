# Decision: cooperative grid density-feature reduction

Status: implemented; scalar fallback below 32 active AOs
Date: 2026-10-03

## Motivation

The unchanged 96-atom warm PBE0 endpoint's Nsight capture, Slurm job 12097,
attributes 9.517 s to the grid `feature_kernel`. One thread owns a point and
serially scans every AO in both spin panels, although adjacent AO values are
contiguous. Native SCF already uses a warp-per-point feature reduction in the
same compiler module.

## Decision and boundaries

The compiler emits a second grid feature schedule: one full 32-lane warp owns
one point, lanes accumulate disjoint AOs, and a fixed shuffle tree reduces
each of the five spin features. Only lane zero publishes. Sigma is formed
after both spin gradients are fully reduced, using the existing scalar feature
policy. Both schedules call the same `add_features` and `sigma` algebra; the
new code changes traversal and summation order, not functional mathematics.

Use this schedule at 32 or more active AOs, matching the existing native SCF
policy. Retain the original scalar kernel below that bound, including empty
maps. The native owner supplies the current active extent, so a small selected
map does not accidentally inherit a large-basis schedule. The public grid tile,
feature mask, work-panel layout, owned arena, memory allowances and thresholds
do not change. There is no shared scratch, floating-point atomic reduction,
host staging, density clipping or extra source evaluation.

Every lane in a warp follows the same point loop and shuffle sequence, even
when some lanes have no AO work. No synchronization crosses warp boundaries.
NaNs in unrequested jet slots must not enter the arithmetic; the existing
sticky error channel still rejects nonfinite requested outputs. Scalar and
warp accumulation need not be bitwise identical, so independent physical
energy/force gates are required before promotion.

## Work and tests

Both routes execute exactly `2 * active_AOs * points` bilinear contributions.
The change distributes those contributions instead of repeating them or
screening them. Instrumented host tests execute the actual emitted kernels
and launcher with a warp shuffle stand-in. Independent long-double sums cover
all fifteen masks, 0/1/31/32/33/65/768 AOs, point/AO tails, grid-stride loops,
signed inputs, poisoned unrequested jets, vacuum, requested NaNs, sticky errors
and output canaries. The focused tests pass. The initial 777-test combined host
run included inapplicable publication-harness combinations; it is not counted
as additional coverage. The tightened final eight-module run passes 672 tests,
with every enumerated failure injection required to fail. Compiler structure
and formatting checks are separate gates.

Node1 Slurm job 5430 passes 133 grid GPU tests, including thirty new independent
CPU-AO/NumPy feature comparisons for Cartesian and spherical s/p/d/f bases.
These cover independent signed spin densities, empty spin channels, vacuum,
full and noncontiguous maps, 31/33-AO boundaries, and 0/1/17-point tiles. Four
representative cases each pass memcheck, initcheck, synccheck and racecheck:
zero errors and zero race hazards/warnings. The same job subsequently passes
fourteen independent PBE0/r2SCAN RKS/UKS geometry cases through 128 atoms,
including constrained resources, geometry changes and tails. Full PBE0
endpoint runs are recorded separately rather than inferred from these checks.

An earlier qualification attempt rejects empty point arrays in the test's
nonempty-only error-summary helper before comparing values. Empty arrays now
use exact shape/value equality; nonempty numerical tolerances are unchanged.
The failed log and original test hash remain in local artifacts.

## Provenance and final qualification

This experiment composes master `52d2322b5`, the cooperative AO point panel,
and the proven-identical resident-spin product optimization. Its rebuilt
native library SHA-256 is
`6cdc4519bc5eb577850bb6d4e73cc5d5d4865401c182e18eade07eeb60ef5464`.
The preceding AO-plus-spin candidate is frozen separately with native hash
`d6b2412c15bc6a62f9be206849c390b7a50f1bacca4bb4b0bd4ea98494e78320`.
Do not mix their source inventories or timing results.

Both that master52 experiment and the final master `74c89369c` composition
complete all six sizes, cold/moved and all ten warm observations per size,
against independent references. The [retained final-master campaign](../../../../benchmarks/results/pbe0-grid-reuse-20261003/README.md)
uses native hash
`955293a77d2f662abdd6b180cd3b9eae957ea1f36f93bdb1de392a6915402644`.
All 72 native and 72 reference gates pass; maximum native energy/force errors
are `1.036824e-10 Eh` and `3.101085e-11 Eh/Bohr`. Node3 jobs 12101/12102 qualify
the exact latest-master composition with all 138 grid tests, native independent
J/K derivative/fallback checks, and the same 14 large geometry cases.

Master52 Nsight job 12100 reduces feature-kernel time from 9.516827 to
0.342265 s with unchanged 9,216 launches. Its instrumented replay is not a
publication sample. Final-master native 96-atom original warm median is
81.308650 s versus reference 10.141946 s: the large-system gap is not resolved.
Local preceding artifacts remain in `.artifacts/pbe0-cooperative-features/`;
final results are under `.artifacts/pbe0-final-master/`. Their frozen n1 sources
are respectively `qc-pbe0-features-20261003` and
`qc-pbe0-final-master-20261003`. Preserve those identities and failed fixture
attempts rather than relabeling old results as measurements of a newer build.

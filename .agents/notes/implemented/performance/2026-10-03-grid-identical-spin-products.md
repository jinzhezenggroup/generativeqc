# Decision: reuse proven-identical resident spin density products

Status: implemented; unproven and orbital sources retain the general path
Date: 2026-10-03

## Problem and measured baseline

The complete 96-atom strict PBE0 warm endpoint in Slurm job 12097 spends
17.294 s in density-product GEMMs. The resident one-spin upload already splits
total density into two bitwise-identical owned matrices, but grid execution
still computes their identical products twice. This is avoidable scientific
work, not a reason to raise the grid tile or memory allowance.

## Decision

Add a private provenance witness to the grid plan. Only a successful, ordered
resident `spins=1` upload establishes it. Every density/source/center setter
revokes it before validating replacement arguments or submitting transport.
A rejected replacement therefore cannot retain the optimization, even if the
previous valid density remains usable by the general route. The existing
readiness rules continue to reject partially transported densities.

For an admitted nonempty density task, compute alpha's requested products once
and copy the resulting panel to the existing beta work region on the same
owner stream. Respect `first`, `count`, and the current point/active-AO stride:
tau-only requests must not read or publish an uncomputed value jet. The full
two-spin task ABI, feature arithmetic, accumulation order, and buffer accounting
remain unchanged. Selected maps gather both owned densities as before; orbital
sources and unproven densities retain the general two-spin implementation.

For a tile with P points, A active AOs and J requested product jets, work changes
from two batched GEMM calls / 2J matrix products to one call / J products, plus
one device copy of `8*J*P*A` bytes. Empty tasks do neither. No allocation,
screening, precision, public tile, or physical threshold changes.

## Rejected alternatives and invariants

- Do not infer equality from method names, matrix dimensions, source-pointer
  aliasing, approximate norms, or a previous successful upload.
- Do not alias beta's ABI-visible panel to alpha or omit one observable spin.
- Do not compare or stage the resident density on the CPU. The producer's
  snapshot and stream contract remains responsible for valid source contents.
- Preserve existing producer/destination event ordering and owned-copy lifetime.
  A panel-copy failure must leave both the view and density-jet lease unpublished.

## Qualification so far

The instrumented host runtime tests execute the actual upload and publication
control flow. They compare every work-buffer byte and untouched canary against
two host GEMMs for all fifteen masks, point tails, empty maps, dense and selected
maps, and deferred consumers. They count GEMM calls/products and copied bytes.
Source replacement tests include host, centers, orbital, RKS and UKS sources,
same/foreign streams, aliased UKS pointers and injected submission failures.
The initial combined host run passes 634 tests (including existing resident
source-contract checks); this is not real-device performance evidence.

The final harness enumerates only legal lease combinations and applicable
failure injections, requiring every injected failure to return an error. Its
focused composed run passes 526 tests, including the separate feature-schedule
checks. Earlier larger parameter counts included unsupported combinations
that returned early or inapplicable injections; they are not extra coverage.

Node3 Slurm job 12099 passes 103 real-device tests, including ninety resident
RKS/UKS transition cases. They compare uint64 views of all requested outputs
against the general two-spin route, exercise masks, selected/empty maps,
point tails and producer mutation after upload. Four representative cases pass
each of memcheck, initcheck and synccheck with zero errors.

An earlier run, job 12098, fails these overly strict comparisons because the
test supplied different density bytes: host upload symmetrizes near-symmetric
fixture densities, whereas resident upload copies its producer's canonical
matrix. The fixture now explicitly symmetrizes before either route; no
production tolerance was loosened. Both failed and successful logs are retained
under `.artifacts/pbe0-spin-reuse/`.

These device tests compile the changed standalone grid with ccache but use the
previous native library for unchanged basis services. They are not relabeled
as full latest-master binary qualification. A separate Release build composes
the candidate and AO-panel optimization with master `52d2322b5`.

## Composed qualification and observed work

The [final-master campaign](../../../../benchmarks/results/pbe0-grid-reuse-20261003/README.md)
completes all 72 native and 72 independent reference endpoints on master
`74c89369c` plus the three recorded production edits. Every endpoint passes
the unchanged `1e-8 Eh` / `1e-7 Eh/Bohr` gates. All native warm calls take one
iteration without fallback. Native and generated JIT identities are retained.

Final-master job 12102 passes all 138 grid device tests; job 12101 passes the
independent native J/K derivative/fallback test and 14 PBE0/r2SCAN RKS/UKS
geometry cases through 128 atoms. The final eight-module host run passes 672
tests. Prior composed master52 job 5430 also passes four representative grid
cases under each of memcheck/initcheck/synccheck/racecheck with zero errors or
hazards; that sanitizer receipt keeps its distinct binary identity.

The independently frozen master52 Nsight comparison records 18,496 -> 9,280
GEMM-family instances, removing exactly one per each of 9,216 force tiles.
Panel copies are observed, not inferred: 9,216 copies of 6,291,456 bytes total
57,982,058,496 bytes and take 0.027549 s. The GEMM family decreases from
17.298339 to 8.655410 s. These diagnostic kernel totals are not publication
wall samples. Final 96-atom native warm time remains 81.308650 s versus
10.141946 s for the reference; bounded derivatives and geometry still dominate.

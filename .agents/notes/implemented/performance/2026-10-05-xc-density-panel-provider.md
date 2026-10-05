# Decision: prepare a materialized density-panel candidate

Status: implemented
Date: 2026-10-05

## Problem

The native XC density contraction consumed the canonical lowering request but
offered only generated scalar/tiled execution. Large dense point panels warrant
an independent matrix-provider crossover gate; small-AO negative evidence does
not establish the crossover at 384/768 AOs.

## Decision

Add a strict FP64 materialize/GEMM/finite-publication candidate to the same
request as the generated implementations. The scientific operation is
`0.5 * (D + transpose(D)) * AO`, including spin and AO-jet panels. The compiler
emits the materializer from the same scalar DAG as the fused density summand.
It executes once per complete XC evaluation and is reused across full/tail
point panels. Every captured body rematerializes; pointer identity never proves
that a density is current.

The previous scientific serialization tagged the reduction `increasing-nu`
despite requiring reproducible, rather than exact-order, determinism. Deliberately
change this tag to `sum-nu` in the scientific identity. The candidate topology
now records `increasing-nu` for the existing generated recipes and
`provider-reproducible` for the matrix recipe. The scalar summand and generated
strict/mixed arithmetic are unchanged. Library reduction/FMA order differs and
is admitted only by independent numerical gates, not by an identity assertion.

The shared panel provider owns one prepared cuBLAS handle, matrix cache, GEMM
submission and finite publication. Jet/point rows are flattened; spin batches
share the AO panel. The DFT owner consumes only the prepared operation.
Diagnostics report the selected candidate, provider version, exact cache bytes,
retained provider bytes, conservative allowance and preparation time.

Resources are separate from the existing numeric arena: 96 MiB provider
allowance, `spins * nao * nao * sizeof(double)` device cache, and a 16 KiB host
reservation covering the optional owner/binding and selection metadata. The
caller must admit both budgets before optional preparation. Preparation is
transactional and forbidden during capture. Execution performs no selection
or allocation and enforces the prepared device/stream, dimensions and disjoint
input/output/cache buffers.

Production keeps generated execution by default. Positive budget alone does
not promote the alternative; qualification is test-only in this change.
Mixed FP32-compute/FP64-accumulation, local AO, and signed response retain their
existing generated recipes. Resource exhaustion or unavailable library
resources retain the bounded generated fallback.

## Evidence

Real-device qualification runs on n1/node1 with finite Slurm `main` allocations,
`--gres=gpu:5090:1`, CUDA 12.9, cuBLAS 120901. Reproduction scripts and complete
logs are retained under `.artifacts/1876-density-gemm/` on local and n1
checkouts. CMake explicitly uses ccache for C++/CUDA; its actual compiler
commands and before/after statistics are retained.

- Fourteen host binding/admission tests and focused type checks pass.
- Eighty-seven additional compiler structure, matrix schedule, publication,
  precision and compiled-resource tests pass. An initially broader invocation
  also selected 46 independent CPU contraction cases requiring a shared native
  library, which this test-executable-only build did not provide. Their
  missing-library failures are retained in `host-contracts.log`; they are not
  counted as numerical passes.
- Direct materializer/provider gates use nonsymmetric densities, long-double
  scalar oracles, 1/17 AOs, RKS/UKS, four meta-GGA jets and 1/7 point panels.
  Three changed-input captured replays check cache refresh. Alias rejection and
  sticky finite errors are exercised.
- Twenty-four complete LDA/PBE/r2SCAN RKS/UKS cases cover qualified execution,
  absent qualification, insufficient budget and unavailable provider, with
  changed density, independent CPU E/Vxc oracles and arena canaries. Mixed,
  local-AO and signed-response suites exercise the optional-resource boundary.
- The full native XC regression passes. Compute Sanitizer memcheck and initcheck
  report zero errors.

The crossover fixture matches the potential-only #1958 domain: synthetic
two-center Cartesian s/f bases with exactly 384/768 AOs, 2,304 points, tile
sizes 128/256/512, original and 0.02 Bohr moved geometry, restricted PBE with
exchange/correlation scales 0.75/1.0. Each endpoint includes one cold and five
warm complete fixed-density XC evaluations, plus independent CPU energy and
every Vxc element checks at `2e-9`. This is not full PBE0 SCF/force evidence.

Initial warm medians in milliseconds (generated / matrix):

| AOs | tile | original geometry | moved geometry | elapsed reduction |
| --- | --- | --- | --- | --- |
| 384 | 128 | 5.513 / 5.331 | 5.514 / 5.333 | 3.28–3.29% |
| 384 | 256 | 3.948 / 4.122 | 3.950 / 4.124 | -4.42–-4.41% |
| 384 | 512 | 3.208 / 2.968 | 3.204 / 2.967 | 7.40–7.48% |
| 768 | 128 | 11.935 / 9.812 | 11.918 / 9.805 | 17.73–17.79% |
| 768 | 256 | 9.792 / 8.569 | 9.798 / 8.575 | 12.48–12.49% |
| 768 | 512 | 9.208 / 7.396 | 9.199 / 7.401 | 19.55–19.68% |

The initial measured binary SHA-256 is
`ebcebf84245c9ea251134dc76a05b27b371723037a249517b59fd4ded11fe0f8`.
The subsequent candidate-index validation is qualified separately in the same
evidence directory: the complete gate, regression, crossover and both sanitizers
pass for binary
`76813bbfebb62ff11b3a0a756558cd94ddf0d0f90809254829b55019fd7a2210`.
The initial timing logs/hash are under `initial/`; do not attribute that snapshot
to the later binary. The maximum observed potential error is below `6e-14`.

Across six evaluations, the 384/768 AO cases each submit 108/54/30 GEMMs at
tile widths 128/256/512. The last 512-point panel has 256 points. The PBE
density operation needs one AO work jet; meta-GGA's four-jet path is covered
separately. The 384/768 AO cases consume 2,038,431,744 / 8,153,726,976 density
summands, and materialize 7,077,888 / 28,311,552 bytes, respectively. This
candidate changes throughput/data movement, not the mathematical summand count.

## Rejected alternatives and revisit conditions

Raw `D * AO` would lose the established nonsymmetric-density rule. Repacking
per point tile repeats invariant work. Implementing symmetrization in the generic
provider duplicates science. Replacing explicit mixed rounding with GEMM would
change the admitted arithmetic. Promoting every large AO shape ignores the
384-AO/tile-256 regression.

Next qualify composition with the symmetric potential candidate and complete
cold/warm/changed-geometry PBE0 endpoints under a shared, explicitly charged
resource plan. Neither isolated kernel time nor this fixed-density fixture
authorizes a production shape policy.

References: #1876, #1886, #1953, #1958;
`docs/developer/xc_native_cuda.md`.

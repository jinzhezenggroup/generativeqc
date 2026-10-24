# Decision: retain full-rank single-packed DF factors in metric eigen-coordinates

Status: implemented
Date: 2026-10-23

## Problem

After prepared Hermite raw generation, a 96-atom def2-SVP resident DF endpoint
still spent about nine seconds whitening its raw panels. The existing preparation
computed `Q^T A^T` and then multiplied by `Q diag(lambda^-1/2)` to retain the
symmetric factor `B = A M^-1/2`. This second large multiplication restores
physical auxiliary coordinates that J/K immediately contract away.

The parent raw-generation implementation is commit
`ef0965959fe6dd1b5ffe78c787ddfba4f87a5ba7` (PR #2682). Its existing CPU multiply
alignment fix remains part of the baseline; do not attribute that earlier fix
to this representation change.

## Decision

When every metric in a `SymmetricLowerSingle` plan is full rank, retain
`E = (A Q) diag(lambda^-1/2)` directly. Use the existing projection GEMM and
in-place FP64 scaling, without enlarging scratch, retaining another tensor,
regenerating source values or changing panel widths. The plan's immutable
auxiliary gauge follows its existing unique metric/source lifetime identity.
J/K need no arithmetic change because `E E^T = B B^T`.

Response views explicitly carry the gauge. A physical auxiliary derivative
requires `A M^-1 = E R`, where `R = diag(lambda^-1/2) Q^T`. Prepare this small
map in the existing, otherwise unused full-rank inverse-matrix response
workspace. The J-only potential, bounded general-density panels, occupied
factors and exact final-K projection reuse all consume the matching map.
The final occupied projection's symmetry reduction remains on occupied pairs;
it does not identify metric eigen-indices with physical auxiliary indices.

The explicit occupied `spectral` diagnostic retains divide-before-rotate
ordering. For E its input is already projected, so it skips a redundant Q^T
and performs one rotation. Its no-final-projection-reuse semantics are unchanged.
Work counters describe the actual contractions and avoided work in each gauge.

## Invariants and fallbacks

- Changed DF value/response arithmetic remains FP64. Grids, convergence, thresholds, auxiliary bases,
  screening and complete grid/auxiliary force response are unchanged.
- Dense, paired-raw packed, public tensor and streamed owners keep their existing
  representations. Rank-truncated or mixed-rank single-packed batches retain
  symmetric B and the existing complete spectral response.
- The forward symmetric inverse root remains owned and unchanged for raw-source
  readers and second-response consumers. R does not replace that public source.
- Geometry replacement builds a fresh plan. No mutable gauge selector or new
  cache identity is introduced. Response retries preserve the view's gauge while
  revoking one-shot projection leases under the existing rules.
- No device allocation, host oracle, extra resident AO tensor or response-budget
  borrowing is added. R and all projections occupy existing disjoint intervals.

## Rejected alternatives

Returning to `A * (Q lambda^-1/2 Q^T)` in one GEMM would reintroduce the known
weak-direction cancellation problem. Forming E first is essential; the saved
operation is the subsequent orthogonal rotation, not the initial projection.
Keeping E while pretending it is symmetric B is also invalid: J/K parity would
hide incorrect physical forces. An additional permanent response-root allocation
is unnecessary because the full-rank workspace already has a free metric slot.

## Evidence

The new device response probe uses independently generated nonsymmetric
orthogonal eigenvectors, condition numbers up to `1e10`, signed multi-density
terms, full/ragged auxiliary panels, symmetric and spectral forward factors,
both occupied root policies, and the real final-K projection. Long-double host
contractions independently qualify raw-integral and metric adjoints.

Existing molecular, force, resource and lifecycle qualifications cover cold,
warm and moved owners; canonical/corrected factors; bounded panel fallback;
RHF/UHF; and truncated metrics. Host failure-injection tests additionally check
that a retry carries the original gauge after consuming the final projection.
Raw receipts and exact build/source identities are retained under
`.artifacts/` and `n1:/data/jzzeng/df-spectral-whitening-20261023/`.

### Complete endpoint measurements

96-atom water, spherical def2-SVP, frozen GPU4PySCF default auxiliary tables,
RTX 5090 on n1 under finite Slurm allocations. Both arms use unchanged native
Standard/Fast profiles, stock XC/VV10 grids and complete force response. Cold
includes constructor/preparation and energy/forces; warm is the median of three
frozen-density calls. No moved-geometry timings were collected.

The following table measures candidate
`351aed18aad44f15f8b853b3c4c2f1a37f73be75` against the original parent above.
It predates the master integration described below; retain that provenance.

| Standard method | Cold parent → candidate (s) | Warm parent → candidate (s) |
| --- | ---: | ---: |
| HF | 33.994 → 29.141 | 5.009 → 5.011 |
| PBE | 32.097 → 25.768 | 7.123 → 7.046 |
| PBE0 | 41.391 → 36.647 | 11.164 → 11.151 |
| B3LYP | 42.094 → 37.650 | 11.502 → 11.511 |
| r2SCAN | 31.825 → 27.384 | 7.922 → 7.948 |
| M06-2X | 44.972 → 40.172 | 11.742 → 11.884 |

All six Fast cold endpoints also improved; their full samples remain in
`results/clean/` and `endpoint-summary.json`. PBE's reversed-order Standard
control measured **30.771 → 25.772 s cold**, **7.062 → 7.036 s warm**. Retain
both PBE observations; the initial parent cold sample was higher. These are
scoped cold gains, with warm medians varying from about -1.1% to +1.3%, rather
than evidence of a warm speedup or a new GPU4PySCF ranking.

Every paired sample has identical SCF/Fock counts. HF remains 11 main cold
cycles / 21 Focks and 2 warm cycles / 3 Focks. Across all six methods, both
profiles and the reversed control, maximum absolute changes are
`8.640199666842818e-12 Eh` and `1.7024603948812e-11 Eh/Bohr`, within the declared
same-task preservation gates of `1e-8 Eh` and `1e-7 Eh/Bohr`. Independent
small-molecule energy/force gates are retained separately; these deltas do not
establish the absolute accuracy of default quadrature or DF fitting.

Separate synchronized HF trace measurements show resident materialization
**11.118 → 6.572 s**: raw generation **1.959 → 1.968 s**, initial projection
**4.576 → 4.588 s**, and removal of the **4.576 s** scaled rotation. PBE0 gives
the same mechanism (**10.977 → 6.493 s** materialization). Both keep the same
15 panels, pair capacity 20,878, 1,067,790,336 raw outputs, and 8,542,322,688
fitted bytes. Whitening reduces from 30 to 15 GEMMs and from
15,444,519,419,904 to 7,722,259,709,952 GEMM FLOPs. Intrusive trace intervals
are attribution evidence, not clean endpoint timing.

### Qualification and identity

- 177 CUDA tests pass, including 72 independent gauge/adjoint populations,
  default-auxiliary Standard/Fast endpoints for all eight closed-shell README
  methods plus UHF/UKS, moved-geometry correctness, corrected/final projections,
  bounded response, arbitrary densities, batches and truncated metrics.
- 174 host tests pass; one local native-artifact-dependent case is skipped.
- Memcheck passes 74 tests (all adjoint populations plus HF/PBE0 complete
  endpoints), with zero errors. Focused racecheck passes all 72 adjoint
  populations, with zero hazards/errors/warnings. An earlier broader racecheck
  run reached its final DFT/oracle test but was stopped for excessive cost;
  retain that incomplete log, and do not count it as a completed qualification.
- Full pre-commit/source checks pass. The early failure receipts are retained:
  one old work-counter expectation required updating for the removed GEMM, and
  two UKS tests required the omitted `pbe_uks[_spd]` AOT artifacts to be built.
  Their completed reruns pass without changing numerical gates.
- 2,408 scientific/build input files match the staged remote source, with zero
  mismatches. Source/patch manifests and selected-launcher cache statistics are
  retained; global sccache statistics include concurrent work on n1.

Parent core SHA256:
`b51b85ea8324bc104bbb7f3264c3220324a3c7e21c338ea714441749ae78c2ad`.
Candidate core SHA256:
`fb8725997439ddeda0cd681212e12f7bba58031a587f9e253e9618f169ecc3c1`.
Candidate native source identity:
`1cc089b379b547ab001e03c0ea6541d9a3c1759c60fe4aa8a41ced6408f1adc6`.

### Master integration

After #2682 merged, this change was rebased onto master `92aef8c12` and rebuilt.
The merged parent also includes unrelated intervening PRs, so its tree is not
identical to the originally measured parent. A one-off preservation audit found
633 HF/DFT value/control/response source files unchanged across the rebase,
including `src/scf`, `src/dft`, the public Python runtime and DF method compiler
sources. The common lowering change adds an optional layout-transfer executor
without changing its default behavior. This audit is retained evidence, not a
permanent source-fingerprint test.

The integrated build passed all 177 CUDA tests again, 174 host tests with the
same one local skip, 74 memcheck tests with zero errors, 72 racecheck populations
with zero hazards/errors/warnings, and full pre-commit checks. Its core SHA256 is
`bfe8927d3a9632aee72700869dbd12041d61654e4a7d56ae36f70451d7a95d93`;
native source identity is
`ca291b81d8e70653b57e77146c98e7449754ba8147b3248eb31bfc866495e410`.

Initial integrated Standard replay measured HF **29.029 s cold / 5.015 s warm**
and PBE0 **37.704 s cold / 11.500 s warm**. Energies equal the original candidate
results; maximum force change is `2.42e-13 Eh/Bohr`, and SCF/Fock counts match.
These two receipts retain an older experimental revision label; their native
build hashes identify the integrated executable. Do not rewrite the receipts
or attribute these timings to the earlier binary.

A subsequent same-GPU paired PBE0 control (Slurm job 13024) compared the original
parent with the integrated candidate: **41.117 → 36.834 s cold**,
**11.211 → 11.215 s warm**. Both use 9 cold cycles/Focks and 1 warm cycle/Fock.
Maximum paired energy/force differences are `2.28e-12 Eh` and
`1.65e-11 Eh/Bohr`. Retain both integrated PBE0 observations; the measurements
do not isolate the cause of their timing variation. Original measured and final
integrated source manifests, binary identities and result cohorts are retained
separately under the artifact locations above.

## Revisit when

Extend the gauge to paired-raw or dense owners only after qualifying their
raw-projection reconstruction and public tensor contracts. Extending it across
rank truncation requires the complete retained/discarded-subspace derivative;
full-rank inverse identities are not sufficient.

## References

- [Current representation contract](../../../../docs/developer/df_packed_storage.md)
- [Prepared raw generation](2026-10-22-prepared-hermite-raw-df.md)

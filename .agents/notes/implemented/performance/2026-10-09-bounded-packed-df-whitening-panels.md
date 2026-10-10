# Decision: batch packed DF whitening inside existing scratch leases

Status: implemented
Date: 2026-10-09

## Problem

The first-stage cold KS repair retained fitted B and removed repeated SCF
source work, but the 96-atom energy endpoint still spent about 20 s preparing
the owner. A completed materialization trace identified 768 lower-row panels,
1536 whitening GEMMs and 16.338880 s of materialization: about 4.859158 s in
raw generation and 11.325358 s in the two factor products.

## Decision

For packed-single storage, fill contiguous lower-pair scratch across source
row boundaries before transforming it. The source API still receives valid
dense AO-pair ranges within each lower row. The fitted output uses contiguous
packed order. A source row may be split at a staging boundary; flush the final
partial panel and reset staging independently for every batch item.

The pair capacity is the minimum of the existing raw/projection and exchange
panel capacities divided by naux. No allocation, total allowance, response
reserve, occupied-U capacity, source identity or lifetime policy changes.
The packed-raw and dense branches are unchanged. Full-rank metrics retain
both Q^T*A and Q*lambda^-1/2 products; rank-deficient metrics retain their
existing truncated inverse-root transform. This batches work, not mathematics.

For each system the number of full-rank transforms becomes
ceil(nbf*(nbf+1)/2 / pair_capacity), rather than at least one per lower row.
The new staging-capacity counter makes this bound independently checkable.
The CUDA KS regression asserts the actual panel/GEMM counts and the existing
independent energy/force, replay, constrained-budget and lazy-fallback gates.

## Rejected alternatives and invariants

- Do not replace the two factor products with a precomputed inverse root:
  the original setup deliberately avoids weak-direction cancellation loss.
- Do not change the auxiliary gauge or metric cutoff to save one rotation;
  existing force and occupied-projection consumers depend on that convention.
- Do not allocate a complete raw A or enlarge a scratch lease to batch panels.
- Every raw lower (pair,P) is generated once; pair coverage and whitening FLOPs
  remain unchanged. Source launch count may increase when a row is split.
- Preparation cannot create an occupied/final-state force lease. The existing
  final-K producer, token checks, one-shot borrowing and bounded response remain.

## Energy endpoint evidence

After initial PR #2164 submission, three alternating fresh-process triplets
compared the saved first-stage binary, this change, and GPU4PySCF 1.8.1 in one
finite-time Slurm allocation on node1/RTX 5090. The same frozen 96-atom fixture,
FP64 PBE0, spherical def2-SVP/cc-pVDZ-JKFIT, unpruned 48x16x32 grid and strict
SCF tolerances from the first-stage qualification are retained. Cold timing
includes prepare through synchronized public result, excluding imports,
context initialization, Calculator construction and post-result teardown.

| Energy cold endpoint | Median | Range | Iterations / Focks |
| --- | --- | --- | --- |
| Saved stage one, freshly resampled | 75.028340 s | 74.931538..75.071935 s | 25 / 25 |
| Bounded cross-row panels | 73.374362 s | 73.350090..73.390982 s | 25 / 25 |
| GPU4PySCF 1.8.1 | 65.753644 s | 65.616460..65.756030 s | 37 / 38 |

Preparation median decreases from 20.123200 s to 18.456659 s. Complete cold
time improves another 2.2045%, without per-Fock normalization. It is 13.0158x
faster than the one valid original 955.023585-s baseline, but remains 11.5898%
slower than matched GPU4PySCF. These are endpoint, not whole-process, claims.

The diagnostic materialization uses capacity 20,338 pairs, 15 panels and 30
GEMMs, returning all 295,296 packed pairs. It generates 8,769,110,016 raw
bytes in 782 source launches versus the original 768, with unchanged
16,275,468,189,696 whitening FLOPs. Its traced materialization is 14.564812 s;
the clean complete endpoint population, not this trace, establishes the gain.

Eleven host budget/source regressions pass. The expanded CUDA suite has 25
passes and one resource-route assertion failure in the 96-atom RHF automatic
exchange test. The exact same complete suite on the saved first-stage binary
also has 25 passes and the same sole failure. Both pass its independent E/F
gates but choose a bounded dense route without occupied-SCF provenance when
run after the preceding cases. Its isolated first-stage fresh-process run
passes. This change does not repair or suppress that unrelated route assertion.
The initial copied-library baseline attempt lacked sibling stationary force
artifacts; those packaging failures are retained separately and do not count
as numerical or causal baseline evidence.

Remote host-probe repairs added during this qualification are preserved. The
integrated budget/source/reservation host suite passes all 14 tests; these
repairs alter probes, not production source or the measured binary.

## Complete force and source-identity qualification

A separate traced, converged 96-atom E/F endpoint returns all 288 coordinates
in 294.721926 s with a 221.229604-s force phase. This is diagnostic qualification,
not a clean force timing population or a claimed force-performance repair.
Its independent GPU4PySCF force error is 1.57396e-10 Eh/bohr; the largest energy
error in the new clean triplets is 2.00089e-11 Eh. The native physical residual
is 3.82286e-13. Gates remain 1e-8 Eh, 3e-7 Eh/bohr and residual <=1e-9.

The same full E/F trace has one resident B materialization, 50 J/K rows over
25 Focks, zero SCF raw-source auxiliary evaluations and one completed final-U
force hit returning all 288 coordinates. The packed planner and allocations
are unchanged; the constrained 112-MiB oracle regression still passes.
No claim is made about whole-KS or CUDA-driver memory bounds.

Qualified library SHA-256:
`eebbe62a00def9603d3ca66cbb8105c8dbb7b41919f36db17768f381f9b499ac`.
Native source identity:
`25378a419371ff0de4cc603c49d0e40a638ebea8d85af7cc04fa60436bae2c45`.
Release sm_120, no fast compile, unchanged FP64 policy. All real GPU tests,
traces and complete endpoint runs use finite-time Slurm jobs with preserved
device visibility. The incremental build reuses the existing compiler cache.

## References and revisit conditions

- [First-stage admission, derivative lifetime and full endpoint qualification](2026-10-09-ks-df-cold-resident-values.md).
- Raw triplets and traces: ignored `.artifacts/pbe0-df-fix/receipts/`, and
  `/data/jzzeng/qc-pbe0-df-cold-20261009-4e20f7a7b/results/` on n1.
- Revisit if scratch ownership, projection capacity, source indexing, metric
  gauge or final-U response semantics change. Further algebraic changes require
  separate independent force gates, not extrapolation from these launch counts.

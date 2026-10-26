# Decision: native HF system-to-system density projection

Status: implemented
Date: 2026-10-26

## Problem

The occupied-space algebra is already native, but `progressive.initialize_from`
still prepares three overlap matrices and loops over RHF/UHF spin projections
in Python. This is a remaining #2227 R4/R6 scientific lifecycle boundary after
the separately proposed scalar verification (#2814) and physical audit (#2818).

## Decision

The implementation adds a bounded CPU system-to-system projection owner that calls the existing
`integrals::cross_overlap` and `scf::projection::project_density` implementations.
The owner validates ordered nuclei, exact geometry, all-electron state and spin,
queries matrix capacity before integral work, prepares the three shared metrics,
projects all spin channels and publishes only after every spin succeeds.
Do not copy overlap, metric, eigen, occupation or density equations.

Python retains typed batch eligibility, source-state transport, per-item reports
and transactional seed installation until a separate prepared-batch lifecycle
cutover. The new owner initially admits CPU all-electron RHF/UHF through-g
Cartesian/spherical models with built-in projection policy. Missing protocol,
custom policy, CUDA and unsupported domains keep named compatibility. Native
execution errors must never retry the Python scientific preparation path.

A query composes the existing occupied-projection workspace with explicitly live
metric/result matrices. This is a numeric matrix envelope, not process RSS or
complete allocator/provider accounting. Retain existing frontend resource gates
and charge the native bound before evaluation. No caller-owned prepared fleet
is implicitly covered by this subplan.

## Acceptance contract

- Independent PySCF overlap/projected-density comparisons including Cartesian
  normalization, RHF/UHF, empty spin channel, changed geometry and d functions.
- Occupied-rank loss and invalid geometry/electron/state refusal; exact/below
  capacity and transactional multi-spin publication, count/alias/nonfinite gates.
- SDK execution without Python; Python admitted endpoint forbids overlap and
  per-spin projection compatibility; failure must close native descriptors.
- Complete source/target progressive endpoint E/F/D/work acceptance with frozen
  source, pilot cost estimate and unchanged ABBA/BAAB 21+3 / 1.05 gates.

## Evidence

Qualification is frozen against `c8ba02453b326084d5175ef644c6be1c810afce7`, patch
SHA-256 `9d4ef8cfaaa80725d1477c69c5aa9f4a4be0381681b5789f1a73ef14ca857494`.
The [complete evidence](../../../../benchmarks/results/native-hf-system-projection-20261027/README.md)
retains every sample, oracle, runner, pilot and source patch.

121 CTests and 279 Python tests plus 25 subtests pass, including independent
Ne+H2 metrics/projection for RHF/UHF, Cartesian/spherical, STO-3G and d-containing
def2-SVP sources, both original and moved geometries. Empty beta, impure beta,
nonfinite/count/alias/ABI errors, exact/below budgets and metadata mismatch have
transactional gates. The installed SDK passes with `PATH=/nonexistent`, and
normal frontend execution forbids Python overlap/per-spin preparation.

The complete source/target progressive campaign covers RHF/UHF x Direct/DF,
two different geometries, 21 samples plus three warmups per block, ABBA and
BAAB. All eight original 1.05 timing/work gates pass (maximum ratio
1.0065456757099125); all 1536 E/F/D publications pass 1e-8/1e-6/1e-7 gates.
The pilot estimated 84.20 seconds and full collection took 85.32 seconds.
This is bounded nonregression, not a significant-speedup promotion.

Independent PySCF named STO-3G has differently rounded coefficients. A strict
comparison must transport the requested basis primitives/coefficients as oracle
inputs, then independently evaluate overlap and projection. PySCF Cartesian
normalization also requires diagonal-overlap scaling. Do not loosen gates or
change the production science to hide basis/representation mismatches. Initial
setup and oracle failures are preserved in the publication.

## Consequences

No overlap or metric-projection equations are duplicated. The public owner can
be called by SDK clients and Python, with every spin accepted before publication.
Capacity is a composed matrix envelope, not physical/RSS or complete provider
accounting. Python per-item source eligibility and seed installation remain;
this step does not close all progressive control, resource authority, CUDA
projection or #2227. Revisit compatibility only after each missing domain has
independent scientific, capacity and complete-endpoint acceptance.

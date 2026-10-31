# Decision: reuse idle CUDA KS storage for seed-validation products

Status: implemented
Date: 2026-11-01

## Problem

CUDA KS borrowed its GPU eigensolver for native cold guesses and strict density
imports, but checked eigenframes and occupation spectra with scalar CPU matrix
products. Each preparation consequently performed several cubic host products,
even when the ordinary owner already had an FP64 matrix-library provider. The
HF provider already had a bounded borrowed-product adapter.

## Decision and invariants

For at least 512 AOs and an available prepared matrix-library binding, borrow
`fock_history`, `residual_history` and `effective` while the KS owner is idle.
The first two buffers are abandoned DIIS storage that `begin()` resets;
`effective` is ordinary iteration scratch, also borrowed by idle warm-density
export. Do not touch `tmp1`/`tmp2`, which may back live stationary force leases,
or final density, coefficients, weighted density, generation tokens or warm state.

Keep the existing row-major FP64 equations and every charge, symmetry,
occupation, finite-value, eigen-residual and metric-orthogonality gate. Both
transpose flags are mathematical operands, not storage reinterpretations.
Preparation, explicit import and strict initial upload use the same adapter.
Native atomic/MINAO construction, initial-guess selection and target SCF controls
are unchanged. No extra device allocation or matrix-library handle is created.
Small owners and unavailable bindings keep host products. A supplied operation
never retries a failed product through the reference implementation. The existing
bounded cold-seed failure policy and allocation-error propagation remain intact.

All borrowed operations reject changed dimensions, active/pending iterations
and stream capture before submission, then drain queued copies on success and
failure before host vector lifetimes end. Transfer/synchronization counters
include the added two-input/one-output traffic. Selection depends on work shape
and prepared capability, without GPU-name or architecture branches.

## Alternatives

Changing the cold seed to MINAO requires separate DF admission and convergence
qualification. The first diagnostic stopped at the public Python admission
guard. A separate native-only experiment then admitted DF MINAO before planning,
retaining native budget and strict-density checks. On the same n5 RTX 5090,
MINAO/default cold times were 29.350/22.342 seconds for PBE0, 34.301/26.814 for
r2SCAN-3c and 240.275/234.853 for WB97M-V. Target iterations were 11/9, 12/10
and 28/28 respectively. All six populations passed the independent endpoint
gates. These cross-allocation observations reject promoting this experimental
seed; they do not qualify the unsupported public combination. The diagnostic
source, adapter, failed admissions and complete controls remain in the evidence.
Merely disabling Huckel was already slower in the prior campaign. This change
removes redundant host algebra while preserving the seed.

Skipping validation, relaxing its gates, borrowing stationary scratch, creating
another GPU owner, or using reference densities would violate the seed contract.
The complete endpoint, not a matrix-product microbenchmark, determines benefit.

## Evidence

See `benchmarks/results/cuda-ks-seed-products-20261101/` for original receipts,
independent gates, source/build identities, scheduler allocations, controls and
retained failures. The final source identity is
`b6b63f9063949ed70191c54b4d0f0b7efb788957274eb680c98f4242bbcf74f3`;
formatting-only pilot source `fdf1573b4b55c9301279c61550c6a940c4da0710967d805319f8484829b88602`
remains separately labeled.

The compiled sparse-matrix oracle exercises all transpose combinations at 512
AOs, shape refusal and active-owner refusal. Small MINAO keeps its original
independent CPU endpoint and bounded preparation tests. Public large-owner
checks use independent PySCF overlap/ensemble spectra, rejected batch import,
retained final-state tokens, frozen replay, changed geometry and constrained DF
storage. Host failure injection verifies that failed supplied products propagate.

The large-owner tests use an explicit small diagnostic grid and tight SCF
controls (energy 1e-12, density 1e-10), retaining the force-parity gate of 1e-7.
At ordinary stopping controls, changed-versus-fresh force differences reached
4.794e-7 despite preserved checkpoint atomicity and frozen replay. This was
convergence sensitivity, not permission to relax the state-preservation gate.
The 96/120-atom tests use 512/1024 MiB DF budgets: 64 MiB refused the smaller
case, and 512 MiB refused the larger case before seed admission. Both final
budgets remain below their multi-GiB dense sources. Retain these refusals and
the earlier tolerance-sensitive failures separately from passing observations.
Both large cases pass on A100 (224116), RTX 5090 (1388/1389) and PRO6000
(3818/3819). The native product/small-MINAO probes pass on both Blackwell
devices; the four public small RKS/UKS cases pass on all three devices.
The 238 host initial-guess and 70 catalog checks pass, as do the complete
pre-commit checks and the unchanged native materialization inventory review.

Performance observations cover the seven README functionals at water96, DF,
Standard, explicit force tile 1024 and ordinary SCF tile 256, on A100, RTX 5090
and PRO6000. They do not qualify every size, Direct/Fast workload, GPU or energy-only
endpoint, nor establish the broader goal of beating GPU4PySCF everywhere.

The A100 WB97M-V diagnostic identifies separate remaining work: cold KS entry
including range-source preparation is 21.925 seconds, and the post-SCF incumbent
LR action used only to rebase the fallback warm energy is 12.151 seconds. The
following warm action retains the source and takes 0.714 seconds. Deferring
unused fallback preparation is worth investigating separately, but a cached
and incumbent energy cannot be interchanged at the strict energy gate. Any
follow-up must cover eviction, changed policies, frozen warm snapshots, export,
changed geometry and failure without weakening physical acceptance. This seed
product change does not alter that lifecycle.

## Revisit

Revisit the shape crossover with complete cold/import endpoint evidence on new
devices. A larger matrix-library workspace requirement must remain charged by
the existing owner; an unavailable binding must retain the bounded host route.

## Integration after the Libxc provenance update

Rebase onto master `04d316ce0` (#2908) changes the complete source identity to
`8b75da56f4efef0a2b8b2dde33541ac1e0151df5ac5aeb95830820c29cba02d6`.
The native sources, CMake build inputs and DFT compiler sources remain byte
identical to the measured seed implementation; the incoming production edits
scope Libxc cache fingerprints and published-consumer validation. The recorded
GPU populations remain at their original identity `b6b63f90...` and are not
relabelled as measurements of the rebased source identity.

The automatic Libxc catalog is regenerated against the exact preceding #2908
summary. Its scalar/capability summary is unchanged, so there are no new
transition acknowledgements. The previous identity-v2 migration summary and
acknowledgements remain recoverable from `04d316ce0` at the original manifest
paths. Historical native qualification stays stale; no new endpoint admission
is inferred. Seed adapter behavior tests, the incoming provenance/consumer
tests and the retained independent performance-evidence reader are rerun.

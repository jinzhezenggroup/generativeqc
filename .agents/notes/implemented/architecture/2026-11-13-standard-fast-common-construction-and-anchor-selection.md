# Decision: common construction and conservative anchor selection

Status: construction implemented; original anchor-default withdrawal superseded
Date: 2026-11-13

## Current default-selection decision

The [superseding Fast-default decision](../performance/2026-11-12-retain-qualified-fast-anchor-default.md)
retains automatic fixed-anchor reuse for eligible budget-Fock/budget-stationarity
Fast owners at 192 or more AOs. Strict Standard and smaller owners remain opt-in;
explicit zero disables reuse, and explicit one cannot bypass scientific,
lifetime or resource admission. Shared construction and selector implementation
do not imply identical Standard/Fast defaults. No new GPU or endpoint-performance
qualification is claimed for this review repair.

The original withdrawal rationale and measurements below are historical. They
are retained to explain the rejected alternative, not as current default policy.

## Problem

The [execution audit](../../proposed/2026-11-05-standard-fast-execution-parity.md)
and [selected-force resource follow-up](2026-11-13-selected-force-resource-parity.md)
left two non-numerical profile differences: a second complete calculator/model
construction for Fast, and a Fast-only automatic FP64-anchor policy. Selecting
precision after publishing an already validated calculator is not a safe fix.
The [shared-anchor trial](../performance/2026-11-05-shared-fp64-anchor-trial.md)
also does not qualify a common automatic performance default.

## Decision

Resolve the scientific model, immutable basis snapshots and native library once.
Separate construction-time execution admission from model resolution. Ordinary
Calculator and all precision profiles run the same transport, arithmetic,
capability and initial-guess validation. The profile selector operates only
inside construction, before molecular preparation or public visibility. Every
optional attempt starts from the same resolved-model attributes, including the
unresolved automatic guess. Failed admission restores the entire previous
configuration; it cannot leak selected controls, budgets or capabilities.

### Original anchor default (superseded)

The original implementation required Standard and Fast to use
`GENERATIVEQC_KS_FIXED_ANCHOR_DELTA=1` for the
existing bounded FP64 linear-operator optimization. Unset and zero retained the
incumbent in that superseded implementation. FP64 eligibility no longer rejects a strict owner merely because its
descriptor carries a budget. Existing complete-model/provider, mixed outer-state,
ledger/chunk, capacity, final physical closure and retirement guards remain.
Invalid selectors still fail for otherwise ineligible owners.

This original proposal withdrew the previous Fast-only automatic floor instead
of enabling an incompletely qualified Standard default. It could sacrifice prior
Fast savings. Review rejected that withdrawal because missing common-default
qualification does not establish a defect in the already qualified Fast path.
A future shared automatic policy still needs independent complete-endpoint
qualification for both profiles.

Together with the resource follow-up, shared construction/resource admission
removes the identified incidental profile/budget-presence engineering branches.
The current retained Fast default remains an explicit qualified policy boundary.
This does not mean that only
SCF tolerances differ: admitted FP32 proposals/gradients, provisional screening,
physical convergence qualification, independent audits and their bounded
numerical fallbacks remain explicit policy differences. Their scientific
qualification must not be erased to manufacture
equal schedules or work counts.

## Test contracts

Host tests remove a basis file after model resolution and require every profile
to retain its snapshot. Failure injection after partial AUTO admission checks
complete rollback, not constructor call counts or source spelling. Existing
custom MethodIR, correction, fallback, target and lifecycle coverage remains.

Public DF tests exercise both the selected consumer and an explicitly injected
Python fallback. Native schema-v3 and Python DF schema-v4 receipts retain their
distinct qualification identities and work semantics; neither is relabeled.
Both must pass the unchanged independent energy/force, strict-versus-mixed RMS
and translational-force gates. The original Python-specific assertions remain
required for the injected fallback, rather than being deleted for native builds.

Scheduled WB97M-V tests compare default/off/explicit anchor selection for both
profiles through complete cold and warm energy/force endpoints and independent
moving-grid observables. These are semantic checks, not large-shape performance
or default-promotion evidence.

## Historical author evidence before review repairs

The following source/binary identities and GPU receipts belong to the original
`6f6969a6` implementation, not the later review repair or master integration.

The implementation is based on refreshed master `f590a214b`; it preserves its
new convergence, resource identity and prepared native-force defaults. Earlier
artifacts and results in the resource follow-up belong to their explicitly
identified earlier source, not this revision. Final build/test receipts are
retained under ignored `.artifacts/fast-parity-20261113/`.

Before the rebuilt core was available, the expanded ordinary-constructor smoke
suite against the earlier library passed 186 tests and failed eight CPU KS
preliminary-guess force-capability cases. Those results are not qualification of
the refreshed source. They are retained as `unified-calculator-host.log` and
must be distinguished from the final rebuilt-library regression below.
Running the unmodified `f590a214b` constructor against that same earlier library
reproduces exactly those eight failures (two passing cases), recorded in
`unified-baseline-constructor-smoke.log`; no production file is replaced for
that diagnostic.

The initial verified sccache 0.16.0 build lost its shared server during the two
large Direct CUDA compilations: server statistics reset, an old compiler child
was orphaned, and clients started compiler fallbacks. Only this task's build
processes and orphan were stopped. The shared server and both persistent caches
were not stopped or cleared. Recovery pins verified ccache 4.5.1 for both CXX
and CUDA with checkout-root `CCACHE_BASEDIR`; the actual generated compiler
command is retained alongside before/after statistics. Compiler-cache recovery
does not change the scientific source identity or acceptance gates.

Finite n1 Slurm preflight job 17310 identified missing CuPy in the default
interpreter. Job 17311 verifies the existing Python 3.12 oracle environment,
PySCF 2.14.0, GPU4PySCF 1.9.0 and the allocated RTX 5090; pytest 8.4.2 and its
pure-Python dependencies are isolated under this task's temporary test directory
instead of changing that shared environment. Both exact generated WB97M-V force
artifacts are built with the verified compiler cache and deployed with their
own manifests; neither is renamed or substituted for the other.

Baseline-only job 17315 passes the two new WB97M-V default/off/explicit selector
tests, including cold/warm observables and independent moving-grid energy/force
gates, in 180.49 seconds. Its library SHA-256 is
`9a50430e953fc7322edde32eaf63ec414a22a7b072b17eb2c9e0ba99325d56e8`.
This checks the test/oracle/artifact prerequisites, not the changed core; it is
not counted as current-source qualification.

### Final rebuilt-source qualification

The complete Release/sm120 build succeeds with ccache; a final incremental
build reports no work. The 2,137 inventoried production inputs have identity
`9a4c72b7b6a648d8632defe972baedda9eee1005a6352ae8bc4a6dc78df71cd5`.
The rebuilt core SHA-256 is
`e77b386f45880133ef1f23959a5465a9cae14b2bb09f216e1fe88cdde52922c1`.
Scheduled device probes verify the same deployed binary/source, Release flags,
disabled fast-compile and Slurm-preserved RTX 5090 visibility.

The fresh-library host run has **1,171 passed, eight failed, four skipped**:
985 focused precision/resource/lifetime tests pass; the ordinary-constructor
extension has 186 passes and the same eight CPU preliminary-guess force
capability failures. Re-executing the unmodified `f590a214b` constructor against
this fresh core reproduces all eight (two passes) in
`unified-fresh-baseline-constructor.log`. These failures remain visible; neither
their assertions nor unrelated CPU force admission are changed. The four host
skips are the explicitly scheduled cross-backend checkpoint cases below.

The first combined current-core GPU attempt, job 17316, is cancelled before
completion while separating the CPU-DF and GPU4PySCF reference environments.
It supplies no passing qualification receipt. The completed finite jobs are:

- **17317:** all nine DF cases pass: matched-control FP64 cold/frozen/moved
  parity plus four PBE0 shapes/representations through each selected-native and
  forced-Python consumer, with unchanged independent scientific gates. Two of
  the four extra CPU/CUDA checkpoint cases pass; two default-control cases fail
  because the expected `CheckpointError` is not raised. The combined receipt is
  **11 passed, two failed**, not an all-green checkpoint claim.
- **17318:** both WB97M-V Standard/Fast default/off/explicit-anchor regressions
  pass their independent cold/warm energy/force and paired-selection gates.
  The diagnostic 120-second faulthandler dump during a native execute does not
  terminate a test; the suite completes in 165.81 seconds.
- **17319:** using the unmodified base constructor with the same fresh core
  reproduces exactly the two checkpoint failures (two passes). The assertions
  and checkpoint implementation are left unchanged in this patch.

Thus all **11 targeted scientific GPU cases pass**, while the broader host and
checkpoint failures above are explicitly retained rather than hidden by test
selection or weakened gates. Receipts are `unified-final-host.xml`,
`unified-current-df-gpu.xml`, `unified-current-wb-gpu.xml` and
`unified-baseline-checkpoint-gpu.xml`. Sphinx HTML with warnings as errors,
public API docstring/contracts, the 113-entry/180-control default-promotion
inventory, Ruff, changed-line clang-format and `git diff --check` also pass.
Cache statistics are shared-host snapshots, not task-exclusive hit counts.
These results do not establish new large-shape memory peaks, automatic-anchor
profitability, identical numerical work or an endpoint speedup.

## Revisit when

Common automatic-anchor qualification includes independent complete
cold/warm/moved observables, real work/resource bounds and paired profitable
endpoint measurements for both profiles. No constructor reuse or matching
FP64 output alone supplies that performance evidence.

## Superseding default-selection decision

The [Fast-default retention decision](../performance/2026-11-12-retain-qualified-fast-anchor-default.md)
preserves the previously qualified Fast automatic path while leaving Standard
opt-in. It supersedes only the historical blanket default withdrawal described
above. It does not relabel the historical qualification above or claim that
existing HF/DF/moved-owner failures have been resolved.

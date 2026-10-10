# HF and SCF module boundaries

This page is the **current ownership and dependency map** for Hartree–Fock
and shared SCF infrastructure. It is not a chronological account of each
source extraction or a claim that every later molecular runtime test is green.

The structural decomposition tracked by
[issue #240](https://github.com/jinzhezenggroup/generativeqc/issues/240)
was delivered and that issue was closed on 2026-09-22. Its historical
production-acceptance audit recorded separate runtime failures; closing the
structural issue does **not** promote those failures to passing results.
For the rationale and pinned historical receipts, see the
[SCF decomposition provenance note](../../.agents/notes/implemented/architecture/2026-10-09-scf-decomposition-provenance.md).

## Ownership map

| Responsibility | Current owner / entry point | Boundary |
| --- | --- | --- |
| Dense reference linear algebra and mean-field arithmetic | `src/scf/reference/linalg.*`, `reference/mean_field.*` | Independent CPU reference numerics; no CUDA or method-driver dependencies |
| Initial density and occupation preparation | `src/scf/initial_guess/` | Validated input state, orbital occupations and initial guesses; consumer owns system validation |
| Method-neutral iteration limits and convergence bookkeeping | `src/solver/iteration_control.hpp`, `self_consistent.hpp` | Reusable solver control, not an HF-specific density or device policy |
| RHF/UHF CPU iteration, DIIS and proposal control | `src/scf/solver/mean_field_driver.*`, `solver/diis.*` | Consumes explicit prepared-provider and proposal contracts; scientific finalization remains method-owned |
| Prepared CPU Fock construction and resource accounting | `src/scf/fock_prepared.*` | Owns direct/DF provider composition and shared capacity observation |
| HF stationary gradient assembly | `src/scf/gradient/hf_gradient.*` | Consumes provider derivative contributions; does not own or mutate an SCF solver |
| Direct CUDA public plan and cache lifetime | `src/scf/cuda/rhf_bucket.*` | Topology/options identity, admission, invalidation, retry, diagnostics and warm cache |
| Direct CUDA Graph lifetime | `src/scf/cuda/rhf_graph.*` | Capture, instantiate/upload, replay and teardown |
| Optional exact-reference value lease | `src/scf/cuda/rhf_resident_values.*` | One-call admission, common Direct provider borrowing, local graphs and retirement before correlation |
| Coupled direct CUDA execution | `src/scf/cuda_rhf.cpp` and bounded `src/scf/cuda/direct_*` owners | Host launch/dataflow orchestration is separate from scientific device kernels |
| CUDA generic numeric services | `src/scf/cuda/eigensolver.*`, `src/solver/cuda/symmetric_eigen_handles.*` and `src/scf/cuda/resources.*` | Solver/library/stream/workspace lifetime; borrowers retain their method-specific policies |
| CUDA density fitting | `src/scf/cuda/df_plan*`, `df_source*`, `df_jk*`, `df_rhf_scf.*`, `df_uhf_scf.*` | Separate source preparation, plan lifetime, J/K, response and persistent iteration |
| Direct work preparation | `src/scf/cuda/queue_plan.*` and `direct_*` task/queue owners | Bounded ranges, compaction, launch descriptors and diagnostics, not duplicated HF equations |

The [shared Fock contract](fock_build.md),
[density-fitting boundary](density_fitting.md), and
[scientific CUDA ownership policy](../maintainer/cuda_ownership.md)
contain the authoritative deeper descriptions of those subjects.
The latter's tracked semantic ledger, rather than a file-size table here,
defines which retained CUDA regions are scientific versus runtime code.

## Dependency invariants

- Reference arithmetic imports reference utilities and standard numerical
  types, not public ABI implementations, CUDA providers or DFT/CC methods.
  Initial-guess code may use core/integral data contracts, but cannot import
  method drivers or later scientific execution.
- Generic iteration-control headers remain method-neutral. RHF/UHF, RKS/UKS,
  future charge/multipole solvers and host-controlled CUDA replay may share
  iteration budgets without sharing their physical equations or convergence
  definitions. Method-specific finalization, spin and occupation conventions,
  and physical residual checks stay with the scientific method.
- The CPU mean-field solver uses an explicit prepared Fock/provider interface.
  Gradient assembly consumes the provider's signed derivative contributions;
  it cannot acquire solver state or silently replace the selected provider.
- The direct CUDA C++ driver preserves launch order, screening, error
  propagation and final-Fock/force publication. Split file ownership does
  **not** imply that a native scientific recurrence has been replaced by
  generated code.
- Resource borrowing is explicit. A Graph must be destroyed on its owning
  device **before** the stream/library owner is released; stream, BLAS and
  eigensolver handles remain with their respective runtime owners.
- Direct task/queue policy, numerical CUDA kernels and the bucket/Graph
  lifetimes cannot acquire reverse dependencies that bypass these boundaries.

The residual `src/scf/cuda_rhf.cpp` is intentionally a comparatively large
coupled execution transaction. A pure line-count split would create broad
callback/context facades without separating independently owned state.
New shared CPU modules have a 600-physical-line **review target**;
legacy execution orchestrators require an explicit owner/compilation argument
rather than a claim that every file passes that target.

## Direct CUDA execution and density-fitting scope

### Optional phase-local exact RHF values

Unset or `GENERATIVEQC_RHF_RESIDENT_VALUES=auto` selects a bounded automatic
execution schedule for a single unscreened, restricted, strict-FP64
physical-reference export. `0` disables it; `1` explicitly requests the
resident schedule subject to capability and resource admission. Other values
are rejected. Automatic routing admits only a fresh reference bucket, an
f-containing domain with uncovered generic Fock work, at least 128 Cartesian
AOs and an iteration limit of at least eight. Known warm/reused buckets,
smaller sources and fully generated/lower-angular routes retain their existing
schedule without rebuilding a phase source or recapturing ordinary graphs.
Ordinary HF/UHF, mixed/incremental Fock schedules and shell-work diagnostics do
not silently acquire this lease. Small references (fewer than 64 public AOs),
iteration limits below four and incompatible canonical domains retain fallback.
The automatic policy is a conservative cold/domain heuristic, not a universal
profitability model or a promise of eight actual iterations. A supplied seed
in a fresh bucket is not assumed to be an already converged reference.

The common Direct provider builds immutable full-range canonical values once
and applies its existing compensated FP64 J/K consumer for each physical Fock
build. Shared RHF assembly applies `h + J - K/2`; SCF/DIIS, exact final Fock,
reference reconstruction and physical/canonical acceptance gates are unchanged.
No fitted reference, CPU integral oracle or molecule-specific dispatch is added.

Admission first charges the complete mandatory reference peak, conservative
provider device/host preparation, Cartesian compensation, two public raw
matrices and diagnostics. Optional values have an 8-GiB ceiling and must also
fit available device memory with a 256-MiB runtime reserve. Resource/capability
refusals retain the already admitted exact evaluator; driver/numerical failures
propagate instead of masquerading as pressure. Preparation finite-audits values
before publication. A completed action must report zero recurrence evaluations
and a canonical census matching the source inventory before reference export.

The lease borrows the RHF stream and owns separate iteration graphs. Every
admitted call recaptures those graphs, including changed/restored geometry when
explicitly forced. Refusal uses ordinary bucket graphs, capturing them lazily
if an earlier resident-only call never created them.
All exits drain work, retire graphs, then release source values and scratch;
the borrowed stream is never destroyed. No optional values survive into
CC/Lambda or appear in retained-bucket capacity. The physical reference's
numeric capacity includes the transient phase peak. Progress scopes distinguish
submitted/built source work, completed Fock actions, refused admission and the
mandatory versus complete phase capacity; graph-capture callbacks are not
counted as completed actions.

The direct path separates a method-facing bucket (identity and cache),
a Graph owner (capture and teardown), generic runtime allocations/libraries,
and reusable scientific launch owners. The bucket supports the existing
topology/options admission, cached state and retry behavior; the driver retains
the coupled numerical launch sequence. The separately built angular-force
kernel has its own device-link constraints: do not change standalone versus
relocatable compilation without checking the real NVCC build graph and endpoint.

CUDA DF is not a single interchangeable `cuda_rhf` execution mode.
Raw/metric source setup, bounded J/K contraction, source-backed derivatives,
persistent solver state and final-state publication have distinct owners.
The DF source and complete gradient contracts live in
[density fitting](density_fitting.md),
[DF final-state selection](df_final_state.md), and
[DF derivative consumers](df_derivatives.md).
Moving a contraction from `.cu` to ordinary C++ lowers a CUDA-source line
count, but is **not** evidence that its scientific arithmetic was retired.

## Structural and runtime verification

Run the dependency/size/ownership guards before claiming a structural cleanup:

```bash
python tools/check_scf_structure.py --json
python -m pytest -q tests/python/test_scf_structure.py
python tools/report_cuda_ownership.py --check
```

These checks catch dependency reversals and ownership-inventory failures;
they do **not** prove native CUDA runtime correctness. A source-level refactor
also needs the appropriate CPU and allocated-device RHF/UHF direct/DF
energy/force, prepared-batch, changed-geometry, resource/failure and
independent-reference gates. Preserve production build identity and complete
cold/incremental-build observations where compilation is part of the claim.
See [Validation](../maintainer/validation.md),
[Performance engineering](../maintainer/performance_engineering.md), and
[Evidence retention](../maintainer/evidence_retention.md).

## Historical acceptance, without a stale project-status claim

At its pinned 2026-09-22 source, the
[production-acceptance audit](../../benchmarks/results/acceptance-closeout-20260922/README.md)
recorded a clean Release/CUDA build and separately failed runtime tests.
The #240 issue documents the reconciliation and split between completed
structure and subsequent runtime bugs, including
[charged DF-UHF final-state convergence #1041](https://github.com/jinzhezenggroup/generativeqc/issues/1041).
Neither the historical failures nor their attempted repairs are a statement
about all tests on today's `master`; check the current CI and the relevant issue.

The [direct-native source-matched results](../../benchmarks/results/scf-direct-native-rtx5090/README.md)
retain numerical comparisons, builds, incremental objects and matched timings.
The [decomposition decision](../../.agents/notes/implemented/architecture/2026-09-18-direct-hf-host-control.md)
explains why bucket/Graph lifetime is separate from numerical launch order.
Do not copy old object-size tables or issue-by-issue progress into this
current-state implementation map.

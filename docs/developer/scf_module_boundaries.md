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
| Common phase-local ERI source | `src/scf/cuda/direct_jk_source_lease.*` | Bounded immutable full-range FP64 values and raw compensated restricted J/K; no method assembly or graph ownership |
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

The RHF adapter borrows the stream and owns separate iteration graphs; its
method-neutral `DirectJkSourceLease` owns values and compensation only. Every
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

### Optional phase-local Global-Hybrid KS values

Unset or `GENERATIVEQC_KS_RESIDENT_VALUES=auto` selects bounded phase-source
reuse for restricted exact-FP64 full-range Global-Hybrid solves.
Explicit `0` retains the incumbent schedule; `1` requests a source without
bypassing scientific or resource admission; other values are rejected.
Automatic selection includes screened through-d cold
candidates as well as the incumbent generic f-shell domain, with at least
64 public/128 Cartesian AOs and eight permitted iterations. Through-d
candidates require a positive prepared Schwarz threshold; actual compact
census and complete capacity admission decide whether values fit. These
eligibility gates do not predict profitability or actual iteration count.
Warm reuse and externally supplied initial densities skip source preparation, including
provider cloning and row sorting. An unseeded moved geometry is a new cold
candidate; a moved solve carrying a warm snapshot is not. Native cold initial
guesses retain explicit owner provenance through `CudaKsSeedOrigin`; a non-null
density pointer alone does not distinguish a production cold guess from a
user seed. Seed normalization and the initial-guess mathematics are unchanged.
Forced selection still requires at least 64 public AOs, four permitted
iterations, a compatible canonical source and complete capacity admission.

KS keeps its prepared mathematical specification, exact-exchange coefficient,
XC, DIIS, convergence and final-state consumers. Source replay preserves the
prepared Schwarz threshold on the canonical Direct contraction. The source
produces raw J/K only; KS applies its scientific coefficients exactly once.
For a positive preparation threshold, the source freezes the existing
geometry-only Schwarz row tails and stores only their compact canonical values.
Construction and replay use identical bucket offsets and bounded work extents;
a replay threshold below the frozen preparation mask is rejected. This does
not introduce a different screening approximation or density-dependent cache.
No ERI recurrence is evaluated during replay, and no derivative ERIs are cached.
The source-only Direct factory skips unrelated generated/MD/range leases and
releases sort staging after the row-construction fence. KS supplies its existing
raw J/K outputs; only compensation and audits need additional source scratch.

Admission includes mandatory KS, grid, XC, DIIS, eigensolver and provider
device/host storage before the optional source's preparation peak, compensation
and diagnostics. The existing internal `ScfOptions::reference_memory_budget_bytes`
can bound this complete phase inventory; zero imposes no finite internal phase
budget. KS default `auto` permits at most 16 GiB for screened through-d values
and retains the 8-GiB ceiling for generic f-shell values; explicit KS `1`
permits at most 16 GiB. All retain the 256-MiB free-device reserve and
complete admission. RHF's unscreened layout and 8-GiB ceiling are unchanged.
After metadata preparation, admission rechecks live free memory without charging
already allocated provider storage twice; the complete numeric budget still
includes the conservative preparation peak.
Public resource-ledger requests retain the original exact route until their
complete resource planner explicitly reserves optional source storage. This
includes forced selection; a diagnostic cannot bypass a public memory budget.

UKS, fitted J/K, range/nonlocal corrections, lower precision and incremental
J/K retain their existing paths. Resource/capability refusal preserves exact
fallback; driver or numerical failure propagates. KS retires compiled/graph
borrowers before releasing the phase source at terminal completion, restart,
failure or destruction. Values never persist into force/response evaluation,
another solve or another geometry. Existing Hybrid graph qualification is not
widened. `CudaKsTransfers` reports source construction, completed submitted Fock
actions and numeric/source peaks separately from SCF iterations.

After a successful source-backed solve that updates the warm cache, KS retires
the source and evaluates incumbent J/K once at the unchanged final density.
The compensated energy uses the retained XC at that density and anchors the
next warm solve to its actual provider, rather than comparing different FP64
summation schedules. This cache-only handoff leaves the returned energy,
physical Fock, residual, convergence gates and derivative generation unchanged.
It is counted in `fock_builds`, `post_scf_fock_builds`, the finalization precision
timeline and `CudaKsTransfers::warm_energy_rebase_fock_actions`, not as an SCF
iteration or resident replay. Frozen warm caches do not perform this work;
failed handoffs invalidate the new cache and cannot publish a successful state.
Ordinary warm/seeded solves still skip source preparation and pass all existing
energy, density and physical-residual gates.
If forced `1` admits another source on a warm solve, that phase does not borrow
the incumbent energy baseline: its first resident build establishes new energy
history, as the device-chunk path already does. This is an explicit bounded
fallback, not a cross-provider convergence shortcut.

The bounded default decision and its limits are recorded in
`.agents/notes/implemented/performance/2026-10-11-hybrid-phase-source-default.md`.
Source-matched default routing, warm handoff and complete E+F qualification
recipes are in `benchmarks/results/hybrid-phase-source-default-20261011/README.md`.

The independent raw J/K, same-grid PySCF energy and optional complete cold energy
endpoint gates are in `tests/python/test_ks_phase_resident_values_cuda.py`. Run
them only through a finite-time Slurm allocation with
`GENERATIVEQC_DFT_CUDA_TEST=1`; add
`GENERATIVEQC_KS_PHASE_VALUES_ENDPOINT_TEST=1` for complete energy endpoints,
including provider preparation and source/Fock work counts. The legacy direct
C-API KS method owner rejects nuclear-gradient requests; its optional
`GENERATIVEQC_KS_PHASE_VALUES_FORCE_TEST=1` is a separate capability gate, not
force evidence. The high-level Python PBE0 endpoint instead uses the existing
compiled stationary force owner and supports complete E+F with matching AOT
assets. Qualify the 96-atom def2-SVP endpoint with `benchmarks.readme_pbe0`, its
independent reference, unchanged grid/tolerances and complete preparation plus
E+F timing. Retain strict-FP64 source construction and Fock counts to distinguish
actual reuse from a successful bounded fallback.

The direct path separates a method-facing bucket (identity and cache),
a Graph owner (capture and teardown), generic runtime allocations/libraries,
and reusable scientific launch owners. The bucket supports the existing
topology/options admission, cached state and retry behavior; the driver retains
the coupled numerical launch sequence. The separately built angular-force
kernel has its own device-link constraints: do not change standalone versus
relocatable compilation without checking the real NVCC build graph and endpoint.

Full-range nuclear derivatives can partition a mixed f basis by the borrowed
class-major topology: the 21 s/p/d quartet classes use separate low-order,
weighted order-four/five, cooperative order-six/seven and materialized dddd
consumers. Every f-containing class remains with the bounded generic recurrence,
including low-total-order f classes. Admission requires both resident derivative
schedules, compatible recurrence modes, the same plan's topology and the original
256-lane launch contract. Missing preconditions retain the complete mixed owner;
the existing whole-basis s/p/d schedule is unchanged. Class pages preserve the
physical pair orientation, original screening and generated-class ownership,
without new resident storage or another full-domain traversal. This changes
execution partitioning, not derivative algebra or the number of physical passes.

CUDA KS final-state export keeps the shared physical/canonical acceptance policy
and orbital-energy weighted-density expression. The optional
`GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=1` route computes validation products
from authenticated resident D/F/C/energies with borrowed matrix-library resources
and the shared tensor primitive. It reuses four phase-local iteration matrices
and an arena-charged bounded reduction packet array; it does not upload detached
host matrices or allocate another dense arena/handle. The default is `0`; other
values are errors. Small or unavailable-library owners, and repeated exports
with a published stationary-weight lease, retain reference validation without
recycling leased storage. An admitted provider failure is explicit, not a silent
reference retry. See the
[resident final-validation decision](../../.agents/notes/implemented/performance/2026-10-10-ks-resident-final-validation.md)
for layout, lifetime and qualification rationale.

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

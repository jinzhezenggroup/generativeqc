# Performance engineering

GenerativeQC performance work must optimize scientific work and data movement, not
only kernel throughput or peak scratch size. This document records cross-cutting
rules for CUDA, generated integrals, response, DFT, and post-HF execution.

## Direct J/K policy ownership

The [tuning inventory](../../manifests/maintenance/direct_jk_tuning_inventory.json)
distinguishes retained bounds, migrated policy owners and pending qualification.
Persistent-ERI admission belongs to `SmallHfProfitabilityProfile`: the no-calibration
AO crossover remains 16, and resident ERI bytes are capped by the profile ceiling
and 1/32 of known target memory (at most 256 MiB for an unknown target).
This is resource admission, not a screened-quartet/reuse profitability model.
The bucket lifecycle queries target facts once and passes them to both packing
and execution. Packing records the resolved admission in immutable topology
identity and prepares Cartesian transforms when resident storage is rejected.
Resource-only small-HF queries remain device-inert and use the unknown-target
fallback; their admitted route agrees with their arena's resident-ERI storage.
They refuse their native-only bound when `GENERATIVEQC_HF_MATRIX_PROFILE` names
a profile, without reading that file or querying a device/provider.

`DirectPersistentScheduleProfile` owns the persistent force/Fock angular-order
domains emitted with the compiler's resident schedule header. Allocation, queue
reset and kernel dispatch consume that same generated definition. The default
domains remain seven force orders and six Fock orders. Smaller compiler profiles
retain fixed-grid workers for the remaining orders; they are testable build
inputs, not shipped runtime autotune winners. The 1024-task-per-shell-pair staging
bound remains an explicit retained bounded-schedule choice.

These ownership and resource contracts do not establish a new schedule winner or
close device qualification. Persistent-ERI profitability still needs actual
screened-work/reuse inputs and complete endpoint storage/setup accounting;
angular queue-domain profiles remain compiler inputs. Any promotion must
retain independent energy/force gates and matched cold/warm/moved-warm RHF/UHF
endpoint timings with semantic work counts. Synthetic-target host tests prove
legality and cache behavior, not GPU performance.

### Opt-in FP64 matrix calibration

`GENERATIVEQC_HF_MATRIX_PROFILE` selects a trusted local calibration file for
ordinary CUDA Direct-HF preparation. No measured winner is shipped or enabled
automatically. `HfMatrixProfile` is the canonical native record: admission
requires an exact source identity, NVIDIA provider, driver/runtime/toolkit and
cuBLAS versions, release/fast-compile mode, loaded runtime/driver/cuBLAS ELF
build IDs (including the native library, binding compiler/linker options), and
the complete target resource record. No GPU name, molecule,
benchmark identity or inferred peak throughput selects a recipe. Unsupported
identity observers (including unauthenticated import trampolines) retain the
incumbent. Matrix-only/physical-reference exporters retain their separate
provider-initialization and capacity contracts and do not consume this profile.

The strict whitespace-separated v1 record has the following ordered fields:

```text
generativeqc-hf-matrix-profile-v1
source_identity <64 lowercase hexadecimal characters>
provider nvidia
versions <driver> <runtime> <toolkit> <cuBLAS major> <minor> <patch>
build_mode <release 0|1> <fast-compile 0|1>
native_build_id <loaded GenerativeQC ELF build ID>
runtime_build_id <loaded ELF build ID>
driver_build_id <loaded ELF build ID>
cublas_build_id <loaded ELF build ID>
target <major> <minor> <warp> <threads/block> <threads/SM> <blocks/SM> <registers/SM> <shared/block> <opt-in shared/block> <shared/SM> <SMs> <global bytes> <FP32:FP64 ratio> <clock kHz> <memory clock kHz> <bus bits>
bounds <minimum dimension> <maximum dimension> <minimum matrices> <maximum matrices>
evidence_sha256 <64 lowercase hexadecimal characters>
native <launch ns> <effective FP64 FLOPs/ns> <effective bytes/ns>
cublas <launch ns> <effective FP64 FLOPs/ns> <effective bytes/ns>
provider_setup_nanoseconds <complete first-use setup ns>
end
```

The native qualification test uses
`query_hf_cuda_matrix_profile_context` to obtain execution-owned stack facts
without constructing a BLAS handle. Bounds must cover the actual AO dimension
and both physical-system count and physical-system count times spin count;
they must not extrapolate measured
rates beyond their qualified workload domain. Six primitive rates must be
positive and finite; setup must be finite and nonnegative. The existing
`launch + max(FLOPs/rate, bytes/rate)` model conservatively charges the **entire**
provider setup cost rather than guessing an amortization from the maximum SCF
iteration count. Nonfinite estimates retain the no-profile threshold.

Admission validates structure and compatibility, not the truth of a measurement:
the evidence digest identifies the retained external qualification artifact,
not an authentication signature or an automatically verified speed claim.
Missing, oversized (>8192 bytes), incomplete, invalid or incompatible files
fail closed. Files are reread before every preparation/replay; accepted profile
contents and provenance participate in immutable topology identity. Packing
resolves the matrix admission once; arena planning, optional context creation
and the immutable execution binding consume that same decision. Context/resource
failure retains the existing explicit native retry. Without an admitted profile,
the native/cuBLAS dimension crossover remains 17.

The opt-in Slurm-only `generativeqc_hf_matrix_profile_cuda_tests` target checks
real preparation, native/cuBLAS bindings, plan invalidation, RHF/UHF batches,
energy-only and energy/force cold/warm/moved-warm endpoints against independent
PySCF fixtures. It uses **synthetic ordering inputs**, not measured tuning
winners; its diagnostics report endpoint wall time, actual iterations/Fock
builds and owned device bytes, not a complete memory-peak or speedup claim.
Run it through the same finite Slurm allocation as the queue test below.

The native `generativeqc_direct_persistent_queue_cuda_tests` target checks actual
device queue resets, optional FP32 buffers and guard words against the emitted
profile. Run it only inside a compatible Slurm allocation with a finite time
limit, for example:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:03:00 build/generativeqc_direct_persistent_queue_cuda_tests
```

This storage test does not replace the complete-endpoint qualification above.

## README production comparisons

The README can report **default-grid endpoint latency** when that is the
requested experiment. Keep stock XC/VV10 quadratures and disclose force-response
or solver overrides. Engines may retain their native convergence criteria while
being checked against common observable error gates; unlike residual definitions
must not be equated numerically. Retain complete endpoints, warm populations,
executed controls, work and provenance. Qualification limited to sampled static
geometries must say so. Default timings alone do not establish the broader
accuracy-qualified production ranking or fastest-configuration claim below.

For a performance-only figure, `benchmarks.readme_defaults_report
--include-unqualified` displays every complete measured population, including
default-grid or fitting accuracy failures. Keep the independent qualification
results in the accompanying tables and receipts; this display option changes
neither numerical gates nor their outcomes. Incomplete calls do not become
complete latency populations.

An accuracy-qualified HF/DFT headline comparison measures the cost of reaching a common
scientific accuracy target with each engine's best qualified production
configuration, not the cost of executing an artificially identical amount of
work. Here, "best" means the fastest configuration among the documented,
independently qualified candidates that were actually measured for the declared
workload, not a claim of a global optimum or the fastest individual sample.

Publication requires:

- **A shared scientific task.** Keep geometry, charge, spin, Hamiltonian,
  functional, orbital basis, method-defined corrections and requested outputs
  fixed. Declare direct versus density-fitted treatment and auxiliary bases;
  do not silently compare different approximation contracts. Use the same
  hardware and declared resource limits. Different execution plans are allowed.
- **Independent accuracy qualification.** Declare common energy and force gates
  before choosing configurations. Each engine must pass against an independent
  reference whose grid convergence is established for the method. Agreement
  between two coarse-grid calculations is insufficient. Validate analytic forces
  with the selected moving quadrature, including grid/partition response, and
  include displaced geometries rather than qualifying only one stationary point.
- **Independent production configuration.** Let each engine select supported
  radial/angular grids, pruning, screening, precision and solver settings that
  meet those gates. Do not disable a reference engine's supported optimization
  solely because GenerativeQC lacks it. Likewise, do not enable unqualified
  pruning or a lower precision in GenerativeQC to make the comparison look
  symmetric. Record exact settings and engine versions for every selected case.
- **Complete endpoint measurements.** Measure setup plus the first complete
  energy-and-analytic-force call for cold results. Apply the same declared
  initialization and reuse policy to both engines, retain all required warm
  repeats, and report their median and variability rather than selecting the
  fastest observation. Retain source/binary identities, compilation-cache state,
  resource settings, SCF iterations and Fock-build counts. Reference densities or
  oracle work must not silently enter a production endpoint.
- **Visible grid and nonlocal work.** Record semilocal and nonlocal grid sizes
  separately, pruning rules, density thresholds and active point-domain sizes.
  Record actual VV10 pair work when available; otherwise identify any pair-count
  estimate as an estimate. A smaller constructed grid need not reduce the
  already density-screened pair domain by the same fraction, and neither ratio
  establishes a complete-endpoint speedup.

Matched-grid runs remain useful implementation diagnostics and regression
evidence under `benchmarks/results/`, but are not the headline production
ranking unless their configurations independently satisfy this policy. Existing
fixed-grid drivers and renderers do not by themselves establish production
qualification. Missing references, incomplete endpoints, failed gates and
timeouts stay visible in retained evidence, not as ranked performance results.
If selected production configurations have not yet been qualified and measured
for both engines, mark the README comparison pending rather than relabelling
historical timings or publishing estimated speedups.

An independent numerical reference may outlive a native implementation revision
when its scientific inputs and reference software remain applicable. The README
qualifier keeps same-revision checks by default. Its explicit `--reference-reuse`
manifest binds each complete original oracle receipt to the audited producer,
driver, method, orbital basis/representation, charge and spin. Geometry, package
versions, reference controls, grid stability, CPU/GPU agreement and every sample's
observable errors are still checked. Keep original producer identities and the
input review; never relabel old receipts. This numerical-reference reuse does
not permit mixing native timing revisions in one figure.

Separate production and reference installations use a version-2 reference-reuse
manifest (`generativeqc.readme-oracle-reuse.v2`). Each receipt-bound entry keeps
the original producer, driver and scientific-input contract and additionally
records `reference_packages`, a list of exact accepted `measurement_packages`
maps, and a nonempty `review` explaining the scientific-input audit. No package
version is rewritten. Unlisted environments fail closed; original version-1
contracts still require equal package maps. Every energy/force, grid-stability
and CPU/GPU spot gate is recomputed independently of this provenance review.
Slurm and LSF receipts retain their actual scheduler identifiers and device
visibility; an LSF allocation must not be labelled with a fabricated Slurm ID.

The LDA/PBE pruning qualification entry point is
`python -m benchmarks.grid_pruning_gpu --atoms 96 --output .artifacts/pruning96.json`
inside a finite Slurm allocation. It measures fresh complete native and
GPU4PySCF energy/force owners, with process/artifact caches warmed and without
cross-sample converged-density reuse. Native uses its versioned policy;
GPU4PySCF uses its own NWChem-pruned radial/Lebedev grids. Shared dense-reference
gates, reference-stability margins, raw samples and physical work counts decide
which measured candidates qualify. This limited water/STO-3G qualification is
not a general VV10, basis-set or whole-README production ranking.

## Optimization method taxonomy

Treat performance as a stack of optimization opportunities rather than as a
kernel-only problem. The categories below are deliberately orthogonal: a complete
endpoint may need several of them, but each candidate should identify which
mechanism is expected to remove time and which measurements can falsify that
hypothesis.

| Symptom | First optimization questions | Evidence to collect |
| --- | --- | --- |
| Work grows faster than the scientific problem requires | Can the algebra be reassociated, factorized, symmetry-reduced, screened, sparsified, or solved with fewer expensive operator applications? | semantic work counts, symbolic degree, active-domain sizes, iteration/operator counts |
| The same expensive source is regenerated | Can one production owner generate it once and feed several consumers, or keep it resident across phases? | source evaluations/passes, cache/residency identity, bytes regenerated |
| Large intermediates are written and immediately reread | Can producer and consumer be fused, or can the producer contract directly into the final reduction? | intermediate bytes, traversals, launch count, endpoint time |
| GPU time is dominated by packing/copies or low useful bandwidth | Is the layout appropriate for the consumer, can indexed/packed/local layouts avoid dense work, and can staging be removed? | H2D/D2H bytes, gather/scatter bytes, achieved bandwidth, active occupancy of the logical domain |
| One generic kernel handles very different workloads poorly | Should execution plans specialize by operator, derivative order, shell/layout class, target, or size while sharing one scientific definition? | per-class work/time, register/local/shared-memory use, fallback fraction |
| Many short kernels or host waits dominate | Can a larger solver region, graph/device-tail execution, batching, or fewer host synchronizations preserve the same semantics? | launch count, synchronization count, CPU gaps, device-idle intervals |
| Transfer/preparation and device work alternate serially | Can bounded prefetch, asynchronous copies, or double buffering overlap independent work? | copy/compute overlap, idle gaps, buffer peak, stream/event trace |
| Compute is throughput-bound in numerically tolerant components | Can compute/storage/accumulation precision be selected independently while retaining strict publication checks? | precision provenance, FP32/FP64 throughput, numerical margins, complete endpoint time |
| Iterative response/CC work repeatedly applies an expensive operator | Can preconditioning, warm state, recycling, block/multi-RHS execution, or control-flow changes reduce the number of exact applications? | exact/provisional action counts, residual history, complete solve time |
| Runtime is good but specialization/compilation is excessive | Can equivalent generated variants be canonicalized, packaged AOT, cached, or left to bounded JIT fallback? | compile/cache identity, artifact count/size, cold compile time, runtime coverage |

The default order of attack is: reduce semantic work first; then remove
recomputation and materialization; then fix layout/data movement and choose the
right execution plan; only then spend effort on low-level kernel tuning. Launch
control, overlap, and mixed precision should be driven by a measured residual
bottleneck rather than enabled merely because the mechanism exists.

### Reduce mathematical and semantic work

The highest-value optimization is usually work that never executes.

- Reassociate or factor contractions when the compiler can prove a lower
  symbolic cost without changing the declared equation. TensorIR is the preferred
  owner for these rewrites.
- Exploit exact symmetry, packed domains, shell/pair topology, and repeated-index
  structure before expanding work into dense ordered domains.
- Use scientifically legal screening or active/local domains to remove work
  whose contribution is provably outside the requested contract. Screening
  policy must be part of the execution identity and must retain the required
  final physical validation.
- Distinguish logically different consumers. Coulomb and exchange, value and
  derivative, short- and long-range operators, or dense and occupied-factorized
  contractions may share one scientific source while needing different optimal
  execution plans.
- For iterative methods, count expensive operator applications as semantic work.
  A better preconditioner or reusable Krylov state can dominate a faster kernel
  if it materially reduces the number of exact actions.

Do not infer complexity from lexical loop depth alone, and do not trade exact
scientific semantics for a cheaper approximation without changing the method or
provider identity explicitly.

### Reuse source work and prepared state

Generate expensive source work at the widest lifetime for which its scientific
identity remains valid. Typical reusable state includes shell-pair topology,
geometry/Boys/Rys data, molecular grids, AO active maps, integral/source tiles,
metric/factorization results, transformed occupied blocks, and prepared solver
metadata.

Prefer one owner with explicit borrowing/lifetime rules over independent
method-specific caches. Reuse is valid only when geometry, operator, basis,
precision, screening, generation, and stream/event dependencies match. A stale
or ambiguously identified cache is a correctness bug, not a performance feature.

The detailed source-reuse and residency rules below remain normative.

### Fuse producers, consumers, and reductions

Materialization is justified only when its reuse value exceeds its storage and
traffic cost. Look for sequences such as

```text
produce full intermediate
write intermediate
read intermediate
permute / weight / reduce once
```

and prefer a generated fused region that applies the permutation, weight,
denominator, response cotangent, or final reduction while the source values are
live. Common targets include AO/grid consumers, derivative contractions,
triples/response tiles, normalized partition derivatives, and permutation-heavy
tensor equations.

Fusion must preserve a readable scientific owner. Do not replace a canonical
equation or AD graph with a second hand-maintained formula merely to obtain a
fused kernel. Keep an unfused or generic route as an oracle/fallback where the
specialized region has bounded admission conditions.

### Choose representation and layout for the consumer

Dense canonical tensors are not always the cheapest execution representation.
The planner may choose packed-symmetric, indexed, block-sparse/local, tiled, or
factorized layouts when the mapping is exact and explicit.

In particular:

- keep grid-local AO work in active/indexed AO dimensions instead of expanding
  inactive rows and columns only to skip them later;
- keep symmetric pair domains packed when downstream consumers can operate on
  the packed identity directly;
- gather or pack once for several contractions rather than once per consumer;
- prefer contiguous/coalesced device access and library-compatible layouts when
  they reduce total endpoint work, not merely one kernel's instruction count;
- avoid GPU -> host -> GPU round trips when ownership permits direct device
  consumption; and
- account for scatter/reduction contention introduced by a sparse or indexed
  representation rather than assuming fewer values are automatically faster.

A representation change must report both the reduction in logical work and any
new gather, scatter, metadata, or packing cost.

### Specialize execution plans without duplicating science

One scientific operator may have several execution plans. Specialization may be
by operator role, derivative order, shell/angular class, tensor/layout class,
precision, target architecture, or bounded size/resource regime.

Useful specialization includes generated shell-class kernels, separate J/K
schedules, value-versus-force schedules, target-specific tile/block geometry,
library-backed GEMM for regular contractions, and generated kernels for
irregular/fused domains. The selection boundary belongs in compiler/runtime
planning, not in molecule names or benchmark identities.

Specialization should shrink the expensive generic residual domain rather than
create a parallel scientific implementation. Retain a bounded generic fallback
for unsupported classes or resource regimes, and record how much production
work still uses that fallback.

Default to mathematically equivalent generated optimizations after verifying
exact producer/ingredient proofs, capability and bounded resource admission.
Keep explicit disable controls and fail-closed fallback. Monitor representative
ingredient/spin/resource classes in CI rather than per-functional whitelists.
Absent hardware work counters do not alone veto a correctness-qualified default;
complete, source-matched endpoint observations remain necessary for speedup claims.

### Tune GPU resource use after the work is right

Use low-level tuning after semantic work, representation, and dispatch are
credible. Inspect at least:

- registers per thread, local-memory spills, and stack usage;
- achieved versus theoretical occupancy and active warps;
- block/warp geometry and tail utilization;
- shared-memory capacity, bank behavior, and reuse;
- global-memory transaction efficiency and achieved bandwidth;
- instruction mix, FP64/FP32 throughput, and dependency/latency stalls; and
- atomic/reduction contention.

A high register count or low occupancy is a diagnostic lead, not an automatic
cause. A kernel can be latency-, bandwidth-, instruction-, launch-, or
work-amplification-bound with similar occupancy numbers. Validate resource
changes with profiler counters and the complete endpoint. Static compiler
resource reports help choose candidates but do not replace device measurement.

### Reduce launch and host-control overhead

When semantic work is already small, repeated launch and synchronization can
become the bottleneck. Candidates include:

- batching compatible work into fewer launches;
- compiler-generated fused solver regions;
- CUDA Graph replay for stable prepared regions;
- device-tail or persistent control when bounded failure semantics remain
  observable;
- eliminating redundant evaluations of the same accepted solver state; and
- replacing unconditional host fences with explicit event dependencies.

Do not hide a required convergence, error, publication, or lifetime check merely
to reduce synchronization. Profiling and clean timing should also remain
separate: instrumentation fences can change the schedule being diagnosed.

### Overlap independent preparation, transfer, and compute

Use asynchronous execution only when a timeline proves there is latency to
hide. Bounded prefetch and double buffering can overlap preparation or transfer
for tile N+1 with computation on tile N, but they also consume more retained
memory and make ownership more complex.

Charge every in-flight buffer to the resource plan, use explicit events for
ownership transfer, retain a single-buffer fallback for tight budgets, and
measure actual overlap rather than adding copy and kernel durations. If the
endpoint is compute-bound with no device idle gap, extra buffering is normally
the wrong optimization.

### Treat precision as an execution-schedule dimension

Precision is not a global on/off performance flag. When numerically legal, plan
compute precision, storage precision, and accumulation precision independently
per component. Throughput-heavy contractions may use lower precision while
sensitive denominators, reductions, residuals, stationarity checks, energies,
and forces remain in FP64.

Every mixed-precision path must expose its actual precision provenance, retain a
strict-FP64 route for the same scientific equation, and pass independent
numerical gates with margin on difficult cases. Tensor-core or SGEMM utilization
is implementation evidence; complete-endpoint speedup plus numerical acceptance
is the promotion criterion. [Application precision
qualification](application_precision_qualification.md) owns the Fast-specific
runners, timing scope and evidence rules.

### Reduce expensive solver/operator applications

For response, CC, SCF, and other iterative endpoints, optimize the control
problem as well as the operator kernel. Investigate stronger legal
preconditioners, compatible warm starts, Krylov/subspace recycling, block or
multi-RHS solves, reuse of accepted residuals, and removal of duplicate
provisional/audit evaluations.

Always publish the exact final residual/stationarity check required by the
method. A provisional approximation can accelerate convergence only when the
final exact physical operator and acceptance gates remain unchanged. Report
both per-action cost and the number of actions; otherwise an apparent kernel
win can be erased by extra iterations.

### Optimize generated-code and compilation cost separately

Compiler/code-generation performance has its own objective. Canonicalize
equivalent specializations, package common qualified variants AOT, cache
immutable artifacts by complete compiler/source/target identity, and reserve
JIT for bounded uncovered domains.

Do not weaken runtime code quality solely to make CI compile faster, and do not
report a fast-build configuration as production performance. Conversely, a
runtime specialization whose compile/artifact explosion makes normal deployment
impractical is not a complete performance solution.

### Build a causal profile before choosing the next optimization

Use an attribution chain from the public endpoint down to the proposed
mechanism:

```text
complete wall time
  -> mutually exclusive phase wall time
  -> semantic work and data movement
  -> launches / synchronization / idle gaps
  -> kernel or library activity
  -> hardware counters and resource use
```

Parent wall intervals must be mutually exclusive. Nested profiler scopes,
CUDA-event intervals, API durations, and summed kernel times are explanatory
views and must not be added together as if they were independent endpoint
components.

Use source-matched clean timing for the endpoint and separate instrumented runs
for attribution. Nsight Systems establishes timeline and causality; Nsight
Compute or equivalent counters diagnose selected kernels. Static resource
reports and the [CUDA timing estimator](../developer/cuda_time_estimator.md)
can prioritize experiments but cannot establish a speedup.

A useful performance issue should therefore state the expected mechanism in
falsifiable terms, for example: fewer exact source evaluations, fewer bytes
materialized, a smaller active domain, fewer launches/fences, higher useful
throughput for unchanged work, or fewer expensive solver actions. If none of
those changes, a speedup hypothesis is incomplete.

### DFT endpoint and direct-force diagnostics

`python -m benchmarks.profile_dft_endpoint --capture warm --method r2scan-3c
native ... --repeats 5 --reference reference.json --output diagnostic.json`
runs the unchanged README scientific driver and stops after cold plus the first
frozen warm call. Run it through finite `srun`. Its `diagnostic_only` output is
not a completed 12-call timing cohort. Keep the reference protocol's repeat
count even though the diagnostic deliberately stops early. Nsight Systems can
use `--capture-range=cudaProfilerApi` to capture only the warm call. The nested
host ledger subtracts immediate children and reports its reconciliation error;
CUDA kernel/API/event intervals are separate views, not additional wall time.

`GENERATIVEQC_DF_TRACE=components.jsonl` adds intrusive CUDA-event regions.
`GENERATIVEQC_DIRECT_FORCE_CENSUS=work.jsonl` separately enables synchronous,
resource-ledger-charged counters for the bounded direct-force dispatcher. The
`generativeqc.direct-force-census.v1` records separate candidate pages, block
rejection, shell screening, admission, and actual consumer work. Only a true
`complete_bounded_spd_domain` certifies complete coverage of the qualified
bounded s/p/d domain. Generated-owned work or other fallback routes invalidate
that coverage; unavailable execution/recurrence fields are `null`. Primitive
products count actual consumer traversals after their final admission, not
theoretical shell capacity. The diagnostic refuses graph capture and drains
all borrowers before releasing its temporary counters, including on failure.

The spatial low-order force owner fills its existing 256-descriptor queue with
two 128-lane admission packets before one class-grouped drain. Its indexed page
width follows that bounded capacity, independently of GPU identity. Other indexed
consumers retain 64-candidate pages; unindexed claims retain complete block
products through their existing fallback.
Changing this width reduces indexed page claims, not quartet visits or primitive
work. Traversal route zero can also include unchanged relative order-six pages, so
its aggregate page count does not shrink by the indexed-page ratio. Compare
per-class admission, screening, AO and primitive counts, and account separately
for changed page counters. See the [queue-capacity decision](../../.agents/notes/proposed/2026-11-03-bounded-direct-force-queue-capacity-pages.md).

Non-captured full-range hybrid Direct value calls with `GENERATIVEQC_DF_TRACE`
also request a source-work census, which excludes the ordinary MD-Coulomb
dispatch. Their traced J/K times
therefore do not identify the clean endpoint's value-path bottleneck. Capture
the actual native endpoint with the profiler API and both diagnostic variables
unset when attributing its kernel time; retain intrusive traces only as
separately labelled diagnostics.

With the census enabled, `GENERATIVEQC_DIRECT_FORCE_CENSUS_SPLIT_LOW=1`
separates the generated low-order and order-six spatial consumers for event
attribution. This diagnostic repeats candidate enumeration; its measured
intervals cannot replace the ordinary combined kernel's clean endpoint time.

`GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS=relative-reuse` opts into a separate
order-six consumer that shares primitive, Hermite and Coulomb/Boys preparation
across AO components and nuclear responses. `relative-density` additionally
projects density weights into Hermite coefficients before the order-six/seven
Coulomb contraction. Factorization requires exact full-range pure J and a
complete AO-Schwarz mask; exchange, bilinear seeds, partial masks, unsupported
domains and nonfinite reordered intermediates retain the component fallback
before any force scatter. These selectors are captured during preparation;
the automatic policy is unchanged, and `spatial-reuse` or `0` selects the
retained spatial or incumbent route. The census's `density_contraction`
subrecord counts density terms and Hermite products separately from component
contractions, including preparation performed by declined attempts. None of
these invocation counts is a hardware FLOP count.

The independent low-order arithmetic rollback is
`GENERATIVEQC_DIRECT_WEIGHTED_FORCE_RESPONSES=0`. Its automatic FP64
relative-response path shares differentiation of the existing weighted scalar;
it does not alter screening or the admitted primitive/source counts. See the
[compiler and fallback contract](../developer/weighted_eri.md#relative-coordinate-force-responses).
To compare the nuclear rollback with genuinely unset selection while retaining
the qualified order-six/seven defaults, run:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:05:00 env GENERATIVEQC_LIBRARY=/absolute/path/libgenerativeqc.so \
  python -m benchmarks.direct_force_reuse .artifacts/weighted-responses-96 \
  --atoms 96 --weighted-responses
```

This mode removes inherited order-six/seven overrides from both arms, records
the comparison axis, and preserves each 1800-second population cap, five warm
calls and five moved-warm calls. Do not combine it with `--baseline` or
`--candidate`, which compare the separate order-six/seven schedule axis.
An original-library comparison is additionally required to detect any cold
module-loading cost hidden by a same-image selector comparison.

Two preparation-time experimental controls retain the incumbent defaults:
`GENERATIVEQC_EXPERIMENT_ORDER7_FORCE_PAGE=8|16|32` changes only the admitted
dd×dp page width (default 32); `GENERATIVEQC_EXPERIMENT_DDDD_FORCE_CLASS_MAJOR=1`
requests dd×dd class-major acquisition feeding the existing materialized
derivative. Unknown values retain the incumbent. The dddd request still needs
the existing qualified s/p/d materialized resources and coherent topology;
ineligible owners keep their bounded fallback. Neither option changes
screening, coefficient lanes, precision, or derivative equations. Compare each
option independently in the same binary, with diagnostics disabled, before
considering default promotion.

`-DGENERATIVEQC_CUDA_XC_MATRIX_TILE=8|16|32` exposes the compiler's existing
bounded matrix schedules for a build-level experiment; the default remains 16.
It changes shared-panel reuse and block dimensions for the same density and
symmetric-potential definitions, without changing jet count or precision.
Retain the CMake setting, generated-grid-source hash and library hash for each
arm, and include ragged/tiny-map and full SCF+force controls. Alternative legal
tile sizes are not qualified production defaults. In meta-GGA campaigns,
identify the shared-branch XC prerequisite separately and report force-side
projection independently of SCF density/Vxc work.

## Work amplification is a first-class metric

A schedule can satisfy a memory budget while repeating expensive work many
more times than necessary. Planners and benchmarks should therefore report both
memory and work, including the relevant subset of:

- consumer tiles/panels and source tiles/passes;
- integral/source evaluations and generated values;
- GEMM/contraction counts and their dimensions;
- bytes generated, gathered, uploaded, downloaded, or regenerated;
- synchronization/wait counts; and
- complete cold, warm, changed-geometry, and batch wall time.

When system size or a smaller budget changes a tile shape, compare actual work
counts before interpreting a kernel slowdown. A sudden endpoint cliff often
means the executed algorithm changed its amount of work.

## Audit native high-order scalar work

Run `python tools/audit_native_complexity.py` to inventory leaf native loop nests
with depth four or greater. The default audit covers GenerativeQC production
sources under `src/` and excludes the vendored xTB native tree; use
`--include-vendored` for a broader diagnostic scan. `--format json` emits a
machine-readable inventory.

The audit separately classifies an avoidable rank-2 matrix-chain candidate when
a high-order scalar reduction writes a rank-2 target from at least three
pairwise-indexed inputs spanning four loop indices, without a genuine
three-/four-index source access. This distinguishes patterns such as a scalar
`C^T A C` implementation from an exact four-index ERI/J/K contraction. The
pre-commit `native-complexity-audit` hook runs
`--fail-on-matrix-chain`, so a reintroduced quartic rank-2 transform fails CI
and must move to TensorIR or the shared dense-linear-algebra owner.

The source audit is deliberately conservative: unclassified high-order loops are
reported, not automatically rewritten. Reports include both lexical loop depth and
an effective depth that removes simple fixed extents such as spin=2 or xyz=3. The
action classes are:

- `matrix-chain-candidate`: conservative structural match for a pure-algebra rank-2 scalar matrix chain; CI-blocking;
- `high-rank-output-materialization`: high-rank permutation/symmetrization pass;
  inspect producer/consumer fusion to remove a complete traversal/materialization;
- `high-rank-source-contraction`: genuine high-rank source access; inspect
  provider/factorization or fuse-consume ownership without assuming lower formal scaling;
- `fixed-extent-inner-loop`: lexical depth inflated by bounded constant dimensions;
  consider unrolling/fusion but do not label it O(N^depth);
- `high-order-loop`: report-only fallback requiring algebra/profile review.

TensorIR symbolic complexity remains the proof-carrying path for legal contraction
reassociation; native findings are review prompts for code that still sits outside
that IR.

### Review high-rank materialization candidates

The `Source work inventory (advisory)` job continues to report high-rank
output-materialization candidates through `audit_native_complexity.py` and
`audit_structured_materialization.py`. These are lexical/static diagnostics,
not a proof that storage is required or that fusion would be profitable.
The `--fail-on-matrix-chain` complexity regression check remains enforced.

Materialization candidates should be reviewed alongside their actual producers,
consumers, lifetime, memory requirements and complete-endpoint measurements
when optimizing the affected code. CI no longer requires whole-file source
fingerprint dispositions: an unrelated addition to a shared source file must
not invalidate another algorithm's review or block its PR. Existing numerical,
resource and independent-reference gates remain authoritative.

The original source-bound review design and why its CI gate was retired are
preserved in the [historical decision](../../.agents/notes/implemented/performance/2026-10-09-materialization-candidate-review.md)
and the [retirement decision](../../.agents/notes/implemented/performance/2026-11-03-retire-source-bound-materialization-gate.md).

The five MP2 derivative-shell stages retain bounded materializations under their
current reuse/callback contracts; the dense orbital-RHS/Lagrangian path remains
an explicitly budget-gated validation oracle. Factorized first/second storage
uses occupied support rather than zero-filled dense dimensions. These are
measured retention decisions, not automatic fusion claims or permanent source
fingerprint requirements. The [six-site decision](../../.agents/notes/implemented/performance/2026-11-06-mp2-six-site-dispositions.md)
records rejected recomputation/view alternatives and conditions for revisiting
them; [raw qualification](../../benchmarks/results/mp2-six-site-20261106/README.md)
retains independent physical derivative/full-endpoint gates, work counts and
simultaneous storage.

`benchmarks/experiments/mp2_materialization/qualify.py` is an opt-in disposable
translation-unit experiment, not a production rewrite or required-CI spelling
test. It reuses a verified cached CMake build without overwriting production
objects. Compare each storage alternative to the matched helper-retention
control, and compare that control separately to production. Use `oracle.py` for
independent PySCF/libcint complete-endpoint checks and `memory.py` in an isolated
Memray environment for fresh-process simultaneous native-heap observations.
Memory-instrumented runs are separate from timing qualification. Tiny-system
oracle timings do not authorize switching production to a dense/reference path.

## Prefer source-driven reuse

Expensive source work should normally be produced once and consumed by multiple
outputs before eviction. In particular, inspect loops of the form

```text
for consumer_tile:
    for every source_index:
        expensive_source_or_projection(source_index)
        consume(consumer_tile, source_index)
```

If the expensive operation does not depend on `consumer_tile`, either move it
outside that loop, retain/batch it under the resource policy, or demonstrate
with endpoint evidence that recomputation is faster.

The same rule applies to AO integral tiles feeding many MO blocks, auxiliary
projections feeding many response panels, grid/AO jets feeding several XC
consumers, and intermediates feeding CC/response equations.

## Reuse resident state before recomputation

Prepared plans often already own large, charged buffers and immutable metadata.
A downstream phase may borrow them when all of the following are explicit:

- scientific identity and generation match;
- the producer no longer needs the contents being overwritten;
- stream/event ordering protects the lifetime;
- resource accounting records borrowed capacity without double counting; and
- an explicit bounded fallback remains available when borrowing is impossible.

A component-local scratch allowance is not a reason to recompute a much more
expensive quantity if compatible resident storage is already owned elsewhere.
Memory safety, ownership, and work efficiency must be planned together.

## Keep oracle work out of production paths

Independent CPU/reference implementations are valuable correctness oracles,
but production CUDA setup must not construct transformed tensors, factorizations,
or derivative data solely because the oracle uses them. Record which consumer
owns each preparation result, and skip unused compatibility work on production
backends.

Likewise, avoid GPU -> host -> GPU staging when a CUDA consumer can use the
producer device data directly. Host-staged providers may remain explicit oracle
or compatibility paths, but performance claims must identify which route ran.

## Derivative and response consumers

Prefer consumer-driven derivatives:

```text
generated derivative primitive
    -> apply current response/adjoint weight
    -> bounded reduction
    -> physical gradient/HVP output
```

Avoid full coordinate-major derivative tensors when a generated kernel can
contract the final weight directly. Reuse common geometry, Boys, moment, or
shell work across requested derivative components when doing so improves the
complete endpoint rather than only an isolated kernel.

## Historical rationale

The DF-response work-amplification cliff fixed by an earlier change motivated
several of these rules. Its exact panel counts, GEMM/transfer amplification, benchmark
conditions, rejected alternatives, and revisit criteria are preserved in the
[DF-response work-amplification note](../../.agents/notes/implemented/performance/2026-09-15-df-response-work-amplification.md).
This document intentionally keeps the current policy rather than the migration
history.

## Performance qualification checklist

The [equal-spin CUDA XC qualification](xc_equal_spin_qualification.md) describes
the opt-in RKS scalar-sharing candidate and its separate point, independent
reference and complete molecular endpoint gates. Point throughput does not
authorize promotion of a complete functional or composite method.

The [experimental CUDA kernel timing tool](../developer/cuda_time_estimator.md)
accepts explicit achieved-rate calibration for offline estimates. Its retained
RTX 5090 arithmetic/copy/streaming calibrations require explicit family selection
and qualify only the recorded warm FP64 probe domain. Refined-model acceptance
uses fresh held-out median/P95/maximum error gates of 5%/12%/20%, overall and per
family. Its engineering uncertainty band is not a statistical confidence interval.
Kernel estimates do not replace the complete endpoint and numerical acceptance
gates below.

The [RCCSD(T) CPU response bundle qualification](ccsdt_cpu_bundle_qualification.md)
records the complete force-endpoint comparator and its cold/warm artifact and
semantic-work gates.

The batch comparator records the loaded native binary hash and selected kernel
profile in `native_build`. Acceptance-matrix runs also write a per-point
`.progress.jsonl` journal before cold execution, so preparation/runtime failures
retain this identity. A source hash alone cannot distinguish builds with different
compiled kernel coverage; see the
[binary provenance decision](../../.agents/notes/implemented/compatibility/2026-09-17-benchmark-binary-provenance.md).

### SCF residual and stopping-rule interpretation

The batch comparator's schema v3 records `convergence_policy` explicitly.
Equal numeric tolerances or reported SCF iteration counts do not establish
equivalent stopping rules or equal Fock work. GPU4PySCF also evaluates an
initial potential before its counted cycles. Compare complete endpoints and
independently gated energies/forces; retain stock reference DIIS unchanged.
An explicitly requested `--reference-full-fock` suppresses both incremental
potential inputs, including on RKS backends that still reuse `vhf_last` when
`direct_scf=False`. Density fitting's own `direct_scf=False` policy alone does
not assert that an explicit full-Fock override was requested.

Convergence payloads with `residual_schema_version=2` distinguish GPU4PySCF's
`density_frobenius` from `density_rms`. RMS divides the backend norm by the
square root of all density-matrix entries, including spin blocks, using shape
metadata only. An unavailable shape yields null RMS, never an assumed size.
`orbital_gradient_norm` remains the backend's unnormalized global norm and is
not the native AO commutator RMS. Callback values describe the last reported
cycle, not an additional final physical audit. First-cycle energy change uses
the backend's preceding energy when available; a cold default guess is not
labeled as a warm density seed.

Historical payloads lacking the residual version marker retain their original
meaning: GPU4PySCF `density_rms` stored an unnormalized Frobenius norm. Do not
silently rewrite archived measurements. Summary readers accept v2 and v3
artifacts, preserve their residual fields and retain the new policy metadata.

Both cold and warm reference energy-plus-force timers include returning the
forces from device to host, matching native `execute()`'s public-output
boundary. JSON/list serialization remains outside timing on both engines.

### General acceptance gates

Numerical acceptance uses the maximum error across every measured repeat pair.
Matching iteration counts only classifies timing; it cannot exclude inaccurate
samples from the energy or force gate. Energy-only runs retain absent force
errors as `null`. Legacy integral/contraction timing field names contain complete
endpoints and explicitly report that no component split was measured. See the
[all-repeat acceptance decision](../../.agents/notes/implemented/compatibility/2026-09-17-all-repeat-accuracy-gates.md).

Before promoting a new default or auto-selection policy:

1. Record the baseline endpoint and exact scientific settings.
2. Record actual work counters for baseline and candidate.
3. Explain any change in algorithmic work caused by tiling or memory limits.
4. Validate numerical equivalence with an independent oracle/reference.
5. Test at least one larger size that can expose a planner/tile cliff.
6. Report cold, warm, and changed-geometry behavior when geometry-dependent
   preparation exists; report batch and constrained-memory behavior when those
   modes are supported.
7. Keep slower/correct fallbacks available where the promoted resource or
   identity preconditions do not hold.
8. Prefer complete endpoint evidence over isolated kernel speedups.

The stable
[`default_promotion_inventory.json`](../../manifests/maintenance/default_promotion_inventory.json)
points at separately owned `manifests/maintenance/default_promotion/<id>.json`
policy records. The inventory validator combines all records in memory, scans
real production controls, and rejects missing, stale or duplicate registrations.
Do not commit an aggregated policy list: each qualification changes only its
own entry, reducing parallel-PR merge collisions.

The assembled inventory distinguishes measured `negative-evidence`, a final `retained-opt-in` decision,
and continuing `needs-qualification` research. Public precision defaults to
FP64 and AUTO remains explicit; TensorIR tuning winners require explicit selection, and
TensorIR Graph replay remains explicit. These decisions do not change existing
qualified method defaults or fulfill separate positive product-delivery gates.
Continuing qualification needs a domain owner beyond the completed policy
review. See the [decision and evidence boundaries](../../.agents/notes/implemented/performance/2026-10-13-default-promotion-closeout.md).

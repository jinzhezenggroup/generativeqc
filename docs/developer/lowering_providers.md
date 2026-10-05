# Tensor operation and provider bindings

The shared compiler boundary has three separate identities:

- **Scientific identity** references the existing TensorIR/ProgramIR equation
  owner. Providers do not define equations or qualify lower precision.
- **Semantic operation identity** describes the operation, logical axes, input
  and publication types, and mathematical semantics. It excludes backend,
  physical strides, provider, and execution precision variants. An `einsum`
  remains an `einsum` when implemented using GEMM or cooperative reduction.
- **Execution binding identity** includes the complete request, target/toolkit
  facts, compilation identity, provider versions, algorithm, layouts, fusion,
  precision, resources, and admitted fallbacks. Changing those execution facts
invalidates reuse without changing the original scientific equation.

COSX value assembly requests `AO * D`, `seed + AO^T * potential` and the
point-batched `weight[p] * ESP[p] * projected[p]` through
`dft.cosx_contraction`. The update includes an explicit donated seed in the
canonical graph. Six fixed full/tail sites bind once in the shared
`PreparedContractionSites` owner. Projection, accumulation and weighted ESP have independent
candidate diagnostics and work counters, while sharing one provider context on
the existing grid stream. Execution performs no search or descriptor allocation.
The owner admits strict FP64, audits outputs before downstream consumers, and
rejects capture until physical replay accounting is available.

`CudaCosxStagingDiagnostic` reports the resolved descriptors, all candidates and
their rejections, actual device/architecture/runtime/provider versions, per-site
calls and scalar summands, scaled elements and split publication passes,
160 KiB host binding reservation, preparation time,
provider version and retained device growth. Optional library execution needs
one additional 96 MiB device allowance for the whole owner. An insufficient
budget or unavailable provider keeps generated execution. Test-only qualification
can enable the three operations separately; production has no promoted library
profile. Weighted ESP retains one fused generated kernel, including the incumbent
increasing-column FMA chain and zero-on-invalid publication with a sticky error.
The library candidate performs a batched contraction and one in-place combined
weight/finite publication pass, without a separate numeric buffer. The canonical
request binds the existing two-node TensorIR region and all three external inputs.
Derivative projections and composed endpoint
qualification remain separate consumers under #1884.

`common.lowering_provider.LoweringRequest` carries one bounded set of admitted
`LoweringPrecision` variants. These refer directly to
`common.precision.ExecutionPrecisionSchedule` and its `PrecisionDirective`;
there is no method-local precision/provider enum. Storage, compute, accumulation,
input, and final publication types remain explicit. Alternative arithmetic needs
scientific qualification. The currently supported arithmetic mode continues to
reject TF32 and implicit tensor-core modes.

`OperandLayout` records normalized logical modes, extents, physical element
strides, alignment, read/write effects, triangle ownership and alias groups.
Virtual operands have `strides=None`. Nonnegative affine strides can express
padding and broadcast input views. Negative-stride views require explicit
materialization in this contract revision. Declared scientific symmetries remain
owned by the source IR; adapters retain their identity rather than infer them
from equal dimensions.

`collect_lowering_candidates` collects ready and unsupported provider offers for
the **same complete request**. Providers can vary precision, physical layout,
algorithm and `ScheduleTopology` fusion within that request. They cannot change
logical axes, symmetry, alias ownership, publication type or numerical admission.
The registry applies resource, determinism and capture constraints, preserving
rejections. Exceeding `maximum_candidates` fails rather than silently truncating
the portfolio in registration order.

`common.lowering_selection.select_lowering_binding` makes a deterministic
prepare-time choice using complete phase costs:

```text
prepare + expected_replays * (kernel + cast + pack + refinement + audit + fallback)
```

Costs carry a retained source and distinguish measurements from estimates. An
unknown phase is `None`, never zero. Fused conversion work has zero standalone
time only when included in the fused kernel phase; its byte traffic still counts.
Kernel timing alone is insufficient. These per-operation costs do not establish
a complete scientific endpoint speedup or promote a method default.

Selection requires target-bound offers, provider versions/source identities,
explicit required capability facts and a complete strict requested-precision
candidate. A selected mixed candidate can fall back to its same precision or to
strict arithmetic. A strict selection cannot silently narrow on resource failure.
Workspace, retained provider allocations, extra temporaries and retained device
cache bytes are summed conservatively; retained host plans/JIT metadata have a
separate limit. Caller-owned tensor storage is outside this additional-allocation
scope and remains charged by the existing storage/lifetime planner.

`LoweringBinding` is an immutable **preparation recipe**, not a native executable
or another cache. Runtime/code-generation owners must prepare the recipe once,
retain its handle/algorithm/resources and validate the context before replay.
The recipe preserves all candidates and rejection reasons. Measurements do not
invalidate an otherwise identical executable cache key. Runtime preparation
failure and OOM handling still require the native owner to prepare only the
listed fallbacks; this compiler metadata does not implement that runtime work.

`common.native_lowering.native_lowering_portfolio` emits that metadata for
`runtime/lowering_binding.hpp`. Native admission checks the complete canonical
request identity and the selected precision identity alongside its lookup index;
equal semantic identities do not permit mixing portfolios. Emission validates
resource/count literals against the compiler's nonnegative signed-64-bit contract,
and native selection retains checked resource and cost arithmetic. These records
and decisions remain preparation metadata; they do not resolve executable owners.

## Current integration

`tensor.cuda_cublaslt.CublasLtMatmulProvider` supplies pure compiler offers for
matrix contractions, using the same canonical planned request as existing
cuBLAS/generated diagnostics. Native row/column layouts include padded inputs and
transposed outputs without packing/scatter. M/N/K and batch dimensions may group
multiple semantic modes only when matching mode order and contiguous strides
prove the flattened addresses; unit axes impose no stride constraint. Original
axis identities remain in the request and provenance. The adapter supports homogeneous
pedantic FP32/FP64 and rejects unimplemented precision obligations and aliases.
Provider version and simultaneous workspace/provider/host/cache ceilings must be
explicit. Ready means preparation eligibility; algorithm selection, exact queried
workspace, native replay and opaque/lazy allocation qualification remain required.
`src/tensor/cuda_cublaslt.cuh` prepares at most eight heuristic offers once,
checks algorithm/workspace legality, and retains the selected algorithm for every
replay. It reports the algorithm ID, tile, split-K/reduction, stages, swizzle,
custom/inner/cluster configuration, version, target and exact queried workspace.
Logical output audits preserve sticky errors and ignore row/batch padding.
Capture is rejected. The opaque host and global heuristic-cache footprint still
needs external qualification; cache capacity is reported without changing global
policy. Native qualification uses explicit test reservations.
`PreparedContractions` accepts cuBLASLt plans alongside generated, cuBLAS and
cuTENSOR bindings. Enable this optional capability with
`GENERATIVEQC_ENABLE_CUBLASLT=ON` on NVIDIA CUDA; it defaults to OFF. The build
propagates the provider macro and link dependency together to internal consumers.
All optional plans for a shape prepare transactionally, including cleanup across
different providers. Admission charges the simultaneous per-plan reservations
and descriptor/pointer tables. cuBLASLt provenance is available by shape/slot via
`visit_matmul_provenance`, with the same context-generation checks as execution.
No scientific region portfolio selects cuBLASLt by default yet; qualified
resource profiles and complete endpoint evidence remain required.

`tensor.lowering.TensorLoweringAdapter` projects existing TensorIR nodes for CPU
or CUDA consumers. It resolves program-wide precision and node hashes once per
preparation. Tensor CUDA provider diagnostics and generated/CUB reduction offers
consume this adapter, including the actual planned layouts and donation aliases.
Legacy diagnostic-only requests retain their v1 payload. Typed extended requests
and candidates use v2 payloads.

DF-CC iteration and Lambda CUDA matrix schedules now emit typed native
contraction descriptors through `tensor.native_lowering`. Each descriptor keeps
the canonical request's scientific, semantic-template and precision identities,
original einsum labels, runtime operand extents/strides and the physical matrix
recipe. Representative AOT hashes identify templates; they are not hashes of
resolved runtime shapes. The native validator checks collapsed matrix axes
against the semantic labels, including batches and transposes.

DF MO source response also emits eight descriptors and a prepared callback
traversal from its existing TensorIR reverse program. Leading source rows and
leading reduction rows are explicit projections of the parent request; the
projection retains scientific identity, parent semantic identity and operand
order. Interior-axis slices are rejected. The caller traverses every fixed
reduction row and accumulates its contribution. The native validator checks all
eight matrix recipes against their projected axes. The CUDA response consumer
prepares all eight sites through the shared contraction table and runs its two
row passes without plan search. Method interfaces accept device inputs and a
stream; they do not accept a vendor handle or provider selector.

The homogeneous contraction-region portfolio offers strict cuBLAS, generated
CUDA and optional cuTENSOR execution. Unknown complete costs preserve a legal
cuBLAS incumbent. A remaining budget that cannot hold the library reservation
selects generated execution; optional preparation rejection releases provisional
plans before the same-precision fallback. The region cannot implement casts,
mixed arithmetic, refinement or precision audits and refuses such obligations.

`PreparedContractionRegion` in the tensor runtime owns context preparation,
provider-specific lifecycle and transactional fallback. Method emitters provide
the semantic portfolio, descriptor factory and resolved shape key. They do not
emit provider-dependent setup branches. The shared owner checks descriptor count
and its minimum binding reservation before accepting the prepared region.

Source-response admission adds descriptor host storage and every simultaneous
plan/provider/workspace reservation to its borrowed inputs and scratch. The
physical source/metric wrapper removes only the exact borrowed overlap. The
response currently owns a separate context, so cuBLAS adds the shared 96 MiB
conservative reservation even when the retained physical source owns a handle.
Diagnostics report binding bytes, selected compiler candidate identity, provider
version, preparation time, queried optional workspace and observed provider
device growth. cuTENSOR still has no qualified production resource profile;
test-only ceilings and costs exercise execution without promoting it.

`src/tensor/cuda_contraction.cuh` owns preparation, provider resources and typed
execution. Each stage pre-binds its full and tail batch shapes outside iteration;
replay checks dtype, shape, device, stream and context generation. The owner
charges descriptor host storage and the existing conservative provider allowance,
retains explicit scalar schedule fallback on preparation/allocation failure, and
audits every matrix result before another kernel can mask a nonfinite value.
Execution failures propagate without replaying partial work. This native table
currently rejects graph capture explicitly: graph replay work accounting is not
yet attached, so counting only capture enqueues would give incorrect diagnostics.

The shared matrix executor supports cuBLAS and an ordered generated CUDA
implementation, including padded views and affine output updates. Both support
FP32 and FP64 storage/compute/accumulation;
DF-CC and Lambda still request FP64. It does not yet execute the general
`LoweringBinding` portfolio or perform joint native precision/provider selection.
Mixed compute/accumulation, casts and refinement require a complete additional
candidate; the current adapter rejects them. DFT, triples precision, other provider
families and complete endpoint qualification remain in
#1886/#1887/#1888/#1889/#1890. No new scientific precision domain is enabled.

Conventional RCCSD iteration also emits typed prepared contractions for directly
representable unbatched and leading-batch layouts. Non-contraction nodes reuse
the existing generated kernels and arena plan. The native owner prepares the
table once and charges its descriptor storage plus shared provider reservation.
Insufficient dimensions/resources retain the original scalar traversal. There
is no conventional provider-selection option or CC-local vendor callback;
execution diagnostics report preparation, work counts and resource capacity.
The separately expanded independent replay remains the final numerical gate.

Physical RHF frame response prepares five stage tables through the same typed
boundary. The method options specify the complete resource budget, without a
matrix implementation selector. Admission charges all five descriptor tables
and one shared provider reservation; insufficient resources or unavailable
optional provider storage retain the original scalar CUDA traversal. The final
orbital residual always uses that independent scalar traversal, even when the
solve used prepared contractions. Exact unscreened J/K, sticky intermediate
finite checks and the independent molecular derivative gates remain required.
Diagnostics retain prepared execution calls, semantic summands and binding bytes.

The same native descriptor also supports `validate_affine()` independently of
the optional matrix recipe. `affine_contraction_initializer` emits original
TensorIR modes with zero matrix dimensions to prevent accidental matrix dispatch.
The optional `src/tensor/cuda_cutensor.cuh` provider consumes these affine fields
and prepares a reusable homogeneous FP32/FP64 plan with explicit strides. It
disables JIT, global plan caching and incremental autotuning; capture is rejected.
Workspace is queried exactly and observed retained device storage is checked
against a reservation. Opaque host allocations have no cuTENSOR query, so host
bytes are an externally qualified reservation, not an exact measured footprint.
`PreparedContractions` can bind `CutensorAffine` alongside the existing matrix
algorithms. Each plan requires an explicit `ContractionProviderReservation`;
the enclosing owner admits its workspace/provider ceilings and qualified host
reservation before calling `add`. Charge `reservation.total_bytes(plan_count)`
in addition to `storage_bytes`, including all simultaneously live shape variants.
`optional_resources()` reports queried workspace, observed retained device growth
and reserved host bytes. These observations do not qualify lazy allocation during
first execution. Zero host reservation rejects preparation. No reservation values
are production defaults.

Preparation publishes a shape only after all its plans succeed. A
`ContractionPreparationUnavailable` permits the caller to prepare another
scientifically admitted candidate; malformed requests, execution failures and
checked cleanup failures propagate. `release()` drains a live table before a
fallback is admitted. Call it outside the global allocation measurement lock and
before destroying the borrowed context/stream. Destruction uses best-effort cleanup.

Native builds opt in with `GENERATIVEQC_ENABLE_CUTENSOR=ON` and
`GENERATIVEQC_CUTENSOR_ROOT=/path/to/cutensor`, using an external cuTENSOR 2.8+
installation within major version 2. The default is OFF; CPU and ordinary CUDA
builds do not probe or link it. Enabling it requires NVIDIA CUDA and an available
header/library; wheel packaging is not implemented. The native CMake test target
`generativeqc_native_cutensor_tests` qualifies the shared and standalone paths.
Build capability alone does not admit a provider for any scientific method.
The occupied-triples energy owner can execute strict and scientifically admitted
mixed W through this table. Its single W request offers cuBLAS, generated CUDA
and cuTENSOR for both admitted precisions. The generated region owns provider
binding, casts and FP64 combination; the method receives no vendor selector.
Preparation admits all three simultaneous plans (one FP64 panel and two W
products), and diagnoses actual provider/version, arithmetic and semantic work.
Resource rejection first retains precision with generated execution; a partial
preparation failure drains both W and panel plans before that retry.

No production cuTENSOR resource profile is installed. The candidate retains an
explicit rejection until provider qualification supplies its reservation. Test
builds can inject reservations and synthetic complete ranking costs to qualify
the real method path; these controls are absent from production builds. Existing
production selection therefore retains its qualified incumbent. Pinned molecular
energy gates and complete endpoint accounting exercise cuTENSOR without promoting
its resource assumptions or claiming a speedup. Validation/benchmark adapters use
the versioned `df_triples_probe_v2` ABI with an explicit diagnostic capacity;
rebuild older adapters before running current benchmark scripts.
Production resource qualification and complete endpoint selection remain open.

Streamed DF MO source response offers cuBLASLt alongside cuBLAS, cuTENSOR and
generated execution for the same compiler region. `PreparedContractionRegion`
reserves all eight simultaneous plans before source callbacks and reuses their
cached algorithms for every row. Reservations are provider-specific; a partial
optional preparation failure drains provisional plans before selecting the
same-precision generated fallback. Diagnostics report the actual provider,
version, preparation time, work and resource counts. No production cuBLASLt
resource profile is installed, so ordinary selection retains the incumbent.

The native cuTENSOR executor fixes the GETT family and kernel rank zero at
preparation, with JIT, cache and incremental autotuning disabled. Unsupported
shapes reject preparation and retain the admitted fallback. cuTENSOR 2.8 does not
expose the algorithm selected by `DEFAULT` through its plan-attribute API; this
explicit policy permits truthful algorithm provenance. It is not a performance
ranking or a globally unique binary kernel identifier. `provenance()` returns
the resolved request, provider/runtime versions, architecture, queried workspace,
algorithm and rank. The shared table exposes these records per variant/slot via
`visit_optional_provenance`; released or stale bindings cannot report live plans.
See the [native provider decision](../../.agents/notes/implemented/architecture/2026-10-05-native-affine-cutensor.md)
for validation and integration boundaries.

The [decision note](../../.agents/notes/implemented/architecture/2026-10-04-joint-lowering-contract.md)
records the compatibility and identity rationale.

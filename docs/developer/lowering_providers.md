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

`src/tensor/cuda_contraction.cuh` owns preparation, provider resources and typed
execution. Each stage pre-binds its full and tail batch shapes outside iteration;
replay checks dtype, shape, device, stream and context generation. The owner
charges descriptor host storage and the existing conservative provider allowance,
retains explicit scalar schedule fallback on preparation/allocation failure, and
audits every matrix result before another kernel can mask a nonfinite value.
Execution failures propagate without replaying partial work. This native table
currently rejects graph capture explicitly: graph replay work accounting is not
yet attached, so counting only capture enqueues would give incorrect diagnostics.

This native slice executes the existing admitted homogeneous arithmetic through
cuBLAS. The shared dispatcher supports FP32 and FP64 storage/compute/accumulation;
DF-CC and Lambda still request FP64. It does not yet execute the general
`LoweringBinding` portfolio or perform joint native precision/provider selection.
Mixed compute/accumulation, casts and refinement require a complete additional
candidate; the current adapter rejects them. The RHF frame-response callback,
conventional RCCSD migration from #1868, DFT, triples precision, other provider
families and complete endpoint qualification remain in
#1886/#1887/#1888/#1889/#1890. No new scientific precision domain is enabled.

The [decision note](../../.agents/notes/implemented/architecture/2026-10-04-joint-lowering-contract.md)
records the compatibility and identity rationale.

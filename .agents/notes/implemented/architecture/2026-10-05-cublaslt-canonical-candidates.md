# Decision: cuBLASLt uses the existing contraction request

Status: implemented
Date: 2026-10-05

## Decision

Add a pure compiler cuBLASLt candidate for rank-2 and rank-3 matrix contractions.
Keep the exact planned TensorIR request, admitted precision schedules and physical
operand views consumed by cuBLAS/generated diagnostics. A provider recipe proves
row/column storage and nonoverlapping batches without inserting packing or result
scatter. Transposed output storage is supported directly. Equal dimensions never
substitute for semantic mode identity.

FP32 and FP64 offers require homogeneous pedantic arithmetic. Mixed arithmetic,
casts, audits, refinement, virtual views, aliases, diagonals, batch broadcasting
and higher-rank grouping remain explicit rejections. Provider choice does not
change scientific precision admission.

## Boundaries

Ready means bounded preparation eligibility, consistent with the cuTENSOR
metadata contract. No native executor or default promotion is supplied by this
slice. Explicit version and workspace/provider/host/cache ceilings are required.
Preparation must subsequently resolve a bounded heuristic, retain the selected
algorithm and exact workspace, and qualify opaque/lazy allocations. Capture and
determinism beyond the unspecified contract are not advertised.

Do not invent measured cost from layout eligibility. Avoided copies alone cannot
establish an endpoint win. The first recipe deliberately excludes fused epilogues
until existing TensorIR semantics can express and independently validate them.

## Evidence and revisit conditions

Independent NumPy einsum tests validate every input/result transpose for unequal
and unit matrix dimensions in both admitted dtypes, with and without batches.
Resource, alias, capability, precision, capture and ordering rejections are
retained alongside other providers for the same operation.

Extend grouping or fusion only when a real consumer needs it and complete native
execution/resource evidence accompanies the extension. Continue native algorithm
caching and CUTLASS/CuTe AOT implementation under #1888; parent #1886.

# Decision: resolve cuBLASLt heuristics once per prepared binding

Status: implemented
Date: 2026-10-05

## Decision

Implement cuBLASLt as a shared native provider of the same ContractionRequest
accepted by cuTENSOR and the existing matrix/generated providers. The rank-2/3
adapter validates semantic modes independently of the optional legacy matrix
recipe. Physical row/column orders, padded rows and batches execute directly.
FP32/FP64 use pedantic compute; unsupported precision and layout requirements
are never weakened.

Prepare a bounded array of eight heuristic results, validate the chosen result
with AlgoCheck, query exact workspace, and store the opaque algorithm in the
binding. Replay always passes that algorithm explicitly. Expose all public
algorithm configuration attributes with provider/runtime versions, architecture
and resolved request as provenance. An algorithm number alone is not a cache key.

## Resource and lifetime boundaries

Caller supplies explicit workspace/provider/host reservations. Host storage
includes simultaneous heuristic results and externally qualified opaque provider
state. The process-global heuristic cache has a capacity query but no exact byte
query. Record capacity and preserve global policy: disabling or resizing the
cache from one local provider could interfere with independent consumers.

Observed preparation-time device growth and queried workspace do not qualify
lazy first-execution allocations. No production resource profile or native
portfolio promotion is installed in this slice. Optional unavailable/OOM
preparation returns a rejection after checked cleanup; malformed requests and
execution failures propagate. Live release drains even the default stream.
Graph capture remains explicitly rejected pending lifecycle/work qualification.

The finite affine audit now has one generic implementation shared with cuTENSOR;
it audits logical elements, ignores padding and preserves prior error flags.

## Evidence

The n1/Slurm RTX 5090 probe checks 192 combinations of input/output transpose,
unit/unequal dimensions, batches, FP32/FP64 and beta-zero/accumulation. Independent
mode-indexed sums match exactly on dyadic inputs; NaN padding remains untouched.
Only scalar pointer alignment is supplied. Three ordinary replays retain one
heuristic query and identical algorithm/workspace provenance. Alias, stale
binding/stream, capture, insufficient host reservation, release and sticky
nonfinite failures are covered. The existing cuTENSOR probe also passes after
sharing the audit.

This is correctness and lifecycle evidence, not a complete endpoint performance
claim. Native portfolio integration, larger-shape/opaque-resource/OOM qualification,
complete cold/warm endpoint comparison and CUTLASS/CuTe AOT fusion remain #1888.
Parent: #1886. Predecessor: canonical cuBLASLt candidate metadata in #1928.

# Decision: prepare a bounded CUTLASS SIMT family for canonical contractions

Status: implemented
Date: 2026-10-05

## Problem

#1888 requires CUTLASS/CuTe AOT execution behind the same semantic operation used
by cuBLAS, cuBLASLt, cuTENSOR and generated CUDA. A kernel family must preserve
precision and scientific identities, and cannot hide preparation or module
storage inside numerical replay.

## Decision

Use the external CUTLASS 3.9.2 C++ GemmBatched SIMT implementation, with homogeneous
FP32/FP64, scalar alignment and one fixed 32x64x8 threadblock shape. The common
matrix view proof preserves semantic modes and accepts direct padded/batched
row/column layouts. CUTLASS's linear-combination epilogue implements only the
canonical alpha/beta operation; this does not rewrite any equation.

Prepare the complete parameter object once; execution only updates addresses.
Fixed small tiles avoid shared-memory attribute changes in replay. Resolve both
the matrix and finite-audit kernels before publishing a plan. There is no JIT,
heuristic search, split-K, tensor-core precision or Python CUTLASS dependency.

The native owner requires a retained artifact digest from the actual compilation
owner. Qualification hashes external header contents under logical paths, native
and compiler sources, toolchain versions and compilation flags. Version strings
alone do not identify the compiled specialization.

## Resource and failure policy

Host admission charges the owner and the concrete prepared specialization before
allocation. Only insufficient host capacity or host allocation failure permits
optional rejection. Malformed precision/layout contracts and execution failures
propagate; a partially executed contraction is never replayed via a fallback.

CUDA retains loaded AOT modules at context scope. An external qualified module
reservation is mandatory, observed setup growth is checked under the shared
measurement lock, and a violation throws. Returning a soft rejection there would
hide retained module memory from the fallback. Releasing a local plan does not
clear its module charge, and a later preparation cannot reduce the reservation.

## Evidence and limits

The real-device probe checks FP32/FP64, all input/output matrix orders, padding,
beta-zero NaN output, nonzero beta, unit axes, nonunit grouped modes, multiple tiles
and repeated replay against independent semantic addressing. It also exercises
sticky finite errors, alias/stream/capture rejection, exact host capacity and
one-byte-below capacity, and retained module accounting.

This is a native provider slice, without a production profile or method selector.
Registry integration, endpoint measurements and production opaque resource
qualification follow separately. No performance or mixed-precision claim follows
from a correct isolated contraction.

## Revisit when

Retained endpoint evidence supports another tile or a fused region with exact
TensorIR semantics. Add it as a bounded AOT candidate with its own artifact facts;
do not dispatch by molecule or method name, or infer arithmetic admission from a
GPU architecture.

## References

- #1886, #1888; shared matrix proof in #1937.
- [Current provider contract](../../../../docs/developer/lowering_providers.md).

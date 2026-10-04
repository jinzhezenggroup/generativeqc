# Decision: prepare cuTENSOR against the existing native semantic descriptor

Status: implemented
Date: 2026-10-05

## Problem and decision

The shared native contraction descriptor retained original TensorIR modes but
validated only a flattened matrix implementation. General affine providers need
to consume those same fields without requiring a pack-to-GEMM recipe. Split
affine semantic validation from the additional matrix-layout proof. Preserve
the original scientific/semantic/precision IDs; a general emitter sets matrix
dimensions to zero so accidentally using the matrix executor fails explicitly.

The optional CudaCutensorContraction prepares one immutable cuTENSOR plan on a
borrowed stream. It creates descriptors, selects the default algorithm once,
queries exact workspace, allocates it, records provider/runtime versions and
measures retained device storage against the caller's reservation. It disables
JIT, global plan caching and incremental autotuning. Replay supplies addresses
only, validates dtype/device/stream/alias/alignment and audits logical output
elements while ignoring padding. beta zero does not read the old output.

Unsupported plans, insufficient workspace and allocation failure are preparation
rejections. Execution and cleanup failures propagate; scientific work is never
silently retried. Reset drains before destroying resources. Capture is explicitly
rejected until the enclosing consumer has replay accounting.

## Resource boundary

cuTENSOR 2.x exposes exact device workspace but no opaque host allocation query.
The host-byte argument is therefore an explicit externally qualified reservation,
not an observed exact footprint. It includes this binding's retained metadata.
The test reserves 64 MiB of host metadata and up to 256 MiB of provider device
storage; these are qualification limits, not production default policies.
Production integration still needs a version-bound conservative host reservation
and complete endpoint admission evidence. Do not infer zero opaque host storage
from sizeof(handle), or advertise this isolated executor as complete admission.

## Evidence

CUDA 12.9 and the cutensor-cu12 2.8.1 package on n1, via Slurm
main/gpu:5090:1: four tests pass, including the existing matrix-provider suite
and the new affine suite. The latter executes FP32/FP64 with two differently
ordered reduction axes, transposed output, padded views and beta updates. An
independent host contraction on exact dyadic inputs supplies the oracle. Three
replays retain one prepared plan. Alias, stale binding/stream, missing host
reservation, capture and nonfinite-output gates pass. A host-only real TensorIR
projection also validates without any cuTENSOR runtime dependency.

Compilation uses ccache, with before/after statistics and logs retained under
ignored .artifacts/1887-cutensor. No timing or production speedup is claimed.

## Remaining integration

This is an optional provider implementation, not yet selected by a production
method or linked by default. #1913 supplies canonical compiler candidate metadata
separately. Production binding, two independent method consumers, full cold/warm
endpoint measurements, provider-absence/OOM fault injection and qualified capture
remain #1887 work. Mixed precision beyond homogeneous FP32/FP64 remains explicitly
unsupported rather than being changed silently.

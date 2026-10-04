# Decision: affine matrix views and a generated native contraction fallback

Status: implemented
Date: 2026-10-05

## Problem

Occupied-triples TensorIR contractions consume strided views of canonical
amplitudes/integrals and use affine output updates. A dense, overwrite-only
native binding cannot replace their method-owned BLAS callbacks. Optional
provider failure also needs a complete executable fallback under the same
semantic descriptor, rather than another scientific implementation.

## Decision

Extend the existing native projection with an explicit matrix cut/row stride
and beta. Validate those strides against the original TensorIR modes and
collapsed matrix recipe. Noncontiguous batches remain unsupported; callers
must materialize them explicitly. Compute borrowed address spans from strides
for overflow and conservative alias rejection. Output finite audit visits
logical elements only, so padding need not be initialized or finite.

PreparedContractions now accepts compiler-selected backend algorithms behind
its existing typed execution boundary. Alongside pedantic BLAS, a bounded
CUDA grid-stride provider implements explicit RN scalar multiply/add in
increasing reduction order for both FP32 and FP64. It preserves affine output
semantics and skips the old output read when beta is zero. The same descriptor,
context-generation, stream, dtype, alias, capture and error contracts apply.
All tables are preparation-only; account for their algorithm entries in the
conservative host storage bound. No method policy or provider choice is added
to ContractionRequest's scientific identity.

A context can bind generated execution without a library handle after optional
provider preparation fails. Backend execution errors still propagate; no failed
scientific work is silently replayed. Capture remains rejected until the CC
owner supplies replay work accounting.

## Evidence and boundaries

Existing DF iteration/Lambda matrix descriptors validate at 27 o/v/q combinations.
The real-device matrix test covers all transpose combinations, both dtypes,
dense batches, padded unbatched inputs/outputs, beta updates, nonfinite padding,
independent exact dyadic results, stale context/shape, aliases and sticky errors.
The first production consumer of the new affine/generated choices is a separate
triples binding PR; existing DF defaults are unchanged. This is no performance
promotion and does not claim completeness of #1886 or #1889.

## Rejected alternatives

Copying strided inputs into method-owned packed matrices would add untracked
traffic and lifetime policy. A raw vendor callback would discard the TensorIR
labels and precision identities. Auditing the physical bounding span would
incorrectly treat padding as scientific data.

## Revisit when

Additional providers can execute these canonical affine requests, or a real
consumer needs explicitly represented strided batches. Add that capability
with independent numerical, resource and lifetime evidence.

# Decision: admit CUTLASS AOT artifacts through the canonical provider registry

Status: implemented
Date: 2026-10-05

## Decision and rationale

`CutlassAotProvider` consumes the existing planned contraction request, shared
matrix view proof and admitted precision variants. It competes with library and
generated candidates without adding a scientific operation or method selector.

Ready offers require an explicit compiled SIMT family, matching header version,
actual artifact digest and qualified host/module byte bounds. A version alone is
insufficient: external headers, flags and source changes can alter the executable.
The family identifier changes with the fixed tile/algorithm ABI, and the native
probe checks its agreement with the compiler record. Native compilation asserts
the actual kernel thread count advertised in the schedule topology.

Context-retained AOT module bytes use the shared candidate's cache accounting.
They are not reported as zero merely because the kernel has no GEMM workspace.
Exact descriptor/module limits, capture and determinism use the same registry
gates as other providers. Missing or malformed evidence produces a retained
unsupported offer, not an inferred resource default.

## Invariants and evidence

Only homogeneous admitted FP32/FP64 is executable in the initial family. Casts,
audit/refinement obligations, mixed accumulation and exact-order demands remain
explicit rejections. Reproducibility of this fixed SIMT kernel does not imply
equivalence to the scalar reduction order.

Tests preserve request identity across providers and artifact changes, exercise
all evidence omissions and malformed resources, reject precision obligations,
and check transposed-output CUDA grid boundaries and exact common limits. The
native affine oracle qualifies actual family metadata and replay separately.

## Limits and revisit conditions

This adds registry eligibility, without a production cost/resource profile or
region loader. Unknown complete costs stay unknown. A new specialization or fused
region needs its own artifact, original TensorIR semantics, resource/lifetime
qualification and complete endpoint evidence before selection is promoted.

## References

- #1886, #1888 and native executor #1938.
- [Current provider contract](../../../../docs/developer/lowering_providers.md).

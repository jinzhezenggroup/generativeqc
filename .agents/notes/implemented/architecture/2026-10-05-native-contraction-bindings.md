# Decision: prepare compiler-owned native contraction descriptors

Status: implemented
Date: 2026-10-05

## Problem

DF-CC iteration and Lambda exposed FP64-only callbacks and independently owned
cuBLAS handles. Their generated matrix calls lost the semantic einsum and
precision identity at the native boundary. Lambda tail batches also needed a
bounded preparation lifetime when replacing callbacks with retained bindings.

## Decision

Project `TensorLoweringAdapter` requests into native descriptors. Keep canonical
scientific, semantic-template and precision identities alongside runtime operand
modes, shapes and strides. Retain matrix recognition in the existing emitter;
validate its flattened dimensions and transpose order against those semantic
modes during native preparation. A representative AOT request hash is explicitly
a template identity, never an assertion that different runtime shapes are equal.

Use one shared provider context per method owner. It prepares a pedantic cuBLAS
handle with explicit zero workspace, records toolkit/provider versions and device,
and invalidates descriptors on reset. DF iteration prepares three stage tables;
Lambda prepares each admitted stage for the complete batch and, when needed, its
tail. Replay reuses these tables and changes only borrowed tensor addresses.
Descriptor host bytes join the existing complete numeric budget. A conservative
second descriptor copy is reserved for construction. Table capacities and batch
variants are bounded. No per-iteration cache, JIT or heuristic is introduced.

## Invariants and retained fallback

- The scientific programs, explicit packing, contraction tree, scalar reference,
  sticky arithmetic flag and publication gates stay with their current owners.
- Shared execution supports homogeneous FP32 or FP64 exactly. Native CC consumers
  continue to request FP64; mixed accumulation/casts/refinement are rejected.
- Only optional preparation or allocation failure permits the already admitted
  scalar schedule. Execution/driver/arithmetic errors propagate.
- Every shared provider release participates in allocation-measurement locking,
  including constructor unwind and optional-arena OOM retry. The lock-held reset
  is explicit because enclosing owners also release arenas under that same lock.
- Capture is explicitly unsupported until graph replay work accounting is
  connected; preparation and execution reject a capturing stream.
- A binding is valid only with the enclosing context, device, stream and runtime
  dimensions. No process-global cache or cross-owner descriptor lifetime exists.

## Rejected alternatives

Merely moving a GEMM lambda leaves precision and science untyped. Recomputing
requests or preparing tail shapes in response actions adds hidden replay work.
Treating the representative semantic hash as a complete runtime identity aliases
unrelated shapes. Introducing a native scientific IR duplicates TensorIR.

## Evidence and limits

The host projection test compiles every DF iteration/Lambda matrix descriptor and
validates all of them at 27 occupied/virtual/batch shape combinations. Deliberate
stride, mode, precision, math-mode, identity, integer-range and rank corruption is
rejected. Initial real-device DF-CC qualification passed all 14 CUDA solver
checks on n1 under a finite Slurm allocation. Existing expanded-equation matrix tests remain independent numerical
gates. Production constructor fault injection and concurrent resource-release
checks now compile the actual shared resource owner and pass.

This is a native execution boundary slice of #1889, not completion of joint
selection: it does not yet execute `LoweringBinding` portfolios, add cuTENSOR or
cuBLASLt/CUTLASS, migrate DFT/triples/conventional RCCSD, or claim an endpoint
speedup. The legacy RHF frame-response callback remains explicitly scoped to
#1890. Real-device evidence belongs to qualification artifacts and PR checks.

## References

- #1886, #1889, #1890 and #1868.
- [Joint lowering contract](2026-10-04-joint-lowering-contract.md).

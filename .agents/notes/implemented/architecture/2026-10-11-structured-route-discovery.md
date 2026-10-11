# Decision: discover structured tensor routes without method-local selectors

Status: implemented (candidate-only compiler slice; production promotion pending)
Date: 2026-10-11

## Problem

The SCF/DFT/GFN route issues (#1873, #1878, #1879, #1882, #1883)
repeat a common pattern: a scientific contraction has a generated
implementation and potentially profitable library implementations, but the
method layer should not independently choose a vendor. Compiler infrastructure
already contains binary GEMM recognition, a canonical lowering request,
cuBLASLt/cuTENSOR/CUTLASS offers, and a strict signed symmetric rank-k
lowering. Those implementations were not exposed through one generic
route-discovery view.

## Decision

- Introduce `tensor.structured_routes.structured_cuda_routes` on an already
  resolved TensorPlan. It discovers eligible GEMM, GEMV, dot, Gram and
  matrix-chain sites without rewriting TensorIR or changing a method API.
- Present cuBLASLt, cuTENSOR and CUTLASS candidates under the **same**
  `plan_lowering_request` identity. Incompatible arithmetic, layout,
  unqualified toolkit resources or unavailable providers remain explicit
  negative candidates. This is not evidence of an executable new default.
- Recognize the exact shared-panel signed weighted Gram via the existing
  `symmetric_rank_k_request` (strict FP64, matching domains and aliasing,
  transactional mirrored output). Refactor its existing candidate builder
  for reuse by the AOT emitter and the generic discovery path; no new
  scientific formula or numerical mode is introduced.
- Use the existing `select_lowering_binding` for optional evidence-based
  selection. Incomplete phase costs fail closed, while a verified incumbent
  may be retained. Discovery alone must never change production selection.
- A matrix chain such as `A^T B A` remains its original ordered binary
  operations. Reassociation/fusion changes rounding and intermediate
  allocation and requires an independent numerical and endpoint owner.

## Rejected alternatives

- **Blindly replace all ternary einsums with GEMM**: changes numerical order,
  may violate ragged/finite/output publication semantics and can regress
  complete endpoints.
- **Method-specific `use_cublas` flags or cutoffs**: proliferates provider
  selection outside the compiler and duplicates architecture debt in #1886.
- **Declare a new provider default from microbenchmarks**: bypasses full
  preparation, resource accounting, Graph/mask correctness and measured
  cold/warm/moved E+F gates.

## Invariants

- Scientific identity and original TensorIR equations remain unchanged.
- All binary provider offers share one semantic request identity; rank-k
  candidates share the exact recognized signed rank-k request.
- Missing toolkit version, capacity or qualified measurement is not a
  zero-cost or an automatic opt-in.
- No new host copies, device allocations or native vendor calls are added.
- Existing generated/direct CUDA paths remain the sole production defaults
  until independent execution and complete-endpoint acceptance.

## Evidence

- Source-level/CPU-only tests: `tests/python/test_structured_tensor_routes.py`
  exercises recognition, shared request identities, explicit rejection of
  unqualified vendor offers, strict rank-k fallback retention, source/target
  mismatch and negative ternary cases.
- The existing rank-k emitter calls the same extracted candidate builder.
  Its candidate construction body is retained verbatim.
- **Not yet qualified**: live CUDA numerical/Graph coverage, staging and
  provider resource bounds, end-to-end HF/DFT/GFN endpoints, provider
  crossover or production callsite migration.

## Consequences

Provider competition can be inspected once per prepared TensorPlan without
teaching HF/DFT/GFN which vendor to use. The next production slice must bind
the exact selected route to an existing native provider execution owner,
retain generated fallbacks and prove resource/capture/endpoints. Until then
this is a shared candidate-discovery mechanism, not a performance claim.

## References

#1886, #1889, #1877, #1873, #1878, #1879, #1882, #1883
PRs #2058, #2083, #2099, #2174.
Agent: ChatGPT
Model: GPT-6

# Decision: consume the final KS RI-K projection in DFT-DF force response

Status: implemented
Date: 2026-10-02

## Problem

The restricted fitted-hybrid CUDA KS path can finish its physical final K with an
occupied RI-K contraction and retain the exact density-generating Cocc together
with the complete U=B*Cocc projection.  DFT analytic forces nevertheless entered
the generic prepared Fock response with only host D/W, so the response could
project B*Cocc again even though the exact final projection was still resident.

## Decision

Carry the token-validated Cocc/U lease from the CUDA KS owner directly through
the prepared Fock provider into the synchronous DF response.  The DF boundary
rechecks device, stream, dimensions, single-B storage, full-rank metric and the
current completed projection rank.  The exchange response consumes the one-shot
U lease before the Coulomb response may reuse provider scratch.

Explicit diagnostic controls remain authoritative: final-projection off, raw
occupied source, dense response space, host weights, or JK-scratch selection
retain the ordinary path.  Unsupported, stale, UKS, streamed, truncated-rank and
non-single-B cases fall back without changing the mathematical model.

## Rejected alternatives

- Expose device Cocc/U pointers through Python.  The native owner already has
  exact lifetime/provenance and Python would weaken it.
- Persist another O(naux*nao*nocc) force cache.  The final K allocation already
  exists and should be borrowed, not duplicated.
- Run Coulomb before exchange.  J response may revoke/reuse projection scratch,
  so the one-shot exchange consumer must execute first.

## Invariants

- No DF approximation, metric cutoff, exchange coefficient or force equation changes.
- No new persistent device allocation or transfer is introduced.
- The lease is consumed synchronously and invalidated by the existing provider
  scratch-generation contract.
- Existing bounded response and exact fallback paths remain available.

## Evidence

Host/source contracts pin the KS-to-response handoff, K-before-J ordering,
provider storage revalidation and explicit fallback controls.  Real NVIDIA
numerical/performance qualification is still required before claiming a latency
improvement.

## Revisit when

The DF provider owns immutable per-state projections or exposes a general
multi-consumer borrow protocol.

## References

Issue #1567; PR #1661.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Review corrections (2026-10-02)

The final-projection diagnostic is parsed before its first use; the original
ordering failed both NVIDIA and CuMetal C++ compilation. A complete producer
projection is not by itself an admission to this consumer: dense and packed-two
value storage keep the ordinary response. For automatic occupied/source
selection, a response-budget failure drains and releases the failed response
arena before one exact bounded-panel retry without the consumed lease. Explicit
occupied/source diagnostics retain their resource gate. This prevents optional
rank-squared response retention from narrowing the previously supported budget
range.

Host/source and emitted-BLAS tests cover the handoff and fallback structure.
No NVIDIA execution of these review corrections was available locally.

Agent: dot

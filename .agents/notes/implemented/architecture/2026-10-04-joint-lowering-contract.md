# Decision: one operation request for joint lowering offers

Status: implemented (compiler contract and TensorIR projection)
Date: 2026-10-04

## Problem

#1886 and P0 #1889 require providers to compete over the same science. The existing
lowering contract described resolved implementations, but Tensor CUDA renamed
einsum sites to GEMM or reduction depending on the selected schedule. Separate
precision dispatch would also prevent comparing conversion, layout, fusion and
provider cost together.

## Decision

Extend the existing LoweringRequest/LoweringCandidate API, using the shared
ExecutionPrecisionSchedule and ScheduleTopology. Keep original equation identity,
semantic operation identity and executable identity separate. Providers receive
the whole bounded precision portfolio and must retain explicit unsupported
evidence. Pure preparation-time selection includes every phase and retains
strict requested arithmetic as a legal fallback. No provider choice enters IR.

Move CastBoundary into common.precision and retain its TensorIR import binding,
so traffic validation has one owner and existing precision payloads stay stable.
TensorLoweringAdapter resolves program-wide maps once instead of recomputing
precision/hash information for each operation occurrence. CUDA diagnostics now
record an einsum's implementation separately from its semantic node identity.

## Rejected alternatives

- A distinct cuTENSOR contraction request would preserve the competition bug.
- Method-local precision/provider flags would create another policy owner.
- Missing phase costs as zero would systematically favor incomplete offers.
- Silent portfolio truncation would make registration order an execution policy.
- A global semantic-adapter cache would retain entire programs beyond preparation.
- Calling a metadata recipe a prepared device executable would hide unimplemented
  runtime allocation, heuristic/plan creation, rollback and OOM behavior.

## Invariants

Scientific admission remains external and mandatory for arithmetic changes.
Provider algorithms cannot change the publication dtype, logical modes, symmetry
or aliases. CUDA cannot fall back to CPU through selection. Capabilities are
typed facts, so integer 1 does not substitute for True. Context/toolkit/provider
changes invalidate executable reuse. Fused casts retain their logical traffic.
The stricter CastBoundary validation rejects incorrect byte counts; it does not
alter the TensorIR serialization of valid existing schedules.

## Evidence

`tests/python/test_joint_lowering.py` covers joint portfolios, conversion/audit
cost reversing a kernel winner, explicit cold/warm amortization, deterministic
ordering, negative provider evidence, aggregate resources, capture and reduction
order, stale context, missing versions/capabilities, candidate bounds and strict
fallbacks. A NumPy matrix product independently checks the unchanged TensorIR
equation. CPU/CUDA requests share semantic identity; direct and packed GEMM plans
share the same request. Existing tensor precision/planning/search/layout and
compiler dependency checks are retained.

Local qualification: 296 passed, 3 GPU-dependent/optional tests skipped; full
`ty==0.0.82 check` exited successfully (existing repository warnings remain),
compiler structure and changed-file pre-commit hooks passed. Six FP32/FP64
direct-GEMM, packed-GEMM and reduction fixtures have byte-identical emitted CUDA,
equation hashes and plan hashes against base `3e71cdca5`. Reproduction used only
Python source generation and CPU metadata/reference tests, without a GPU run.

This change contains no real-device performance measurement and makes no speedup
claim. Native preparation and full method endpoint evidence are still required.

## Consequences and remaining work

This is the first contract slice of #1889, not closure of #1886. A LoweringBinding
is a deterministic recipe for the existing runtime owners. Native consumers must
still execute its selected typed operation, retain prepared provider handles,
account for real JIT/heuristic allocations and honor rollback/fallback. Complete
DFT and CC mixed/strict endpoint gates remain required.

Continue #1889 with native consumers and the blocked #1864/#1865/#1868 plumbing;
then same-request cuTENSOR including #1885, cuBLASLt/CUTLASS, and full #1890
migration/classification with the source guard. Preserve valid scientific
precision admission in #1867/#1872. n3 is unavailable for validation; use
n1/n2/n4/n5 and schedule every real GPU run through the node's Slurm allocation.

## Revisit when

An additional dtype/arithmetic mode has independent qualification; negative
strides need a retained native view representation; or native provider preparation
requires a more exact overlapping lifetime model than the conservative extra-byte
sum. Those extensions must preserve the shared scientific request.

## References

- #1886, #1887, #1888, #1889, #1890
- #528 shared precision; #933 provider lifetime; #934 ownership guard

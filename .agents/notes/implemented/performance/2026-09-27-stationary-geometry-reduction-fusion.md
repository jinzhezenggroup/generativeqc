# Decision: fuse stationary semilocal geometry reduction into the worker kernel

Status: implemented
Date: 2026-09-27

## Problem

The shared stationary CUDA semilocal geometry path launches one 32-thread
geometry worker kernel per grid tile and then launches a second
`geometry_reduce` kernel only to sum the 32 worker partials into the XC source.

The producer and reduction are strictly ordered, share the same bounded partial
scratch, and the second launch does no independent scientific work.

## Decision

Fuse the lane reduction into the end of the existing stationary geometry kernel.

All 32 workers now reach one block barrier. A lane that encounters invalid input
sets the shared device error and stops processing additional points, but does not
return before the barrier. After the barrier:

- any device error prevents source publication;
- successful threads reduce each output coordinate over source lanes in the exact
  former order `0, 1, ..., 31`;
- the fused kernel adds that sum to the same stationary XC source buffer.

The ordinary semilocal and external/nonlocal geometry consumers use the same
fused kernel.

## Rejected alternatives

### Atomic accumulation directly into the source

Rejected because it would change floating-point reduction order and introduce
contention.

### Keep early thread returns and add a warp synchronization

Rejected because exited lanes cannot participate in a reliable block/warp
publication barrier under independent thread scheduling. The fused schedule
instead converts early exits into a lane-local validity state so every worker
reaches the barrier.

### Tune the second reduction kernel

Rejected because eliminating the redundant launch preserves the existing
reduction order while removing more orchestration than retuning a tiny kernel.

## Invariants

- Successful-path point algebra is unchanged.
- Reduction order remains lane 0 through lane 31 for each coordinate.
- No source element is published when any worker has set the device error before
  the fusion barrier.
- The same `workers == 32` bounded scratch contract is retained.
- External/nonlocal geometry seeds use the same fused reduction semantics.
- Detailed profiling charges fused reduction time to `geometry_kernel`;
  `geometry_reduction` is retained as a zero compatibility field.
- Endpoint numerical gates and complete-work counters remain authoritative.

## Evidence

The source-generation tests require:

- no stationary `geometry_reduce` kernel;
- one explicit `__syncthreads()` publication barrier;
- no thread `return` before that barrier;
- the original ascending lane reduction loop;
- one stationary geometry launch per tile in both synchronous and deferred
  runtime paths.

Existing analytic-gradient and failure-isolation GPU gates remain the numerical
acceptance tests. No endpoint speedup is claimed before matched GPU qualification.

## Consequences

The stationary source launch count decreases by exactly one launch per semilocal
geometry tile. Scratch size and source storage are unchanged. The fused kernel
contains a barrier and reduction tail, so future geometry worker-count changes
must preserve the all-worker barrier contract and deterministic lane order.

## Revisit when

- the geometry worker count changes from one 32-thread block;
- a different deterministic reduction schedule is measured and independently
  qualified;
- whole-grid fusion removes the tile-level geometry launch entirely.

## References

- #1423 shared CUDA DFT force productionization
- #1479 resident semilocal geometry response
- #1494 deferred stationary geometry drain

Agent: ChatGPT
Model: GPT-5.6 Sol

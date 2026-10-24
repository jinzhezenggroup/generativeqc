# Decision: execute bounded layout copies through the shared contraction owner

Status: implemented
Date: 2026-10-20

## Problem

The explicit transfer contract from #2642 correctly refused native execution
without an executor. Selecting a cheap packed layout cannot reinterpret the
source pointer, and reverse publication cannot be omitted for strided outputs.
A useful next #1889 slice must actually execute the complete copy/contract/
publication sequence rather than relabel metadata as a device executable.

## Decision

Add one structurally bounded execution projection for positive, exact-shape,
unbatched ordinary FP64 matrix einsums. The compiler emits the same canonical
request/candidate portfolio plus immutable native source/physical descriptors.
Source modes, shapes, precision, coefficient and beta remain bound. The initial
physical implementation uses packed row-major views; unsupported precision,
checked updates, aliases, triangular ownership and capture remain refused.

The tensor owner borrows the existing provider context and PreparedContractions
table. It performs before copies for converted reads, reuses the existing
generated or pedantic arithmetic, then publishes converted output elements.
Beta-nonzero output is a read/write seed: its dtype appears in the canonical
precision input list and its incoming data must be copied before arithmetic.
Beta-zero does not read the prior output or uninitialized packed output.

The caller precharges host storage and simultaneous numeric scratch. Replay
uses fixed-size stack bookkeeping and no allocation, provider search or retry.
Every host-checkable capacity/ownership/context failure precedes enqueue.
An unexpected driver or partial enqueue error propagates; a different provider
must never retry partially executed work. Provider-unavailable fallback occurs
during preparation through the existing strict generated candidate.

The generic native portfolio emitter still refuses transfers by default.
Only the explicit v1 adapter declares the executor. Abstract ranking, a build
option or a synthetic cost does not establish method capability/profitability.
Existing DFT/CC production portfolios retain their behavior.

## Rejected alternatives

- A second matmul/scientific implementation for the packed route.
- Automatic conversion admission from shape or a provider's layout hash.
- Scratch allocation and tuning on every execution.
- Treating a beta seed as a pure output to avoid its precision/resource contract.
- Mutable prepared views or resource facts after their validation.
- Capture support inherited merely because individual copy kernels can capture.

## Evidence and qualification

tests/python/test_layout_contraction.py compiles and executes the emitted native
projection and budget/identity refusals. The explicit NVIDIA gate compiles
tests/native/test_layout_contraction_cuda.cu with a verified compiler cache.
It uses independent long-double scalar arithmetic for generated and cuBLAS
copy/contract/publication endpoints, beta-zero poisoned output, beta read/write,
padding preservation, rejected capacities/aliases/host pointers/capture, ordinary
replay with changed inputs, provider refusal and stale-context recovery.

Raw source-bound logs, cache statistics and environment information are retained
under .artifacts/1889-native-layout-20261020/ in the single Inspire checkout.
Cost ranking in this qualification uses synthetic, explicitly estimated phase
costs to exercise selection, not measured production profitability. No complete
DFT/CC method timing or new scientific mixed-precision domain is claimed.

## Revisit when

A production consumer supplies complete phase costs and end-to-end numerical,
resource and performance qualification, or a separately justified extension
admits batched/FP32/cast/checked/alias/capture semantics. Those extensions should
consume this shared compiler/executor boundary rather than method-local switches.

## References

- #1889; merged prerequisite #2642.
- [Transfer-contract rationale](2026-10-20-explicit-layout-transfers.md).
- [Current provider contract](../../../../docs/developer/lowering_providers.md).

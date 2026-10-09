# Decision: retire unused generic solver-region mask completion

Status: implemented
Date: 2026-10-08

## Problem

Issue #835 has already delivered structured regions, bounded CUDA execution,
qualified replay/second-consumer evidence and real RCCSD implicit VJP binding.
Its sole remaining decision is whether generic `per_item_mask` completion has a
production consumer or should be retired. Keeping a schema mode and native enum
that imply execution semantics without an implementing consumer is misleading.

## Consumer audit

Originally audited on master `7bcf6ac1dc665659868b353cb480b7126fa5f770`,
and rechecked on `6b439bb003d50b6dd854e81400977195bb4bf460`:

- `RegionCompletion("per_item_mask", ...)` appears only in synthetic contract
  tests. `tools/generativeqc_cc/solver.py` uses default scalar completion.
- The native `PerItemMask` enumerator has no use outside its declaration.
  `src/dft/cuda_ks.cpp::solver_region_binding` binds `Scalar`, including existing
  semilocal replay and the bounded global-hybrid chunk use of this executor.
- `SolverRegionCudaExecutor` previously incorporated the enum into replay identity
  without implementing active-mask semantics. Its callback owns scientific work.
- GFN2's specialized SCC graph/device-tail controller is not a generic mask
  consumer. Adding another graph owner only to exercise this mode would duplicate
  lifecycle ownership; the prior #835/#977 investigation rejected that approach.

## Decision

Admit only scalar completion in both the Python contract and native executor.
Remove the unused native enumerator and dead mask-output role. Reject unsupported
native numeric values before graph state, metrics, callbacks or publication can
change. Old mask payloads fail closed on replay; no scalar conversion is inferred.

Preserve schema-v1 scalar serialization, including the reserved
`active_mask: null` field, native scalar value zero, binding layout and replay key.
Removing the null field or the whole completion object would unnecessarily change
existing scalar region identities and invalidate RCCSD response provenance.

Ragged batches retain their existing method-owned controllers. This retirement
moves no convergence thresholds, DIIS, failure isolation, max-iteration handling,
or warm-state publication into the generic compiler/executor.

## Rejected alternatives

- Retain a deprecated but accepted mask mode: leaves an unimplemented scientific
  execution promise, and still silently admits numeric mode values natively.
- Force GFN2 or KS into a generic ragged adapter: no demonstrated production need,
  duplicate lifecycle ownership, and fresh GPU qualification would be necessary.
- Remove scalar completion fields or increment the schema: invalidates correct
  existing scalar artifacts for no scientific change.

## Invariants and evidence

- `test_solver_region.py` pins the pre-retirement scalar fixture identity, validates
  unchanged strict roundtrip/resource reuse, rejects old well-formed masked
  payloads and unknown modes, and preserves derivative fail-closed behavior.
- `test_solver_region_cuda_executor.py` executes a cached host C++ probe with a
  fake CUDA runtime: unsupported values cannot mutate a captured scalar graph or
  submission metrics, the scalar graph remains replayable, remaining-step bounds
  and width invalidation remain unchanged. This is not GPU evidence.
- `test_cc_solver.py` uses pinned independent CCSD/FCI endpoints and tests a real
  scalar solve followed by exhaustion, injected numerical failure and recovery.
  Failed work cannot modify the accepted immutable warm input; failure results
  retain owned last-finite state and reconstruct the same bounded scalar region.
- `test_ks_chunk_control_schedule.py` runs the actual KS control body as a host
  scheduling test. It checks successful warm publication, continued iteration,
  failure, iteration exhaustion, inactive items, frozen warm state and the
  physical maximum-residual gate. No CUDA numerical qualification is claimed.

## Validation on the refreshed base

On 2026-10-09, the focused CPU suite against master
`6b439bb003d50b6dd854e81400977195bb4bf460` passed **189 tests**, with **12
explicit skips** for full native-provider endpoints because `libgenerativeqc`
was not available. The suite includes `test_solver_region.py`,
`test_solver_region_cuda_executor.py`, `test_cc_solver.py`,
`test_cc_lambda_solver.py`, `test_cc_lambda_response.py`,
`test_ks_chunk_control_schedule.py`, and `test_program_ir.py`.

All eleven repository structural gates, changed-file Ruff check/format,
clang-format 23.1.2, and diff whitespace checks passed. Directly loading the
base-version scalar contract confirmed the same scalar payload and fixture hash
`8aaddba979c201f4e008b9bd7145acb108e581ae343f3e0de5b06ee8ffc8bff5`.

Host probes and native Lambda JIT used verified ccache 4.14.1 with the existing
cache and retained before/after statistics. The preferred sccache 0.16.0 launcher
could not execute in this sandbox (`Operation not permitted`); the allowed
ccache fallback completed the same tests without a source change. This validates
CPU contracts and host scheduling only, not CUDA execution or performance.

## Revisit when

A concrete production consumer needs a single region with different active items
and cannot use its current owner. Any new mode must include an executable mask
contract, bounded-step and failure/publication semantics, matched endpoint tests,
and real-device evidence if it changes CUDA execution. A synthetic mask fixture
alone is not a reason to restore this surface.

## References

- #835, #847, #899, #977, #370, #1697, #1699
- [Original structured-region decision](../architecture/2026-09-21-structured-solver-regions.md)
- [Original native execution decision](../architecture/2026-09-21-cuda-solver-region-executor.md)
- `docs/developer/program_ir.md`

# Decision: cache molecular dispatch metadata, never molecular results

Status: implemented
Date: 2026-11-13

## Problem

Issue #2293 attributes substantial fixed overhead to the complete molecular
Array API endpoint. The November 4 attribution proposed caching source input
requirements, but that proposal was not an implemented or qualified fix. A warm
call also recreated generic/scientific TensorSpecs even when its shape/dtype
specialization was already captured.

## Decision

Construct TensorSpecs only on a shape/dtype specialization miss. Continue
concrete-array, supported-dtype and finiteness admission on every invocation;
new shapes/dtypes must still satisfy explicit scientific TensorSpecs.

Cache each bound program's live source names and first-/second-derivative
requirements by its immutable logical hash, within the existing executor owner.
Adding a source invalidates those requirements, including an earlier empty
selection. Closing the owner clears them. The cache contains neither feeds nor
geometry values, directions, gradients, results or native calculator state.
Geometry evaluation, native work, publication checks and diagnostics remain
per-call. Snapshot VJP keeps its existing explicit saved-forward semantics.

## Rejected alternatives

- Reusing results at equal geometry would change the fresh-native-endpoint
  contract and would confuse dispatch metadata with scientific state.
- Bypassing TensorIR for an assumed identity gradient would require independent
  proof for composed geometry, nonlinear objectives and higher derivatives.
- Removing result checks, diagnostics or native work to meet the timing target
  would weaken correctness and invalidate the matched-work comparison.

## Evidence and qualification boundary

Behavioral tests cover changed geometry, multiple live-source specializations,
detached prior results, warm nonfinite rejection, scientific shape/dtype
admission, recovery after native failure, concurrent replay and closed owners.
The polynomial source is an orchestration fixture, not a QC numerical oracle.

The combined targeted host run passes 479 tests with 19 native/device-dependent
skips. The broader Array API run passes 2603 tests with 21 skips and 36 failures
inside the independent `array_api_strict.clip` oracle (integer bounds on floating
arrays). All 36 failures reproduce against unchanged base `fc9bbe6be`; this
change does not alter those tests or the oracle.

The source-matched n4 Release CUDA build (`GENERATIVEQC_CUDA_FAST_COMPILE=OFF`,
verified sccache 0.16.0) exceeds its initial 20-minute bound. A cache/object-reusing
resume is stopped without a completed library. The outstanding native CUDA
translation units are not replaced with compiler-oriented fast-compile output
to manufacture endpoint acceptance. Build/cache logs and prepared qualification
scripts remain in ignored `.artifacts/array-api-2293-20261113/` and the isolated
remote `/tmp/gqc-2293-20261113/` checkout. No new n1/n4 endpoint measurements or
real-device scientific test results are claimed by this change.

Real-device endpoint evidence, when available, is recorded separately from these
host checks. No host test or reduction in metadata work establishes the
two-node H2 <=0.2 ms gate, complete exact-J/K inventory, peak memory or residency
qualification. Keep `gpu_performance_qualified=false` until those independent
requirements are satisfied. No RHF equations, tolerances or SCF policy change.

## Revisit when

New measurements identify tensor execution or another outer stage as the
remaining bottleneck. Retain complete native/Array API endpoint controls rather
than promoting an isolated metadata microbenchmark.

## References

- GitHub issue #2293 and its November 4, 2026 measurement comment.
- `docs/maintainer/performance_engineering.md`.

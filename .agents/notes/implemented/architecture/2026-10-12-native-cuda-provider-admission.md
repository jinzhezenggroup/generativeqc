# Decision: compose retained CUDA providers with numeric admission

Status: implemented (private stationary qualification; no public force promotion)
Date: 2026-10-12

## Problem

The static stationary controller reused TensorIR `Context`, whose numeric arena
used raw `cudaMalloc`/`cudaFree`. Its reported arena polynomial did not authorize
the actual allocation against the prepared request. A separate per-controller
grid-provider allowance also let several retained controllers each spend the
same available capacity. Neither a small arena nor a bounded serialized loop
establishes a bound on all simultaneously retained owners.

## Decision

Shared TensorIR arenas now use the existing numeric resource allocator and retain
its ledger owner until actual release. No scientific equation, artifact/source
identity override, or replay-time allocation is introduced. Host registry OOM
stays distinct from device OOM, so optional smaller-device-arena retries do not
retry a host bookkeeping failure.

`DeviceProviderAllowance` reserves non-numeric admission capacity in the same
ledger. The stationary controller reserves the existing 96 MiB grid-provider
allowance before constructing either arena. Its shape preflight and optional
center-geometry selection use the actual remaining request capacity, including
other retained numeric/provider owners. Measured provider retention still must
fit the allowance. Stationary users drain/die before the grid, and the allowance
is released last. Shared ledger lifetime survives observation-handle closure and
release on a different submitting thread.

The numeric v1/v2 ABI and journal retain their original meaning. A provider
reservation has no fabricated allocation identity, successful requested-byte
sample, or numeric event. Private v3 separately returns request limit, numeric
live/peak, provider-reserved live/peak, and combined admission peak under one
lock. These are admission bounds, not a complete CUDA context/RSS census. Never
add a provider reservation again to an already planner-withheld allowance.

Stationary executor guards now preserve TensorIR's device/host OOM statuses 2/3.
The native adapter retains them through nested creation; the controller decodes
stationary and native-grid status taxonomies separately. A below-minimum pure
shape preflight throws OOM without performing CUDA work or fabricating an actual
allocator rejection. Partial preparation releases both actual arenas and the
reserved provider capacity before recovery.

The additional-controller shape authority is factored into
`cuda_stationary_capacity.hpp`, using the actual emitted arena polynomial, packed
AO layout and grid storage helper. Private `generativeqc_resource_stationary_cuda_v1`
returns its minimum/cached arenas and combined admission, provider allowance,
tile and geometry scheduling metadata without creating a CUDA owner. The
constructor calls the same helper and checks the packed basis size. The query
is only a planning bridge: native model/context admission and Python planner
cutover are not implied by the existence of a profile's capacity metadata.

## Rejected alternatives

- Recording the provider allowance as an invented numeric CUDA allocation would
  corrupt allocation journals and conflate a conservative bound with measurement.
- Subtracting the allowance only from each constructor's local maximum does not
  compose multiple retained controllers against one request.
- Clearing ownership at observation exit would forgive still-live cached arenas.
- Matching error messages to infer OOM would misclassify driver or numerical
  failures; swallowing host OOM into device OOM would authorize unsafe retries.

## Invariants and scope

- No numeric or combined admitted live ownership may exceed the request limit.
- Exact minimum capacity selects the existing uncached geometry fallback;
  below-minimum admission performs no allocation, and failed preparation leaves
  no retained numeric/provider charge.
- Warm stationary replay allocates no additional numeric storage.
- Numeric v1/v2 and journal semantics remain unchanged; v3 is additive/private.
- This changes shared TensorIR CUDA allocation ownership, including consumers
  other than the stationary controller, so shared real-device regressions are
  required in addition to the controller's independent scientific receipts.
- Public SDK/Python force routing remains unchanged. A single legacy Python
  serialized 512 MiB staging cap is not a per-item retained native-controller
  plan. Public cutover still needs complete native item sizing, full endpoint
  admission/receipts, installed compiler-free execution and performance gates.

## Evidence

Focused host runtime/final-state tests pass 2/2. The current CPU build passes
89/89 CTest tests; 49 targeted host contracts and 196 shared resource/lifecycle
contracts pass. The compiler audit covers 528 modules with zero dependency
errors, CUDA ownership covers 358 files, and Sphinx warnings-as-errors passes.

Real-controller tests cover exact/below minimum, optional cached geometry,
multiple retained owners, separate v1/v2/v3 receipts, partial device/host OOM,
recovery, release outside observation, and ledger lifetime after handle closure.
The twelve-case complete stationary test additionally checks the actual arena
and provider receipt and allocation-free warm replay. Each fixture also repeats
a fresh cold SCF and requests its first stationary gradient at the exact native
minimum, without prewarming integral or W consumers. Below-minimum rejection
performs no numeric allocation; the exact-minimum full gradient agrees with the
independently qualified cached route to `1e-12`, with two actual arenas, no dense
frame export and complete charge release after closing outside observation.

The capacity bridge passes 70 host contracts and its portable workspace tests;
the current CPU library passes 89/89 CTest tests. Slurm qualification passes 4/4
native tests with all twelve cold-minimum receipts, 9/9 public lifecycle tests,
4/4 targeted HF resource cases and 7/7 shared TensorIR regressions. The twelve
independent PySCF 2.14.0 fixtures accept energy, complete gradient and final D/W.
The current qualified CUDA library is
`118652fb70178b66139491e659a2aede5c9c393e7823342521dab32aa71dc307`.
Constructor/admission memcheck on this library reports zero errors/leaks.
Full endpoint memcheck on the same library and the updated minimum-budget test
executable also reports zero errors/leaks, covering all twelve cached and twelve
cold-minimum complete gradients, ragged rollback/nonconvergence and 18-AO leases.
Its separate source manifest, executable SHA, library SHA and Slurm receipt are
retained under `native-stationary-minimum-*`.

The initial shared TensorIR invocation had two loader failures because its JIT
DSOs could not locate `libcublas.so.12`; five tests already passed. Supplying the
installed CUDA library search path restores all seven without a repository code
change, and the initial failure and launcher/cache receipts are retained. Full
cached endpoint memcheck also passed on the earlier resource-instrumented
`c2bb5dc4c6adfa905ed263ea8c3cc7e2c289015d3f3156426ef7545de513398b`
library, not the later capacity-refactored library. Do not relabel that result
as exact-head uncached/cold-minimum sanitizer evidence.

## Revisit when

The provider exposes a reliable requested/retained-allocation contract that can
replace its conservative allowance, or a shared native planner adopts explicit
provider leases across all production consumers. Do not infer that change from
one `cudaMemGetInfo` delta or a smaller measured arena.

## References

- Follow-up: [per-item native force fleet admission](2026-10-12-native-cuda-force-fleet-admission.md)
  integrates this retained-provider reservation with prepared dispatch and the
  common planner; the historical qualification above remains source-specific.

- #2227 R1/R2; #933/#934 resource/prepared lifecycle gates.
- [Static stationary controller](2026-10-12-native-cuda-stationary-source-controller.md).
- `src/runtime/resource_ledger.hpp`, `src/runtime/resource_cuda.cuh`,
  `src/tensor/cuda_runtime.cuh`, `src/dft/cuda_stationary_controller.cpp`.
- `tests/native/test_runtime_workspace.cpp`,
  `tests/native/test_native_stationary_cuda_binding.cpp`,
  `tests/native/test_ks_resident_weights.cpp`.
- Ignored `.artifacts/2227/native-admission-*` source manifests and validation logs.

# Resident ordinary ABI and extension execution contracts

Status: implemented, host-only tests
Date: 2026-10-23

## Decision

Extend the [resident host ABI seam](2026-10-20-resident-host-abi-contracts.md)
to compile the ordinary emitter unchanged for control passthrough plans. A
test-only GraphContext/CUDA dependency header provides queued host transport,
allocation metadata and deterministic probe responses. It does not implement
scientific kernels, device allocation, or Graph capture. Capture requests fail
explicitly in the fixture.

The generated ordinary and resident exports execute on the same fixture owner.
External-static mode must reject both execution APIs before initialization,
reject an incorrect payload size and double initialization, and allow the
zero-payload control plan after initialization. Embedded mode has no external
initialization export. Both APIs transfer distinct control arrays correctly.

Post-run extensions now execute through the production extension argument.
Tests cover admission, owner/metrics/profile forwarding, positive and negative
failure statuses, exceptions, recovery, and omission after an arithmetic-status
failure. The default extension-free fixture also compiles and loads, so an
undefined default hook cannot pass through a source-string assertion.

Compile complete emitted input-validation kernels separately with serial host
CUDA lanes. Independent NumPy permutations check FP32/FP64 finiteness, positive
and negative symmetry signs, multiple symmetries, and absolute/relative tolerance
boundaries. Each damaged symmetry leaves the other constraints intact. This is
an arithmetic unit test, not evidence that device launches execute these checks.

## Guard audit and remaining scope

| Removed source assertion | Executable replacement |
| --- | --- |
| Ordinary export and static-state strings | Compile/load/call the ordinary ABI and resident ABI on one owner |
| Resident error-handler spelling | Existing positive/negative status injection, diagnostic, drain and recovery checks |
| Validation function/tolerance strings | Complete generated validation arithmetic against independent array permutations |
| Absent default hook spelling | Default resident artifact compilation, loading and execution |
| Post-run call/return spelling | Declared/undeclared hook execution, failure propagation and pointer forwarding |

Keep the floating/computed-output span assertion: control passthrough arrays do
not qualify computed FP64 output storage. Keep planner pinning, lease lifetime,
dependency/cache identity, and extension-source identity checks. Do not use the
new host seam to claim production Context lifetime, CUDA scheduling, kernel
arithmetic, molecular parity, or performance qualification.

The source-extracted DF retained-cache retry probe remains separate migration
work under #2296. The Direct coefficient and ordered-ring tests already execute
independent host oracles; this change does not replace or weaken those tests.

## Validation

Run the resident tests with `VIBEQC_RESIDENT_HOST_CONTROLS=1` to include the
existing opt-in corruption/formatting controls. Required compiler fixtures make
missing native prerequisites fail instead of silently dropping the replacements.
Use the verified compiler cache and finite compile/link timeouts. The broader
host regression includes all seven issue audit candidates, TensorIR CUDA plan
tests, and compiler identity tests. No production source or device behavior is
changed, and real-GPU qualification remains separate.

On Linux with Python 3.13.9, G++ 11.4.0 and ccache 4.5.1, the nine-file
regression passed 295 tests with no skips. After tightening fresh-diagnostic
and hook-pointer assertions, the final resident suite passed all 56 cases,
including the 34 opt-in controls. These are host results only.

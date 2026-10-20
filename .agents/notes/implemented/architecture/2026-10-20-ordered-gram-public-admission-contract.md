# Decision: keep private kernel packets outside host admission

Status: implemented
Date: 2026-10-20

## Problem

This supersedes the private-packet observation contract in
[the original ordered-Gram admission decision](2026-10-19-ordered-gram-host-admission.md).
PR #2621 merged as `1e0aed564c27af8a2fd6b7cb3ee93e71d1691fd7` while
review thread `PRRT_kwDOUFEw6s6uBPJ8` remained unresolved. Its recorder decoded
private kernel arguments 0 through 19. A consistent reorder of the final two
producer parameters and their call arguments failed that required test despite
preserving semantics. Avoiding the private kernel name did not remove the
implementation-layout dependency.

## Decision

The host recorder treats the private packet as opaque. It checks explicit public
grid, block, shared-memory size and stream forwarding, actual production-DSO
interception, zero CUDA submissions on rejection and a mandatory positive hook
hit for admission. Private packet decoding, the memcpy helper and expected
payload fields are removed. Kernel data, normalization and work-record routing
require behavioral device/output evidence outside this no-GPU admission gate.

## Rejected alternatives

Retaining fixed parameter positions rejects semantics-preserving refactors.
Introducing a private packet schema would merely move that implementation
contract into another gate. A copied preflight helper cannot establish actual
public-wrapper admission or DSO interception. The consistent producer reorder
remains an opt-in migration control, not a permanent source-shape assertion.

## Invariants

Keep the original 13 rejection / three admission scenarios, all 37 extended
rejections, 300 independent integer byte-boundary cases, poisoned borrowed state,
unreadable addresses, disabled history, public optional-input invocation and
discovery EOF. Missing interception fails closed. The original 378 arithmetic
scenarios, device-output tests, production code, public ABI, CMake, required CI
selection, timeouts and parallelism remain unchanged.

## Evidence

The complete cleaned driver was compiled from a snapshot based on
`27b79cc8006ec65ff721a259986a7b9c23db1a34`. All 2,382 archived input files were
checked against Git blobs with LF conversion disabled. Six relevant qualified
inputs, including the final LF driver, match committed bytes. Baseline and
corrected DIIS translation-unit DSO CTests pass using the existing CMake helper;
the unchanged arithmetic pytest passes its 378 scenarios. The corrected driver
passes private-helper rename and consistent private-parameter reorder. All 25
guard mutations, wrapper bypass, admission-nbf misrouting and missing
interception are rejected (28 controls). Private kernel-nbf misrouting survives.

Clang-format passes. Verified sccache 0.16.0 launcher commands, before/after
statistics and opt-in receipts remain under `postmerge-lf-receipts` in
`/inspire/hdd/project/chemicalreaction/ordered-host-2296-1019/`. Earlier CRLF
snapshots are retained separately and do not establish LF byte identity. The
local component build retains the earlier generated SDK; it does not replace
required CI execution against the complete production shared library.

## Consequences

The original note's 29-rejected claim describes its historical packet-aware
observer and does not transfer to this contract. Private kernel-nbf misrouting
is unobserved here, and current device-routing qualification is UNAVAILABLE.
Retaining an existing device-output test is not a new GPU PASS. This test-only
correction changes no production numerics and relaxes no independent device or
scientific acceptance gate.

## Revisit when

Any production kernel, dispatch, synchronization, math or cache-lifetime change
needs its applicable independent output/device qualification. Extend public
admission tests only for a changed public contract, without pinning a private
packet layout.

## References

Refs #2296; remaining migrations stay open. PR #2636 implements this correction
following the original #2621 packet finding and its own note-supersession review.
The [historical decision](2026-10-19-ordered-gram-host-admission.md) retains the
prior contract and its evidence, including the stronger claim now superseded.

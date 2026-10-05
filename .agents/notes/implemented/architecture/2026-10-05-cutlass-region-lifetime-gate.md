# Decision: keep call-local CUTLASS regions unavailable

Status: implemented
Date: 2026-10-05

## Problem

The source-response region destroys its shared table when construction throws.
The CUDA primary context outlives that C++ object and can still own unmeasured
modules. A later response call creates a fresh region with neither the failed
state nor its retained charge. The single-plan and shared-table quarantine fixes
cannot preserve an obligation after both owning objects have died.

A host API fault probe using the exact eight emitted source-response descriptors
reproduced a failed second load with 2,097,152 simulated retained bytes. A new
region then admitted generated execution under a 536,120-byte budget, reporting
11,824 binding bytes and zero cache. This also reproduced with the stronger
single-plan and shared-table owners. These are synthetic control-flow numbers,
not GPU measurements or qualified resource profiles.

## Decision

Keep the CUTLASS offer as explicit unavailable evidence. The call-local region
returns no qualified CUTLASS profile, including under test hooks. Its constructor
also rejects a forged or stale CUTLASS admission before context setup, descriptor
factory execution or module loading. Existing cuBLAS, generated, cuTENSOR and
cuBLASLt eligibility and arithmetic remain unchanged. Standalone/shared-table
CUTLASS retains its explicit enclosing-context ownership contract.

No suitable longer-lived owner is passed through this response API: its stream
and device are borrowed, and its provider context only owns BLAS resources.
A global ledger or an assertion that destroying a stream resets CUDA would not
establish the required ownership, so neither is introduced.

## Validation and reopening

The host admission test covers provider-enabled/disabled and test-hook/production
builds, repeated rejected construction, no provider setup calls, exact generated
budget boundaries and incumbent/other-provider selection. Real-GPU response tests
now require CUTLASS qualification controls to retain the legal incumbent.

Reopening requires an explicit CUDA-context/build-lifetime owner that survives
failed construction and subsequent calls, preserves retained charges and hard
quarantine, and provides a verified recovery contract. Resource profiles and
endpoint timings alone are insufficient.

This supersedes the region eligibility claim in
[the initial source-response qualification note](2026-10-05-cutlass-source-response-region.md).
Its negative timings remain historical evidence; they do not qualify this
missing lifetime boundary.

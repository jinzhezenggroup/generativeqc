# Decision: D4 CUDA storage shares the prepared allocation authority

Status: implemented
Date: 2026-10-29

## Problem

D4's native plan calculated a bounded worker schedule and retained-byte inventory,
but its actual CUDA allocations used raw `cudaMalloc`/`cudaFree`. A surrounding
prepared device ledger therefore neither admitted nor observed these bytes. Adding
a frontend resource estimate first would leave the central enforcement gap intact.

## Decision

Use the existing runtime resource allocator and matching release for every D4
retained buffer, including EEQ workspaces and scientific tables. Keep the stream
and partial buffers in a custom-deleter owner throughout construction, so a host
exception while reporting a failure cannot bypass cleanup. The allocator retains
its shared ledger with each actual buffer; release works after scope exit and on
a different thread. No D4 equation, launch schedule, coordinate reuse or table
identity changes.

## Boundaries and rejected alternatives

Do not add a second ledger or estimate-only reservation around the raw allocator.
Do not mark D4 global ResourcePlan admission supported merely because physical
storage is now observed. A future planning slice must reuse the native resource
sizing authority and account for per-system KS owners and composite corrections.
Standalone unbudgeted D4 still uses the existing maximum-bytes worker selection;
without a ledger, the shared allocator preserves ordinary CUDA behavior.

## Invariants

An exact native device-byte diagnostic admits one retained owner. A smaller
ledger refuses before exceeding its cap and retires all partial allocations.
Successful replay retains ownership. Closing observation metadata before the plan
is legal; final plan destruction retires the remaining ledger. Scientific outputs
still pass independent D4 energy/gradient/charge and finite-difference gates.

## Evidence

`tests/native/test_d4_production.cpp` exercises first/partial/final allocation
shortage, exact capacity, independent energy/gradient output under that cap,
post-allocation host failure, recovery, cross-thread release and detached metadata
lifetime. Checks execute in Release builds. Existing independent fixtures,
finite-difference tests and changed-coordinate failure recovery remain required.
Real-GPU execution and endpoint measurements are recorded separately with their
source/binary identities; this note does not substitute for those receipts.

## Revisit when

D4 worker storage or allocation lifetime changes, or a complete global correction
inventory is admitted. Reuse the runtime authority rather than recreating budget
policy in the correction frontend.

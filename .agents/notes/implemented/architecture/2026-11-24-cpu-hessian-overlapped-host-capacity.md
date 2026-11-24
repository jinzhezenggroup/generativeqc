# Decision: Select CPU Hessian headroom in native resource composition

Status: implemented
Date: 2026-11-24

## Problem

Python and the CLI composed the native derivative floor with their respective
primal inventories, then selected the optional workspace independently. Python
read back the portable plan's peak, while the CLI read the inventory JSON. Both
needed the native selection and effective budget that were already available
before materializing frontend diagnostics.

## Decision

Keep capacity selection in the C++ resource planner. The derivative workspace
is pageable and spans the full mandatory composition, so increasing it cannot
affect pinned/device usage, resident bytes, or which provider was selected at
the floor. Apply slack only after the mandatory composition succeeds and use the
effective limit after headroom and reserve. Preserve an independent numerical
ceiling interpreted by the actual SDK derivative owner at execution. The Python
bridge returns the resulting cap in the same transaction as the mandatory
composition; the CLI uses the same policy directly. Keep marshalling costs with
the frontend that creates them, so absolute host peaks need not match across
frontends.

## Invariants and evidence

This is inventory admission, not shared `HostResources` ownership, a process
RSS bound, or a promise that an optional schedule fits other work caps. A
prepared HVP uses its already-admitted fleet provider choices and must leave
that fleet and its accepted final state unchanged on refusal. Native wire tests
cover exact and one-byte-short effective limits, unbounded budgets, independent
ceilings, overflow and publication on failure. Prepared PBE tests cover exact
admission and reuse after rejection. The CLI also checks PBE-D3(BJ) parity
with the SDK at the floor and with optional host headroom.

## Revisit when

A CPU derivative can prove a smaller overlap or participate in shared host
lease accounting. Shorten the lifetime or promote execution admission only with
provider-specific allocation and rollback receipts, not by lowering the
declared peak on the strength of a successful capacity query.

## References

- #2227 R2; `src/runtime/resource_planning.cpp`; `docs/user/hessians.md`.

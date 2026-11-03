# Decision: account retained ECP source allocation once

Status: implemented
Date: 2026-11-03

## Problem

Scalar-ECP post-HF admission added ECP payloads to the common source capacity,
while CUDA RHF source handoff already reserved the same vector separately.
The executable compact-source contract exposed this composition error in PR
#2957's core-a job 125983532074: removing 17 ECP terms changed the retained
reservation by twice their allocation rather than once.

## Decision

The common source-capacity owner reserves the complete ECP vector capacity,
including spare allocation. Pending, resident, and compact RHF source handoff
consume that common bound without a second ECP reservation.

## Invariants

Exact-budget acceptance, one-byte-short refusal before allocation, copied ECP
identity, completion fencing, allocation-failure fallback, transport errors,
and resource cleanup remain executable requirements. The regression also
reserves extra ECP capacity, so counting only vector length is insufficient.
No scientific acceptance gate or actual allocation is removed.

## Evidence

The host-compiled compact-source probe uses CUDA API fakes and performs no
real-device execution. Source changes require refreshing live catalog input
identities, not historical qualification measurements. This correction does
not requalify an integrated CUDA scientific endpoint.

## Host lifetime fixture follow-up (2026-11-04)

Core-c job 126046989772 exposed an outdated stand-in: the lifetime-only harness
returned a constant common source bound, so removing the adapter's duplicate
reservation also removed ECP accounting from that fake environment. The fixture
now supplies the retained ECP allocation through the common-owner contract. Its
existing ECP-copy, budget, terminal-failure and deferred-fence assertions remain
unchanged. Actual exact/short/spare-capacity arithmetic is independently checked
by the compact-source probe and native contracts; no production bound changes.

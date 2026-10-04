# Decision: transfer and fence resident RHF interaction ownership

Status: implemented
Date: 2026-10-04
Agent: dot

## Problem

The resident interaction handoff in #1870 exposed a raw CUDA bucket plan to a
throwing solve before installing its owner, and submitted borrowed ERI reads
before allocating and recording their completion event. Its retained query
also omitted persistent host topology and warm buffers. The reusable executable
introduced by #1869 cannot independently own or reserve that same storage.

## Decision

The uncached handoff uses the exception-safe cached adapter with a scoped local
slot. After a converged, identity-checked reference, source publication moves
the plan out of the cached slot. The source reports the established complete
`hf_cuda_retained_numeric_bytes` query plus its additional normalized-system
allowance and ECP payload. Allocation or capacity failure leaves the completed
physical reference usable through the bounded source fallback.

The reusable slot is empty while the resident source owns the executable, so
its external reservation is zero. Provider planning charges the source once;
subsequent CC/triples/force phases keep its reservation in the retained CC
problem. Memory-limited phases may retire it once and restore only that source
reservation before retrying. This composes with #1869's existing executable
retirement. The older force-planner source double charge is outside this change.

Only a complete successful CC or CC(T) energy/force endpoint may return the
plan to the empty reusable slot. Reclaim requires the sole shared source owner
and completed consumer uses; raw source views are cleared first. Warm-state
capture and result preparation precede reclaim. Exceptional endpoints destroy
the source instead, and result moves avoid a new allocating copy after reclaim.
MP2's existing executable owner is unchanged.

Per-stream event creation and vector insertion precede submission. Each launch
marks its use pending and invalidates any older event generation before the
copy. Every post-submission failure drains that stream, falling back to a device
fence if needed. Successful destruction/reclaim waits on the recorded events.
If the runtime cannot establish completion by any fence, reclamation fails and
destruction retains the backing allocation rather than releasing a live borrow.
This is deliberate fail-closed retention for an unrecoverable CUDA runtime error.

## Evidence

Host tests extract the production adapter, source, phase callbacks, and complete
accounting helper. They exercise throwing plan allocation, event creation,
vector insertion, first-record/re-record, post-launch failure, deferred multiple
streams, exclusive transfer/reclaim/replay, numeric overflow, ECP copy capacity,
and memory-only source retirement. Existing real-owner accounting tests cover
cleared capacities, warm storage, and exact/one-byte-short budgets. Publication
fixtures now populate the resident-source slot and check that failure never
reclaims or publishes a completed triples diagnostic.

The actual adapter translation unit also compiles with real project owner/API
declarations and a host CUDA declaration shim. These checks establish host
ownership and accounting only. No GPU numerical execution or timing is claimed.

## References

- #1870 / #1857: resident physical-reference interaction handoff
- #1869: reusable correlated CUDA executable
- [Retained capacity and retirement](2026-10-04-correlated-plan-capacity-fallback.md)
- `tests/python/test_rhf_reference_source_lifetime.py`
- `tests/python/test_posthf_prepared_source_lifetime.py`
- `tests/python/test_correlated_reference_plan_capacity.py`
- `tests/python/test_correlated_reference_retirement.py`
- `tests/python/test_rccsdt_publication_transaction.py`

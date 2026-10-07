# Decision: bind producer work to existing schedule and trace owners

Status: implemented
Date: 2026-10-07

## Problem

The streamed DF occupied exchange schedule can remain memory-bounded while
regenerating earlier source rows. A source-site warning cannot quantify that
amplification or establish whether a completed endpoint executed it.

## Decision

The receipt consumer has a strict, generic scientific/problem/resource/producer
identity and exact work counts. Its first static adapter uses the production
compiler schedule and enumerates its visits independently. Its diagnostic
adapter reads the existing CUDA component trace and native streamed-row
counters. The trace remains incomplete execution evidence because those
counters occur before final endpoint success can be established.

## Rejected alternatives

A second method-specific work ledger would duplicate existing ownership.
Treating a trace submission or graph capture as completed execution would
overstate evidence. Treating amplification above one as a bug would ignore
legitimate bounded-resource recomputation without reusable-dependency proof.

## Invariants

- Compare only identical scientific, resource, producer, dependency, reuse, and
  invalidation domains, with each source receipt bound to its own source bytes.
- Keep memory budget and peak separate from semantic work.
- Reject partial, invalid, duplicate, overflowing, or stale evidence.
- A static PASS cannot become a runtime or numerical PASS.

## Evidence

`tests/python/test_producer_work_audit.py` covers schedule tail census,
source-bound work-growth detection, exact ratios, ownership mismatch,
legitimate unproven recomputation, and incomplete native-counter adaptation.
The existing `test_df_projected_exchange_schedule.py` separately exercises
the emitted native callback traversal and two-slot lifetime when a host C++
compiler is available.

## Revisit when

An endpoint-owned completion receipt and independently verified source/build
identity can be joined to the existing trace without inferring execution from
submission counters.

## References

Issue #1628; `docs/maintainer/producer_work_receipts.md`;
`python/generativeqc_compiler/method/df_exchange_schedule.py`;
`src/runtime/cuda_component_trace.cpp`.

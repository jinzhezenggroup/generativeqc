# Decision: consume allocation evidence without adding an allocator

Status: implemented
Date: 2026-10-07

## Problem

The #1668 source scanner reports lexical sites, not runtime allocation events.
The existing direct-HF prepared ledger counts successful owned CUDA allocations
and live/peak capacity, but does not export total requested bytes or a complete
host/device journal. A zero count from that ledger cannot establish a broader
zero-allocation replay guarantee.

## Decision

Add an independent std-lib consumer with a separately pinned identity, phase
schedule and ownership-domain contract. Reconcile observed journals against
count, cumulative requested bytes, simultaneous peak ownership and final live
ownership. Require complete observation and event/live continuity across phase
boundaries. Retain the existing ledger's known measurements, explicitly report
missing fields and return INCOMPLETE for its current v1 export.

## Rejected alternatives

Do not estimate total requested bytes from live/peak differences: transient
allocate/free pairs can leave identical final ownership. Do not infer runtime
zeros from scanner output, capacity plans, or CPU iteration-boundary samples.
Do not add a parallel allocator or modify active solver/stationary production
owners merely to make this consumer emit a PASS.

## Invariants

The consumer cannot authenticate claimed runtime coverage or provenance. A
reviewed source-matched capture mechanism and independently pinned contract are
still required. PASS applies only to named observed ownership domains. Setup,
rebuild, publication and hot windows cannot be interchanged. Missing fields,
counter/byte confusion and lost events must fail closed.

## Evidence and consequences

Adversarial std-lib tests cover identity mismatch, unobserved phases, lost events,
resize uncertainty, transient allocation, zero-byte allocation, ownership
continuity and existing ledger incompleteness. No native/CUDA execution is
claimed. The full #1630 scientific/runtime acceptance remains open.

## Revisit when

An existing production measurement owner exports complete host/device journals,
requested-byte totals, synchronized phase boundaries and installed-build
provenance. Add its capture adapter and real endpoint qualification then.

## References

- #1630; advisory prerequisite #1668.
- `docs/maintainer/replay_allocation_receipts.md`.
- `python/generativeqc/resources_native.py` and `src/runtime/resource_ledger.hpp`.

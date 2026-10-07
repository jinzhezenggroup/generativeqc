# Decision: require complete event coverage for CUDA residency claims

Status: implemented
Date: 2026-10-07

## Problem

Existing CUDA DF traces and KS owner diagnostics count useful movement and
synchronization, but their scopes differ. A zero in one owner is not proof that
an endpoint has no transfers or waits. The trace also inserts diagnostic CUDA
events and waits, which must not be attributed to production.

## Decision

Use a separate receipt consumer with a source/build/device/endpoint contract,
explicit owner and role regions, complete event coverage, and conservative hot
ratchets. A host round trip requires an ordered link between the same payload
and dependency domain. The existing historical DF component trace is adapted
only as partial owner evidence and always yields `INCOMPLETE`.

## Rejected Alternatives

Byte-total H2D plus D2H pairing can join unrelated final publication and
setup uploads. File-path allowlists hide newly hot calls in previously legal
owners. Treating profiler CUDA event pairs or final trace waits as production
synchronization would charge measurement overhead to the endpoint.

## Invariants

Capture, incomplete submission, missing coverage, invalid identity or malformed
event order cannot issue a zero-transfer PASS. Legal preparation, publication,
oracle and compatibility roles remain counted and visible. A new replay wait is
reported separately from final publication.

## Evidence

`tests/python/test_residency_receipt_audit.py` checks independent arithmetic,
roles, payload links, tampering, truncation, capture, overflow and the SHA-matched
historical DF trace. Its retained identity and limitations are in
`benchmarks/results/residency-receipts-1629/README.md`.

## Consequences

The slice provides a strict consumer and an owner diagnostic adapter. A complete
runtime transfer/sync event producer and current-device qualification remain
future #1629 work. No production kernel, scientific API or compiler changed.

## Revisit When

An owner can provide complete source-matched transfer/wait events and payload
links for a declared hot region, including graph replay rather than construction.

## References

Issue #1629; merged advisory scanner #1668; `docs/maintainer/residency_receipts.md`.

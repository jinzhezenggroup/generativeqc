# Decision: retain AOT module charges outside provisional contraction plans

Status: implemented
Date: 2026-10-05

## Problem

The shared table releases provisional optional plans when any member fails.
CUTLASS modules belong to CUDA's context, so rolling back those descriptors does
not roll back device code. Summing resources only over published plans would
lose both successful partial loads and loads interrupted by an exception.

## Decision

The table accepts the same typed affine requests and an explicit AOT artifact
digest. It prepares CUTLASS plans outside replay, visits resolved provenance and
counts canonical summands. Add a separate retained-cache field to provider
reservations and a persistent charge in the table. Transfer each new plan's
module charge immediately, on both success and exception, before any descriptor
cleanup. Keep that charge after table release and across subsequent attempts.

Charges are conservative per plan, including retries; there is no module-sharing
deduplication claim. The enclosing resource owner must carry the charge across
fallback and table destruction if it continues using the same CUDA context.
Region selection remains unavailable until that caller contract is integrated.

## Rejected alternatives

- Zeroing resources after descriptor rollback undercounts CUDA-owned modules.
- Keeping only the largest plan charge assumes unqualified module sharing.
- Treating loader exceptions as soft fallback can hide an exceeded reservation.
- Promoting from isolated matrix tests omits complete endpoint cost and lifetime.

## Evidence

The native probe exercises shared-table execution for all 160 independent affine
layout/precision/beta cases, two variants, semantic work and AOT provenance. It
also rejects the second plan after a successful first load and injects a hard
failure after loading, checking persistent charges after cleanup and retry.
Existing non-CUTLASS native probes protect optional-off compilation.

## References

- #1886, #1888, #1943
- [Native owner](2026-10-05-native-cutlass-aot-contraction.md)

# Decision: retire mandatory source-bound materialization dispositions

Status: implemented
Date: 2026-11-03

## Problem

The source-bound materialization ratchet required manually maintained
producer/consumer bindings to the normalized identity of entire source files.
PR #2938 added unrelated position-integral functionality in
`src/integrals/s_integrals.cpp` and its header, while the materialization
candidate census remained nine at both the base and candidate revisions.
Nevertheless, three MP2/UMP2 source bindings became stale and the advisory
workflow failed. The gate tested manifest currency, not a demonstrated
materialization regression or numerical correctness.

This conflicted with the repository's policy against permanent required-CI
checks pinning source or function-body fingerprints for semantic-preserving
changes. Maintaining those dispositions imposed review churn unrelated to the
scientific producer/consumer under investigation.

## Decision

Remove the mandatory `ratchet_native_materialization.py` workflow step, its
entire-file fingerprint disposition manifest, the dedicated ratchet script,
and its implementation-specific tests. Remove the obsolete audit artifact and
update current maintainer documentation.

Retain the source-work census, structured materialization diagnostics, the
blocking rank-two matrix-chain regression check, and the DF producer-work
ratchet. Materialization findings remain prompts to assess ownership, source
work, peak storage, parity and complete-endpoint timing; the absence of a
source binding does not certify performance or safety.

## Rejected alternatives

- Rehashing MP2/UMP2 records after each unrelated modification only masks
  the structural brittleness.
- Disabling the entire advisory work audit would lose useful independent
  complexity and producer-work checks.
- Treating the static candidate count as a new blocking threshold would not
  show semantic or execution regressions.

## Invariants

Scientific correctness, independent-reference tests, resource/lifetime
admission, compiler-owned complexity protections and source-matched endpoint
qualification remain required where applicable. Do not reintroduce a
mandatory file-fingerprint update to accept unrelated refactors.

## References

- PR #2938, advisory work audit run 41570561638, job 125702201110.
- [Original decision](2026-10-09-materialization-candidate-review.md).
- [Current performance engineering guide](../../../../docs/maintainer/performance_engineering.md).

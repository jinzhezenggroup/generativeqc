# Decision: Bind CodSpeed baseline receipts to benchmark selection

Status: implemented
Date: 2026-10-08

The subset-selection policy below is superseded by
[identical PR/master selection](2026-10-10-codspeed-identical-pr-selection.md).
The version 2 receipt fields and source/environment checks remain in use.

## Problem

PR #2123 initially published only default PR-tier energy cases on master, while
performance-sensitive PRs also selected WB97M-V. A matching CPU/runtime receipt
therefore did not prove that every requested endpoint had an exact-base sample.

## Decision

Master push runs include WB97M-V alongside the default PR tier. Version 2 receipts
record the benchmark file SHA-256, tier, and normalized extra-case selection.
Qualification requires identical benchmark source, PR tier, and a requested extra
set contained in the published extra set. Old or incomplete receipts fail closed.

## Invariants

Keep scientific correctness and 5% regression thresholds unchanged. Do not treat
an uploaded GitHub receipt as proof of CodSpeed backend indexing or selection.
The existing CPU/runtime comparison remains mandatory.

## Consequences

Editing the benchmark definition intentionally makes comparisons advisory until
an identical definition has been published on master. Master runs pay the bounded
WB97M-V sentinel cost. Future selector semantics must preserve this subset relation
or replace the receipt contract; full-tier receipts are intentionally unsupported.

## Evidence

Regression tests cover missing optional coverage, differing source, old/malformed
selection metadata, successful subset comparison, and advisory lookup failures.

## References

- https://github.com/jinzhezenggroup/generativeqc/pull/2123

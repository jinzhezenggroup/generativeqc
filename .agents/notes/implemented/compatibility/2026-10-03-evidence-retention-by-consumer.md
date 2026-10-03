# Decision: retain evidence by its consumer

Status: implemented
Date: 2026-10-03

## Problem

The results tree approached its 64 MiB limit while historical archive-format
tests kept old campaign ZIPs alive, and necessary publication samples repeated
large JSON structures. Small campaign-specific compression patches obscured the
underlying distinction between current numerical inputs and archival reports.

## Decision

Retire six archive-mechanics-only ZIPs and two historical JSON reports through
one verified snapshot at published master
`1a4acc519eb881cc19d418de65ecca72324359d4`. Preserve their README conclusions,
failures and limitations, original bytes in Git, and existing no-fetch recovery.
Use small synthetic archives and temporary Git repositories for storage tests.

Keep the active independent density inputs and all numerical/timing records of
live publications. Use the existing shared JSON/gzip representation, resolve
members through publication inventories, and discover tracked publications for
compaction checks. Do not introduce campaign-specific codecs or weaken gates.

## Storage dependency correction

The previous compactor rebound only the evidence envelope. A samples record can
itself contain `record_parts`, so compressing its children left dangling plain
paths even though the outer publication's hashes validated. Resolve the complete
JSON reference dependency graph from leaves to parents before committing any
changes. Preserve exact leaf bytes and scientific values; reject cycles without
writes. Regression tests cover plain and already-packed parents, corruption,
rollback, repeated checks and size-threshold crossings.

## Invariants and limits

- Preserve exact independent fixtures, paired samples, failures, thresholds,
  input/source identities and current scientific mutation tests
- No history rewrite, external archive upload, new artifact service or cap raise
- Historical recovery is optional for normal tests; shallow/source-only checkouts
  use synthetic recovery tests and report missing history without implicit fetch
- Current-tree savings do not shrink the existing historical Git object database
- Unmerged/local raw objects are not automatically durable history anchors

The current retention guide is corrected from its stale 96 MiB wording to the
actual 64 MiB policy. Archive exceptions are retired only for the removed payloads;
all other ownership rules and the 1 MiB member/2 MiB incoming limits remain intact.

## References

- [Retention policy](../../../../docs/maintainer/evidence_retention.md)
- [Historical report snapshot](../../../../benchmarks/results/retention-reports-20261003/snapshot.manifest.json)
- `tests/python/test_evidence_archives.py`
- `tests/python/test_compact_evidence_paths.py`
- `tests/python/test_publication_record_reader.py`

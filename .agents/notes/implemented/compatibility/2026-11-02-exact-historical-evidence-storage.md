# Decision: losslessly compact historical evidence before retaining new runs

Status: implemented
Date: 2026-11-02

## Problem

The benchmark checkout was near its fixed 80 MiB aggregate limit. New default
qualification needs raw populations and scheduler proofs. Removing slow/failed
measurements or increasing the limit would not preserve the evidence contract.

## Decision

The H100 Direct WB97M-V byte-object container now stores valid UTF-8 objects as
text and other bytes as base64. The same 840 original object SHA-256 values,
inventories and unavailable-binary boundaries are retained. Text representation
lets the outer XZ compressor recognize repeated JSON/source content; the reader
continues to enforce bounded expansion and exact object hashes.

The 2026-10-20 README baseline uses the existing exact-FP64 v3 byte-plane codec.
Its 256 records and 765 samples reconstruct the complete original compact JSON
container byte for byte. The report loader authenticates both streams, individual
record bytes and the original ordered container SHA. Publication attachment
metadata changes only the stored-byte checksum using the existing metadata
patcher; scientific values and acceptance decisions retain their original bytes.
The existing historical-review exception follows its lossless representation;
individual, aggregate and change-review size limits are unchanged. The existing
README change-review exception moves with its numerical FP64 stream. A separate
SHA-pinned exception covers the exact H100 storage-only re-encoding, whose
840 objects and all 11 reader tests are unchanged; new GPU evidence remains
charged to the review budget.

The fitted-range-cold 2026-10-24 provenance stream uses stronger XZ compression
(587736 to 505596 bytes). Its decoded bytes retain SHA-256
`18ea0197a88fdbb431da3e474bea366067b851771790e09774fdaef10c1ce932`;
both historical reproducers produce identical output before and after.

The new tracked automatic-tile capsule retains the whole final-source campaign,
including slower, failed and cancelled owners. Entire superseded source campaigns
remain in an authenticated ignored investigation capsule and original task
directories; the tracked capsule binds its streams by size and SHA-256. This
source/campaign boundary does not select observations by timing or gate outcome.

## Evidence and consequences

One-off comparisons reconstructed every original object/container and compared
all historical report outputs. Ordinary and optimized Python replays agree.
The H100 host-history test accepts an explicitly supplied restored snapshot
through `GQC_H100_HOST_HISTORY_ROOT`; it never fetches or skips a missing or
corrupt fixture. Restoring the existing Git snapshot and running all 11 tests
passes, including the original failed HOST producer assertion. This separates
the optional historical fixture location from current scientific evidence.

The original compressed transports remain in ignored task artifacts and Git
history. No release, tag, asset, external backup, failed sample or slower
population is created or removed by this storage-only change.

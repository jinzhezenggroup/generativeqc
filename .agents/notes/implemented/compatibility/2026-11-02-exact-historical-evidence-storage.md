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

## Follow-up: bounded text-context recompression, 2026-11-04

The native precision candidate cutover's conflict-free evidence projection against
master `e9dd9a26320052204a0911a67213855fbf75a278` exceeded the unchanged
80-MiB limit by 33,347 bytes. Six existing XZ transports now use LZMA2 preset
`9 | PRESET_EXTREME`, `lc=4`, `lp=0`, `pb=0`. Their **complete decoded byte
streams**, not merely parsed numerical values, remain identical. The change
saves 36,156 bytes; no measurement, failure, slow sample, source identity,
floating-point spelling or scientific decision is regenerated or removed.

| Historical family | Previous bytes | Stored bytes |
| --- | ---: | ---: |
| capability-fp64-20261101 | 348,612 | 344,724 |
| df-raw-coulomb-20261023 | 485,212 | 478,368 |
| df96-local-autotune-20261103 | 134,752 | 132,976 |
| fast-range-warm-20261021 | 103,744 | 98,016 |
| occupied-xc-cold-20261023 | 346,752 | 335,532 |
| xc-prepared-parameters-20261028 | 284,648 | 277,948 |

The existing family review rows, raw-J bundle checksum, capability checksum file
and DF96 stored-byte description follow their new transport identities. No
retention threshold, cap, exception, reader, numerical gate or producer changes.
The compaction remains charged to the ordinary changed-evidence review budget.
Independent before/after report replays cover the four families with standalone
reproducers; the other two retain their whole decoded source/receipt containers.

The one-off admission check streamed all 1,999 tracked XZ/Gzip benchmark
archives (1,230,514,952 decoded bytes) looking for incoming pins to the selected
compressed-byte hashes. None target these six transports. This is an explicit
storage-migration check, not a new permanent CI source-fingerprint requirement.
All original transports, proposed streams and exact-byte receipts remain in
ignored local investigation artifacts and existing Git history.

Rejected alternatives matter: the initial independent-force-tiles recompression
grew its transport; another recipe saved only 192 bytes and was not applied.
The CUDA KS seed transport offers larger savings but its old compressed hash
is embedded in the XC-unroll receipt, so it is left untouched. Likewise, the
Fast-source-floor integration receipt pins its qualification transport.
Updating those scientific containers just to rescue storage would expand the
ownership and validation boundary unnecessarily. The historical README DFT
summaries also retain their explicitly promised original compressed bytes.

These are historical storage changes, not new GPU runs or requalification of
current precision sources. The small remaining space is not permission to skip
the aggregate or changed-evidence check when upstream adds more evidence.

## Follow-up: retain the complete CPU RCCSD closeout, 2026-11-04

Integrating the subsequent master evidence still exceeds the unchanged aggregate
cap after those six recompressions. The CPU RCCSD closeout now stores its original
22,605-byte JSON as a 2,256-byte XZ stream using preset `9 | PRESET_EXTREME`.
Its complete decoded bytes match the original Git blob recorded in the bundle
README. The existing shared JSON reader handles transport decoding; ordinary
and optimized replay preserve all 24 observations and reject corrupt/truncated
streams. This does not promote the slower measured CPU-native implementation,
change its explicit opt-in, or qualify the newly integrated source. The occupied-XC
negative evidence and every original numerical token remain untouched. The
one-off incoming-pin audit and rejected insufficient recompression recipes remain
in ignored storage-investigation artifacts; no permanent implementation hash test
or retention exemption is added.

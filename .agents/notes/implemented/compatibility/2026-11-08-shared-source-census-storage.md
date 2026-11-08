# Decision: share complete source censuses without changing evidence

Status: implemented
Date: 2026-11-08
Agent: /root/audit_3044
Model: GPT-6 Astra Pro

## Problem

Individually admissible PR evidence exceeded the unchanged aggregate retention
budget when combined. Several historical source inventories were repeatedly
stored inside active numerical replay inputs. Moving those inputs to Git history
would have introduced a new offline-replay dependency.

## Decision

Fifteen containers already present in published master
`6746722694f7e59e291d8dd792e60acb58e998f7` share a complete source path table,
base hashes and explicit changes/removals. Original map insertion order remains
part of the representation. Every other original JSON character is retained in
UTF-8 fragments, including embedded JSON strings and unusual number spellings.
The original paths hold small authenticated descriptors. Complete October 15 and
October 23–24 campaigns are stored together without splitting original members
or pooling their statistical populations.

The shared reader checks physical and expanded bytes, source-map counts/order
and original JSON digests. Compressed-identity consumers reconstruct the actual
original XZ stream with its recorded preset and check its complete digest.
Byte-keyed caches follow authenticated input bytes, so file mutation cannot be
hidden by a prior cached path. All source and numerical checks in consumers
remain in place.

## Rejected alternatives

The H100 DB36 profile, DF, screening and build archives remain untouched: their
measurements, compiler commands and receipts participate in active acceptance
replays. Whole-archive retirement was therefore rejected. A separate digest
dictionary added complexity for little saving; a shared path-name table removes
repeated removal/order strings while retaining the existing base/delta model.

## Evidence and consequences

All fifteen original JSON and original XZ streams were compared byte for byte
against published master before and after installation. Eight existing CPU
reproducers have separate complete-output comparisons. Synthetic decoder tests
exercise reordered/missing maps, duplicate changes and metadata keys, corrupted
checksums, bounded expansion, trailing bytes and corruption after cached reads.
Numerical evidence, failed observations, source reconstruction, storage limits
and independent-reference gates are unchanged. No new GPU result is claimed.

The transport proof used the migration Python runtime. Encoder output can differ
across versions or builds despite equal settings, so original-XZ recovery checks actual bytes and
fails closed on a mismatch. It never substitutes a declared digest. Original
JSON recovery is independent of the encoder and retains its exact-byte check.

An independent bounded encoder audit reproduced these fifteen originals with
actual liblzma 5.6.3, 5.8.1 and 5.8.4, using the xz 5.8.1 frontend, one thread,
CRC64 and the recorded presets: all 45 comparisons matched. This matrix does
not remove the encoder-version limitation or the fail-closed transport check.

The fifteen new decoder cases live in the existing
`test_compact_evidence_metadata.py` module. The tracked Python test-module
inventory remains unchanged, preserving the existing alphabetical CI shard
assignments without changing routing, slots or workers.

Current recovery instructions and the exact member inventory are in
`benchmarks/results/shared-source-censuses-20261108/`.

# Decision: preserve embedded benchmark text with exact FP64 storage

Status: implemented
Date: 2026-11-01

## Problem

The benchmark checkout was within 36,977 bytes of its unchanged 80 MiB limit.
New CUDA qualification must retain failed and slower populations alongside wins.
Some historical containers store entire JSON receipts as strings, preventing the
existing exact-FP64 codec from compressing their numerical arrays.

## Decision and invariants

Version 3 of `benchmarks._exact_fp64_evidence` adds explicit compact/spaced JSON
formats and references to previously authenticated text entries. Every original
FP64 bit is retained. The reader restores each original receipt's formatting,
checks its original SHA-256 digest, then reconstructs and authenticates the outer
historical container. References can only address earlier entries in the same
bundle; they cannot recurse or read external files. Both compressed streams must
have unique checksum entries. Historical v1/v2 data remains readable unchanged.

The portable DF response campaign is migrated without changing its scientific
report, numerical gates, original producer identities, oracle/reuse files, or
101 original receipts (including incomplete controls). Its original uncompressed
container SHA-256 is
`38034e964b033aa02de380e8e98005cf77af7c569a0d4831363c7c133fecd28e`.
The prior XZ container SHA-256 is
`dcd5af827ddd0b46a130224b6d575d6193112fdbafc03d90fca3460b87286330`;
that exact historical compressed object remains in parent Git history. The new
streams total 296,736 bytes versus 553,912 bytes previously. No claim is made that
recompressing the restored text reproduces the former XZ bitstream.

The self-contained SCF validation-GEMM campaign uses the same representation for
its 96 original endpoint strings. Its uncompressed historical container digest is
`7b9cdca33dc01a17ef92aaacfb6a8570b49a13384b6e540a8636d32557acdd45`,
and its former XZ digest is
`8ff3395641e4db3704676b36a5579547c5274e2d7f9dcb0c9f93148ce0492263`.
The streams shrink from 500,048 to 277,104 bytes. Its independent endpoint,
changed-geometry, work and population checks remain unchanged. The separately
owned provenance and reference packets are not rewritten.

The guarded-XC campaign likewise retains all 108 original records and its
complete outer container. The original uncompressed digest is
`08ada7c3e9365fbf38ecc4133962ce552f7c310b2424ddbeffe5433cd27bb35a`,
and the historical compressed digest is
`97f18093e34c5559e7e47a16e83a371a581f421d2c44431dd0fd796f3dbb421f`.
Its two streams use 297,692 bytes instead of 534,772. Independent replay still
qualifies all 352 complete calls while retaining slower timing controls.

The self-contained DF charge-partition endpoints retain all 83 original strings
and 242 complete calls. Their original uncompressed container digest is
`577089d58fe3e5a54f6abe5579effb4c287b29e48c05599b8dfb4e41f26606be`,
and the historical XZ digest is
`49175505473966880804d22ff7b8c1b5d882dd50e2db104fa7a534790df79715`.
The streams use 232,308 bytes instead of 392,256. Its separately owned provenance
and downstream source projections remain unchanged. The spaced-JSON format
preserves this container's original default Python separators without rewriting
its checksum or weakening independent population validation.

## Evidence

A one-time migration check compared the decoded entire container byte-for-byte
with the parent XZ's decompressed bytes. All independent gates in the campaign's
`report.py --verify` replay successfully. Host tests cover signed zero, subnormal
and normal FP64 values, historical formats, compact newline distinctions,
authenticated embedded text, altered inner text, invalid references, and missing
stream digests. Existing independent outer-manifest tamper tests remain active.

## Rejected alternatives

Increasing the checkout limit, deleting negative populations, rounding values,
or rewriting the identity of a reference dependency would weaken retention or
scientific provenance. Migrating this self-contained campaign avoids touching
other reports' historical reference bindings. The storage reader performs no
scientific qualification; consumers must still run their independent gates.

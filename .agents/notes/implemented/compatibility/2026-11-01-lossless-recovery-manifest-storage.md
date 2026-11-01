# Decision: authenticated XZ storage for historical recovery manifests

Status: implemented
Date: 2026-11-01

The benchmark checkout cap leaves insufficient room for new complete endpoint
receipts. Two large historical recovery inventories now use adjacent XZ payloads
with small JSON storage records at their original paths. Every original manifest
byte is retained; both compressed and decoded SHA-256 and decoded size are checked
before ordinary recovery validation. Payloads must be adjacent and decoding is
bounded. Existing recovery links, Git revisions and scientific payloads are
unchanged, and no external host, Release or rewritten history is introduced.

This frees about 245 kB without raising the checkout budget or dropping samples.
The original inventories still describe 955 and 758 files, respectively. Actual
offline recovery, historical profile aggregates, and corrupt/path/size controls
pass in `test_restore_retained_evidence.py`, `test_historical_profile_retention.py`
and `test_retention_20260925_objects.py` (50 cases). Storage is not permission to
change or remove the underlying evidence.

## Fleet evidence follow-up

The retention-granularity snapshot (464 files) now uses the same authenticated
storage descriptor. Decoded inventories
and ordinary recovery records compare exactly before and after compression,
saving another 57,792 bytes without changing recovery paths or history.

The fleet receipt container also supports a sequential-marker v2 encoding:
FP64 words remain exact XOR deltas, while JSON float markers omit redundant cursor
integers. Traversal order, total word count and every original file's SHA-256
still authenticate reconstruction. Separating predictors by method and engine
improves compression without altering observations. The prior v1 reader remains
compatible. The full fleet report restores original bytes before numerical gates,
and retains slower candidates and incomplete allocations under the 80 MiB cap.

The same exact FP64 representation is now shared by the bounded-projection,
XC-unroll and fleet reports through `benchmarks/_exact_fp64_evidence.py`. The
603-member XC-unroll archive saves 415,708 bytes, and replacing redundant
float-cursor integers in bounded projection saves another 103,372 bytes. Every
original member checksum and the original numerical readers still pass; this
storage conversion does not change or retire any measurement. Family review
shards bind the new compressed payload sizes and hashes. The 80 MiB checkout cap
is unchanged.

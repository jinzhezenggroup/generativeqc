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

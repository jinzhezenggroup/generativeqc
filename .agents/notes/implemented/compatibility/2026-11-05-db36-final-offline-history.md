# Decision: retain the historical DB36 final archive in published Git

Status: implemented
Date: 2026-11-05

The original H100 DB36 result remains tied to `db3657f08d983791c45b7d949cb9028fe0055e3f`.
Its two raw archive parts (1,225,848 bytes) are already published in merged commit
`40f126a74fcf527012f2f4b2a53b158a87334967`. Both Git blobs were read and compared
byte-for-byte with the live files before retirement. The adjacent historical
snapshot records the full revision, original paths, sizes, SHA-256 and Git blob
identities. The cohort declares this storage mapping and its family review no
longer requires nonexistent live parts.

The existing shared offline recovery API supplies only the two declared missing
parts, using canonical repository paths even when a publication is staged elsewhere.
Present corrupt files, symlinks, directories, and other missing members fail.
No implicit fetch is added. Small synthetic transport tests cover that behavior;
the opt-in public historical reader still recomputes the original scientific proof.
No old scientific computation or replay is needed for this byte-preserving move.

This makes space for current-source cold measurements under unchanged storage
limits. The original full E/F values, failures, 21 warm wins, nine nonwins, audit,
source labels and proof hashes are preserved. It does not qualify current main,
change any gate, retire other-owner evidence, or publish a release/external backup.

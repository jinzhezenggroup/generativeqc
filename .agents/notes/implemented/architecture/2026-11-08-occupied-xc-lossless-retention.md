# Decision: retain occupied-XC capsules through exact offline reconstruction

Status: implemented
Date: 2026-11-08
Agent: /root/audit_3041
Model: GPT-6 Astra Pro

## Problem

PR #3013's source integration passed the affected host/evidence checks, but its
700,720 bytes of endpoint/integration XZ transports contributed to the unchanged
aggregate evidence budget being exceeded. These live capsules contain accepted
and negative measurements, full independent/paired vectors, discarded campaigns,
diagnostics and execution provenance. Removing observations or moving ordinary
replay behind a history fetch would lose required evidence availability.

## Decision

Represent the two complete original JSON texts using the existing exact-FP64 v3
decoder, with two receipt rows and all 418,921 floating-point values retained.
The compressed streams and manifest occupy 425,265 bytes, saving 275,455 payload
bytes before small reader/receipt/documentation overhead. The source commit
`e71606ddecf9d7a8e527ff79572427bb2acbb309` and original Git blob identities remain
additional provenance; an unmerged PR head is not the sole archive anchor.

Both original XZ transports reproduce byte-for-byte from the decoded compact JSON
plus LF using the original XZ/CRC64/preset-6 writer recipe. The local reader checks
every original text and compressed-stream SHA-256, size and Git blob identity.
An explicit export command can restore both files with no Git/network access.
Integration preserves its original endpoint compressed-byte binding by checking
the actual reconstructed baseline, not by substituting a new storage digest.
The small compression cache keys on the complete original text, so a changed
baseline cannot reuse an earlier transport identity.

## Rejected alternatives

Simple XZ recompression saved little. Splitting by measurement record,
observation or vector increased per-row checksum metadata or disrupted useful
compression contexts; tested bundles ranged from 431,553 to 501,637 bytes.
Whole-text v3 rows remain simpler and smaller. A bounded LZMA context comparison
reduced the FP64 stream by another 420 bytes without changing decoded bytes.
No new shared codec, size-budget exception or archive publication is introduced.

## Invariants and evidence

The migration compared every original decoded byte and every reconstructed XZ
byte, then walked all fields with exact Python types, dictionary/list order and
binary64 bit comparisons. Original identities and payload counts are retained in
the scoped storage receipt. Normal and optimized replay preserve the scientific
gates and corruption probes; storage tests cover offline original recovery,
transport identity corruption, changed baseline contents and rehashed FP64
corruption against the original text digest.

The PBE0/3 confirmation remains classified as a warm regression; the original
r2SCAN-3c cost gates and all scientific tolerances remain unchanged. No new GPU
qualification or timing evidence is claimed.

## Consequences and revisit

This saving does not alone close the 434,367-byte excess seen on PR #3013's
published integration head; further integration also shares the aggregate
budget with other PRs. Report the staged aggregate check honestly and handle
additional storage decisions separately. Revisit only for a demonstrated simpler
lossless representation or if an XZ implementation can no longer reproduce the
authenticated original transport; do not drop the original-byte binding.

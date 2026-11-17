# Decision: compact historical transport without changing scientific evidence

Status: implemented
Date: 2026-11-15

## Problem

Complete R1 endpoint observations include actual failed and incomplete attempts.
Their retention must fit the existing aggregate, change-review and per-file
budgets. Raising budgets, dropping failures or rounding full-precision values
would weaken the evidence rather than solve its storage cost.

## Decision

Use the existing authenticated `generativeqc.compressed-manifest.v1` XZ
transport for ten bulky historical JSON documents. Preserve their original
paths as explicit descriptors, retaining decoded byte count and SHA-256 and the
compressed payload SHA-256. Expose an exact-byte reader from the restoration
tool so current H100 delivery wrappers authenticate decoded original attachments
before handing them to unchanged frozen receivers.

Use deterministic gzip for ten post418 raw JSON documents because that frozen
receiver already supports the suffix fallback. Keep every decoded byte and
original scientific identity; do not rewrite a receiver or measured attachment.

Keep R1 numerical transport separate: each collection member is an ordinary
existing exact-FP64 v3 bundle. A bounded collection authenticates each stream
manifest and rejects unsafe names, linked members and duplicate logical
receipts. Splitting transport members does not split, omit or resample a
qualification population. No observation recovery depends on Git or a network.

## Evidence and invariants

The historical index records 959,798 original bytes and 170,246 descriptor plus
payload bytes, before the new index/documentation overhead. A direct offline
test authenticates every original size and SHA-256. Storage/restore/post418 and
H100 cohort tests check executable recovery and unchanged scientific outcomes.

Keep existing size/digest validation, finite decompression limits and linked-file
refusals. A descriptor is not numerical admission. Preserve original failed
audits even when a maintained auditor corrects a historical work-history join.
Do not reinterpret a historical campaign as current source qualification.

## Revisit

Change transports only with another exact-byte recovery audit and executable
consumer tests. If a frozen receiver cannot consume an existing transport,
retain the original rather than silently changing its scientific contract.

## Provenance-only retirement

Complete negative R1 populations still exceed the hard aggregate budget after
JSON compaction. Four further files contain only build metadata or source
checksum streams, not scientific observations: the DB36 H100 production-build
packet and three historical Becke censuses. Their 507,403 original bytes are
authenticated against already published ancestor
`17f9317a2495c142941d7390d465fcf79887d643` before retirement. Exact size, SHA-256,
Git blob and mode remain in explicit snapshots. No Release, external upload or
implicit network access is introduced.

The H100 wrapper recovers only missing build metadata and still passes the exact
original archive to its unchanged frozen receiver. Present invalid metadata and
missing numerical packets must not fall back to history. Becke reproduction
restores its original checksum stream explicitly before the unchanged GPU
recipe; the pre-GPU identity rejection remains tested. Every independent input,
numerical observation, failure and scientific gate remains in the checkout.

Case-fragment transport was measured but rejected: extra per-fragment hashes
cost more than the small FP64-stream improvement. Do not introduce a new
numerical codec, round observations or discard failed populations to fit a
storage budget. Instrumented diagnosis is non-acceptance data: retain all scalar
timing rows and pauses compactly, rather than duplicate their full E/F vectors
or retain raw profiler/syscall logs in Git.

Final sizing additionally keeps the shared automatic-force qualification
manifest and one cold-force audit report in the same local exact-byte transport.
The older automatic-native manifest stays raw because the unchanged frozen
AOT verifier authenticates it as an original-byte reference. Rewriting or
weakening that receiver is rejected. Recovery of the 126,184-byte derivative-AD
measured code patch was also verified, including byte-identical numerical/work
replay, but retiring it was rejected: the generic shared publication contract
requires a local dirty-source reconstruction patch. Keep its original publication,
patch and replay unchanged rather than weakening that contract or adding a special
missing-member exception.

Instead, two existing CUDA-ownership summaries use ordinary XZ and the established
publication reader, retaining every original decoded byte. The 24 historical
documents total 1,192,712 original bytes and 217,636 stored descriptor/payload
bytes, excluding index overhead. Standard XZ settings with lc/lp/pb zero save
35,436 bytes across the five R1 FP64 streams without changing a decoded byte,
codec, scientific value, observation count or acceptance gate. Standard delta
filters were measured and rejected because they increased storage.

The exact-SHA change-review exceptions cover both rejected ABBA and BAAB FP64
streams, preserving failures in both orders instead of selecting passing attempts.
Accepted streams remain charged. This existing review mechanism does not waive
the unchanged aggregate or per-file cap, or any scientific/performance gate.

Shared source maps embedded in active numerical
containers remain local: moving those maps would introduce a history dependency
into reconstruction of the observations themselves, so that alternative is
rejected.

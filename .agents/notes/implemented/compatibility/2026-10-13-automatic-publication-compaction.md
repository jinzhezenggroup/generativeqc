# Decision: automatically compact declared publication data before commit

Status: implemented
Date: 2026-10-13

## Problem

A check-only compactor detects large retained JSON but makes every author perform
another manual migration. A suffix-wide compressor is unsafe: custom benchmark
readers, reference manifests and hash-pinned provenance can depend on exact names.

## Decision

Use the existing shared publication schema and evidence/samples roles as the
bounded opt-in storage contract. A first local pre-commit hook considers only
publications containing touched paths, validates complete staged dependencies,
and writes deterministic gzip plus declared storage bindings. It exits for review
and restaging. Keep CI's separate read-only check and all scientific validation.
Do not migrate any historical dataset in the automation change.

Prepare every selected publication and legacy review update before touching the
working tree. Keep original recovery copies until verified output and metadata
replacement have completed; roll back reported failures and block on leftover
recovery directories after a hard interruption. Do not promise filesystem-wide
multi-file atomicity. Never modify the Git index.

A token-span editor changes only storage metadata leaves. Reserializing an entire
scientific JSON object would round high precision decimal tokens, underflow tiny
numbers, normalize signed zero or alter string escapes, despite lossless gzip.
The manifest's scientific source identity is preserved in the same way.

Interpret active bindings only in evidence/samples roles. Historical reproduction
envelopes deliberately preserve pre-storage attachment names and must stay opaque;
parsing every JSON object as an active graph would corrupt those frozen originals.
Bound compressed input, expanded output, XZ decoder memory, aggregate batch size,
JSON depth/tokens and member counts before writes. Reject concatenated/trailing
streams. Register rollback intent before destructive calls, including interrupts
raised immediately after an OS operation has already succeeded.

Pre-commit may restore tracked inputs after an unstaged-edit conflict while leaving
an untracked generated gzip. Reuse only an exact deterministic byte match without
overwriting it; preserve differing companions for inspection. Stop the hook chain
immediately after edits so later formatters never receive retired filenames.

## Rejected alternatives

- Compressing every file in results would rewrite README/scripts/source and break
  consumers lacking the shared role/path/hash reader
- Automatically staging or committing would hide the resulting data/metadata
  changes and could capture unrelated author edits
- Updating manifests first without validated recovery bytes could leave metadata
  pointing at truncated payloads after a partial write
- Regenerating references or timing data is outside a storage-only operation

## Evidence

`test_automatic_evidence_compaction.py` exercises actual Git-index behavior.
`test_compact_evidence_metadata.py` checks lexical scientific preservation,
malformed/truncated/mixed metadata and review/staging write failures. The existing
path and transaction tests retain symlink/collision, rollback and rerun coverage.

## Revisit when

Another retained-data producer adopts an equally explicit shared reader and
storage identity contract. Add its adapter and failure tests before making it
eligible; never infer opt-in from a filename, folder or file size alone.

# Decision: publish generated source as exact UTF-8 bytes

Status: implemented
Date: 2026-10-23

## Problem

The semantic source cache measures the caller's `source.encode("utf-8")`, but
the canonical source publisher used default text encoding and newline handling.
On Windows, two LF-terminated ASCII comments containing 19 UTF-8 bytes were
published as 21 bytes with two CRLFs. The first call returned the original
string, while the manifest recorded size 19 and the hash of the actual 21 bytes.
The next lookup correctly rejected this internally inconsistent entry. Default
locale encoding could also reject or change non-ASCII source. Existing-entry
text reads normalized newlines and could accept different raw source bytes.

## Decision

Keep `common.source_cache.cache_source` as the single publication owner. Encode
the caller's string explicitly as UTF-8, compare existing raw bytes, and write
the encoded data through a binary stream. Preserve the existing temporary-file,
flush, fsync, replacement and cleanup sequence. No concurrency algorithm changes
are included.

The byte oracle is the original caller string encoded as UTF-8. Explicit CRLF,
mixed LF/CRLF/CR, Unicode, and presence or absence of a trailing newline are
preserved without normalization. The semantic loader's bounded snapshots,
manifest sizes, SHA-256 validation, recipe schema and key calculation stay intact.
Source recipes, emission, mathematics, backend/target/precision policy and
consumer defaults are unchanged. Existing Linux UTF-8/LF output remains identical.

## Rejected alternatives

- Normalizing on load or relaxing length/hash checks would conceal changed
  source identity and permit unchecked bytes.
- Clearing or silently repairing old Windows entries would hide the failure.
  Existing malformed entries remain visible and rejected; this is no migration.
- A second semantic-cache publisher would duplicate ownership and leave other
  canonical publisher consumers vulnerable.

## Evidence

At upstream baseline `cb33ac12388ecde389f513f8edcc8332af82a87a`, a separate
Windows Python 3.13.5 reproduction observed 19 expected bytes, 21 published bytes,
and a second-call integrity failure caused by source hash mismatch.
Tests added before the production fix plus the existing semantic suite yielded
16 failures and 22 passes. Failures exposed raw newline expansion, encoding
failure, newline-corruption acceptance and invalid concurrent-winner manifests.
After the fix:

- Source, semantic and stationary semantic suites: 49 passed, including actual
  PBE0 source lowering followed by a fresh-process hit with emission/IR/AD
  forbidden. Binary compilation/linking in this existing test are stubs.
- Related bulk-cache, cache-boundary, compiler-cache and CC provenance suites:
  45 passed.
- Compiler structure: 578 modules, zero dependency errors. Changed Python files
  pass Ruff lint and format checks.

New tests compare published raw bytes and manifest size/hash to caller UTF-8,
use exact byte budgets and fresh-process producer-forbidden replay, preserve
cache-disabled identity, and retain malformed Windows entries unchanged on
rejection. Existing partial/corrupt/oversized/nondeterministic-winner checks remain.
A real I/O-boundary wrapper applies Windows newline and cp1252 defaults on any
host, providing reproducible Linux-CI coverage without mocking stored bytes.
Publication failure also verifies temporary-file cleanup.

Local RED/GREEN logs and private caches are under ignored
`.artifacts/1123-byte-publication/` in the implementing worktree; no shared cache
or historical evidence was cleared. These are host Python source-publication
results, not native binary, GPU, numerical endpoint or scientific qualification.

## Consequences and remaining gates

Existing entries with bytes different from the caller now fail exact comparison;
text newline equivalence is deliberately insufficient. Valid source reuse keeps
the existing owner, recipe and binary-cache boundaries. Revisit only with an
explicit source-format contract change and corresponding byte/integrity tests.

Refs #1123. Its catalog, installed packaging/runtime, CUDA numerical/resource and
other original acceptance gates remain open. This fix does not supersede or
reinterpret the merged offline closure slices or their historical evidence.

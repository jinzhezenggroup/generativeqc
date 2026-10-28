# Decision: preserve exact historical bytes while restoring merge headroom

Status: implemented
Date: 2026-10-28

## Problem

Several individually valid PRs could not share the aggregate benchmark checkout
budget. The original merge order worked only until another PR added evidence;
repeatedly reordering the queue did not resolve the storage constraint.

## Decision

Compress existing reconstruction patches with XZ and selected historical
practical-auxiliary records with the gzip format already understood by its
reader. Preserve original bytes and hashes in a dedicated storage inventory;
update only publication storage bindings and recovery instructions. Keep the
two residency-audit trace fixtures at their existing ordinary paths.

## Invariants

Do not change the 83,886,080-byte aggregate limit, source identities, clean
samples, failed results, numerical gates or acceptance decisions. Reconstruction
patches must be decoded to their original paths before application. The original
source revision remains available in existing Git history, with no new release,
tag or external archive.

## Rejected alternatives

Raising the limit merely postpones recurrence. Removing samples or failure
records weakens evidence. Repinning scientific subjects to the current checkout
would falsely requalify old measurements. Compression changes only storage and
does not require a GPU rerun of historical experiments.

## Evidence

The inventory and opt-in verifier in
`benchmarks/results/pr-merge-storage-20261028/` authenticate both stored and
decoded bytes. Thirty-four files save 1,394,547 payload bytes before metadata.
Existing publication integrity and independent practical-auxiliary replay
remain the semantic checks.

# Decision: recover exact H100 Nsight parts from merged Git history

Status: implemented
Date: 2026-11-02

## Problem

Integrating the fitted-LR branch with current master exceeded the unchanged
80 MiB aggregate evidence budget by 625,875 bytes. The H100 supporting packet
includes two exact Nsight archive parts totaling 1,440,220 bytes. Truncating
SQLite observations or dropping unfavorable results would destroy evidence.

## Decision

Retain the complete original parts in the already-merged #2940 commit, an ancestor
of current master, and authenticate recovery using the existing offline Git
snapshot mechanism. Keep the original cohort inventory, part hashes, reconstructed
archive checksum and independent SQLite reduction. Missing local Git objects
fail with an explicit fetch instruction; the reader does not access the network.
Present but changed or linked parts cannot silently fall back to history.

Only the two now-absent storage exceptions are removed. All limits remain fixed;
all scientific gates, original Standard84 failures and later-source restrictions
remain unchanged. Historical audit metadata remains the original audit record.

## Validation

Replay all five supporting phases, verify byte-for-byte recovery, reject changed
parts/digests, missing historical objects and symlinks, and run existing retention
and evidence-change checks. This is storage/integration validation, not new GPU
qualification or timing evidence.

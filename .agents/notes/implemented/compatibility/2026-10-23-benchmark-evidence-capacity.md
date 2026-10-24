# Decision: bound aggregate benchmark evidence at 80 MiB

Status: implemented in this change
Date: 2026-10-23

## Reason

The 64 MiB aggregate limit blocked complete source-bound scientific receipts.
At upstream `ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`, tracked results occupied
67,014,464 bytes, leaving 94,400 under the 67,108,864-byte limit. The complete
#2681 build and water raw bundles alone occupied 202,024 bytes. Their original
capacity failures and all raw data remain with that scientific slice; this
storage-only change does not reinterpret them as successful delivery.

The user directly authorized increasing capacity ("允许扩大容量"). The bounded
64-to-80 MiB increase is the implementation choice, not a claim that the user
specified that exact number. This avoids modifying unrelated historical assets
just to admit the pending #2679/#2681 evidence.

## Decision and limits

Change only `benchmark_results_max_bytes` from 67,108,864 to 83,886,080 bytes.
Retain the 1 MiB per-file cap, 2 MiB change-review budget, checksum-pinned
exceptions, publication integrity checks, numerical gates and protected
CI/review/merge requirements. The cap remains finite; this is not permission for
unbounded growth or a change to any scientific acceptance rule.

The policy validator and existing budget tests are unchanged. The proposed
value is checked against actual tracked evidence; the unchanged unit cases
still exercise aggregate overflow and invalid budget values. No production,
reference data, historical receipt, cache or resource lifecycle is changed.

## Scope

Refs #2679 and #2681. Merging this dependency admits storage under the new
aggregate cap. It does not complete either scientific issue, qualify a device,
or turn compile/run/CI success into independent scientific acceptance.

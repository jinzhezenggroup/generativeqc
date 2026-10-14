# Decision: recover a missing native CLI compiler cache from trusted master

Status: implemented
Date: 2026-10-14

## Problem and measured boundary

The shared native CLI package steps restored only `actions/cache`, although the
master producer also publishes a complete CUDA compiler-cache artifact. An
expired or evicted primary cache could leave CLI qualification cold while the
trusted artifact was still available.

This is a resilience gap, not an established cause of the long #2452 queue run.
Run 38483373351 / job 115591063777 restored key ending
`344919efe9b33737df0f92fe993891ed2a8a838601aadb0575ae7cbadd650514`
at 2026-10-13 23:20 UTC. That cache came from successful master run 38463722427,
source `36574095876e4d12a3945b39cef72f6c77895f00`, saved at 20:52:45 UTC;
its CLI build had 549/549 hits. The queue measured 153 hits and 397 misses
among 550 cacheable calls, with no uncacheable calls. The newer queue source
`eeb988b15dccd7bf341e5f293b343b0bcf071ac5` changes `direct_jk_kernels.cu`
and headers consumed by `direct_bounded_fallback.cu`. Their approximately
1269/1765-second compiles include legitimate changed-input misses; a transport
change cannot eliminate those misses without violating correctness.

The latest retained successful-master snapshot available when that queue began
was older: run 38459005549, source `a80d04b00782e4dcfd2ac6ed7efc3c9f9e81c8cd`,
artifact 11847430547 created at 20:25:52 UTC. Subsequent snapshots from runs
38479511514 and 38483078685 were created at 2026-10-14 00:02:16 and 00:19:59 UTC,
after the queue's cache restore. They cannot establish that a newer snapshot
would have accelerated that queue.

## Decision

Keep `actions/cache` first and retain every successful exact or prefix hit. Only
a successful restore with no `cache-matched-key` can attempt the optional
artifact fallback. The helper refuses an existing destination before any API
request and again before activation. Eligible consumers are non-skipped PR,
merge-group, and master-push CLI builds. Tags and manual runs are unchanged.

Discovery inspects at most 20 completed successful push/master runs of this
repository's exact `ci.yml` workflow, and at most 10 artifacts per candidate.
Require matching repository and head-repository names/IDs, source SHA, artifact
run ID, nonexpired artifact, expected name, positive size up to 6 GiB, and a
SHA-256 digest. Download by immutable artifact ID with explicit repository/run
and mandatory digest validation. Discovery has a 25-second process deadline;
download is capped at two minutes. Every added stage is non-gating.

Only a successful digest-checked download can activate. An atomic same-filesystem
rename moves this attempt's isolated staging directory to the still-absent
canonical `~/.cache/ccache`. Staging is an attempt-owned sibling under `~/.cache`,
so a separate runner-temp mount cannot force cross-device transport. No copy
or partial activation is allowed. A missing,
failed, slow, cross-device, or untrusted snapshot retains normal compilation.
Cleanup can remove only the unactivated attempt staging directory.

## Invariants and rejected alternatives

- Never prefer an artifact unconditionally: it may be older than a prefix hit
- Never consume a PR, merge-group, fork, failed, or other-workflow producer
- Preserve compiler/source/flag keys, required compile tiers, full catalog,
  link, package, smoke, and existing master fast-delta handoff checks
- Preserve original primary-key outputs and canonical master save/publication
  paths. Recovery must not suppress a requested snapshot refresh
- No new permissions, source identity normalization, cache sloppiness, compiler
  campaign, CI cancellation, or Release operation

## Evidence and consequences

Behavioral tests exercise missing/malformed provenance, wrong producers, event
scope, exact/prefix/missing primary caches, failed downloads, path ownership,
atomic activation/cross-device failure, cleanup, and master save/publication
conditions. The live API receipt for successful master run 38483078685 and
artifact 11860653235 passes the same provenance predicates. No multi-GB artifact
was downloaded locally and no native compilation was started for this repair.

The fallback can save compilation after eviction while adding bounded transfer
cost on a true cache miss. It does not establish a before/after CI improvement;
measure naturally triggered jobs, preserving their source and flag identities.

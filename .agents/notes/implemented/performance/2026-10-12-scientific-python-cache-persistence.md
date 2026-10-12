# Decision: persist every Python shard's strict test compiler cache

Status: implemented
Date: 2026-10-12

## Problem and observed evidence

All eight Python shards already direct test compilations into
`$RUNNER_TEMP/ccache-python-probes` with `CCACHE_MAXSIZE=256M`, separately from
the native-library cache. The restore filter admitted only core-a/b/c,
runtime-heavy and compiler-heavy. Consequently posthf, uccsd-forces and
ecp-forces could neither restore nor save their test objects.

The CPU JIT compile-to-object repair in #2372 makes this omission consequential.
Its merge-group [run 38281919172](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38281919172)
reported the following cold post-test ccache observations:

| Shard / job | Tests passed / skipped | Object hits / misses | Uncacheable calls | Cache limit used |
| --- | --- | --- | --- | --- |
| posthf / 114923445731 | 361 / 57 | 0 / 485 | 457 | 4.16% |
| uccsd-forces / 114923445666 | 20 / 7 | 0 / 171 | 143 | 1.10% |
| ecp-forces / 114923445675 | 134 / 59 | 0 / 100 | 101 | 0.44% |

These are evidence of discarded cacheable work, not measurements of a future
warm run or an overall CI speedup. Links still execute and are not object hits.
The queue itself remains read-only even after this repair.

## Decision and invariants

- Restore the existing test-only overlay for every Python matrix member. Keep
  its per-shard key, source inputs and prefix fallback unchanged. Retain strict
  ccache source/compiler/flag acceptance, without sloppiness or path spoofing.
- Preserve the separate shared native-library cache and its master producer.
  No new master-push Python jobs, reference seeding, permissions, test selection,
  scientific acceptance gates or compiler implementation changes are included.
- Merge groups may restore but never inspect for publication or save. PR
  snapshots remain scoped to their PR merge ref; they do not become shared
  master or cross-PR caches. Existing scheduled/manual runs can write in their
  own ref scope, including the default branch when that is the selected ref.
- Keep exact completed snapshots immutable. Successful tests may save the
  complete key; failed/cancelled tests retain the unique run/attempt partial
  suffix. Missing/failed restores, missing primary keys, skipped tests and exact
  cache hits do not publish another snapshot.
- Before saving, inspect the pinned ccache 4.14 machine-readable
  `files_in_cache` counter. Zero entries do not publish stats-only directories.
  Missing/malformed/non-integer/negative counters or a failed ccache process
  prevent saving. Explicit `shell: bash` enables pipeline failure propagation;
  the save also checks the inspection's original outcome, not its softened
  conclusion. Inspection is best effort with a one-minute timeout; the existing
  save remains best effort with a two-minute timeout. Neither masks failed tests.

## Size and expected reuse boundaries

Keep the existing 256 MiB configured per-shard limit; the observed scientific
caches do not justify increasing it. Three additional shard namespaces can
produce additional immutable snapshots on eligible runs, including partial
attempts. The local limit is not a repository-wide retention budget. Empty
snapshots are suppressed, but GitHub storage/eviction still applies.

Do not promote PR caches into shared native caches or run the full Python matrix
on every master push just to seed them. Ordinary same-PR reruns and already
scheduled/default-ref qualification can establish actual reuse. Strict source
paths, compiler identities and flags may still produce legitimate misses;
persisting a directory does not guarantee hits or a timing improvement.

## Validation and revisit conditions

Executable workflow tests cover all eight matrix members, required events,
push exclusion, orphaned queues, restore failures, exact hits, test failures and
cancellation, immutable partial keys, and empty/malformed/error statistics.
The inventory regression mirrors the declared Actions shell, including the
case where ccache emits valid positive statistics but exits unsuccessfully.
The exact pinned ccache 4.14 binary also verified empty and existing nonempty
cache inspection without compilation or cache clearing.

Use natural CI observations to revisit the storage/reuse tradeoff. No additional
performance campaign or hosted warm-cache claim accompanies this change.

## References

- [GitHub cache scope, immutability and eviction](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
- [GitHub runner shell semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsshell)
- [ccache 4.14 machine-readable statistics and size management](https://ccache.dev/manual/4.14.html)
- [CPU JIT object-cache repair #2372](https://github.com/jinzhezenggroup/generativeqc/pull/2372)

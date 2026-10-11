# Decision: shared Linux compiler snapshots with recoverable master seeds

Status: implemented in this change; CI rollout metrics pending
Date: 2026-10-11

## Decision and trust boundary

The Linux CPU snapshot contains sequential GCC Release, GCC coverage, Clang
Release and Python-native builds, bounded by a 2 GiB local ccache limit. The
CUDA 12.9/sm_120 snapshot contains fast CI and full native CLI builds, bounded
by 3 GiB. Snapshot keys describe transport generations; ccache still validates
compiler identity, command flags, source and included headers independently.
Workflow/dependency recipe inputs participate in the snapshot key so a recipe
change can publish a new immutable generation.

Only master-push producers save the shared compiler caches. PRs and merge
groups consume those snapshots without publishing them. Complete primary keys
are saved only after successful seed stages. A started
master seed that fails can save its successful compiler outputs under a distinct
per-run partial key. Prefix restoration can recover that work, but a partial
snapshot cannot claim the immutable complete key. Both kinds of upload are
non-gating; partial recovery has a finite two-minute ceiling and cannot guarantee
execution after a hard runner/job termination.

Only the complete CUDA producer publishes the separate artifact mirror. Queue
artifact discovery and cleanup continue to require this CI workflow, master,
push, and a completed successful run. In-progress/failed runs and other refs or
workflows cannot replace either retained trusted snapshot. No permission scope
or Release behavior changes here.

## Coverage build-directory identity

GCC coverage embeds its output directory. The seed therefore uses `build`, just
like the CPU PR consumer; the other compiler variants keep separate directories.
Moving the coverage seed to `build-seed-gcc-coverage` gave a cache miss when its
otherwise identical compile was replayed in `build`. Do not repair this by
ignoring paths, flags, headers or compiler identity.

## Validation and limitations

A bounded independent review probe used ccache 4.14.1 and GCC with the workflow's
`-fprofile-arcs -ftest-coverage -fprofile-update=atomic -O3 -DNDEBUG` flags.
The first directory missed, the different consumer directory also missed, and a
repeat in that consumer directory hit. This is identity evidence, not a project
build timing result.

The focused workflow tests execute the CPU seed loop with fake compilers,
exercise master/PR/queue/tag/manual failure-publication boundaries, and execute
artifact discovery/retention against a fake API, including failed discovery and
untrusted artifacts. No real artifacts are deleted by these tests.

The independent review initially removed the 256 MiB per-shard Python probe
overlay, then corrected that decision before rollout. Master CMake builds do
not compile the test-generated NativeCxx wrappers, and the Python test job
excludes all push events. A master-push-only post-test save would never run.
Existing restored probe snapshots recorded 346/408 and 347/407 cache hits in
the test phase (84.80% and 85.26%). These are aggregate hit counts, including
possible within-job reuse, not measured time savings. Removing persistence
would lose a demonstrated source of reuse even though not every call must miss.

Retain the existing five eligible shards' 256 MiB probe overlays under
`ccache-python-probes-v2-<shard>-...`, using the same separate temporary-directory
path for restoration, probe compilation, statistics and save. Non-queue refs may
publish this overlay; GitHub's ref scoping confines PR entries to that PR.
Queues remain restore-only. Completed tests may fill the immutable complete
probe key; failures/cancellation use a distinct per-run partial key. Exact cache
hits skip publication, and all overlay saves are non-gating with a two-minute
ceiling. Neither shared master producer restores or promotes the overlay.
Scheduled/manual runs on master can seed master-ref overlays, but are not a
replacement for reusing unmerged PR-specific probe sources.

The existing NativeCxx executable test verifies reuse across distinct pytest
session/worker roots, as well as changed-header, source and flag invalidation.
No cache identity is relaxed to manufacture hits.

Combined variant sizes may evict objects within the shared local ccache limit. The limits bound each snapshot, not the repository's aggregate
size across immutable generations or partial recovery keys. Monitor real
restore/upload time, per-variant hits, seed duration and repository cache usage
before claiming a net speedup or changing budgets. Manylinux and CuMetal caches
retain their separate toolchain boundaries.

## References

- [Prior probe cache job, 346/408 hits](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38114774936/job/114400380661)
- [Prior probe cache job, 347/407 hits](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38113859405/job/114398555883)
- `tests/python/test_native_cxx_cache.py`

PR #2310; `.github/workflows/ci.yml`; `tests/python/test_ci_cache_lifecycle.py`;
`test_ci_cuda_cache_cleanup.py`; `test_ci_cuda_cache_refresh.py`;
`test_ci_native_probe_cache.py`; `test_ci_merge_queue_cache_policy.py`.

Agent: dot

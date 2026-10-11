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

Only master-push producers save these compiler caches. PRs, merge groups and
Python probes consume the restored local directory without publishing it.
Complete primary keys are saved only after successful seed stages. A started
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

Consolidation does not imply that every former cache entry is seeded. In
particular, master does not run Python's native probe test suite, so probe-only
objects newly compiled on PRs are local to that job and are not retained for
later runs. Combined variant sizes may also evict objects within the local
ccache limit. The limits bound each snapshot, not the repository's aggregate
size across immutable generations or partial recovery keys. Monitor real
restore/upload time, per-variant hits, seed duration and repository cache usage
before claiming a net speedup or changing budgets. Manylinux and CuMetal caches
retain their separate toolchain boundaries.

## References

PR #2310; `.github/workflows/ci.yml`; `tests/python/test_ci_cache_lifecycle.py`;
`test_ci_cuda_cache_cleanup.py`; `test_ci_cuda_cache_refresh.py`;
`test_ci_native_probe_cache.py`; `test_ci_merge_queue_cache_policy.py`.

Agent: dot

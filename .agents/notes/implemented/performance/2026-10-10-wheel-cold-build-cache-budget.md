# Decision: bound cold wheel qualification and retain compiler progress

Status: implemented
Date: 2026-10-10

## Problem and evidence

Wheel run 38003071721, job 114065497728, started at 23:11:03 UTC on
2026-10-09 and ended cancelled at 00:11:18 UTC. The job had a 60-minute
timeout. Both the v3/v2 compiler-cache restore and the cibuildwheel tool-cache
restore missed. Both success-only cache post steps were skipped afterward.

Ninja began its 637-task build at 23:12:53 and reached task 275 at 00:10:34.
No compiler error was recorded. Large generated AOT/direct CUDA units explain
the slow interval: tasks 156 through 161 completed between 23:37:13 and
23:57:49. Ordinary compilation continued afterward; this was not a stopped
build. The failing TRSM declaration from #2181 was not reached in this run.

Extrapolating the observed 57.69 minutes for 275 tasks linearly gives about
133.6 minutes for 637 tasks. This is a sizing estimate, not a completion-time
prediction: code generation, individual CUDA units, device linking and wheel
repair have different costs. It nevertheless shows why another unchanged
60-minute cold run is not a credible cache-seeding strategy.

## Decision

- Bound the complete build/repair/installed-wheel-test step at 150 minutes and
  the job at 180 minutes. A separately bounded 10-minute compiler-cache save
  has room to run after a build-step failure or timeout.
- Use the existing pinned cache action's restore/save entry points for ccache.
  Save with an explicit `always()` condition when restore supplied a key.
  This preserves completed compiler results even when a later compile fails.
- Suffix immutable compiler-cache keys with run ID and attempt. Restore the
  same source-input prefix first, then the existing v3/v2 prefixes. An incomplete
  snapshot can therefore be replaced by a more complete later snapshot without
  trying to overwrite an immutable cache key.
- Retain the 3 GiB compiler-cache bound and normal GitHub Actions cache eviction.
  Do not add cache artifacts, change repository permissions, change ref scopes,
  or move fork/PR cache data across trust boundaries.
- Leave the cibuildwheel tool cache success-only. Partially installed tool
  environments are not treated like completed compiler-cache entries.

## Preserved qualification and rejected alternatives

The build still fails normally; cache persistence does not use
`continue-on-error`. Wheel repair, installed-wheel smoke testing, payload/linkage
checks and the aggregate remain unchanged. Running qualifications are not
cancelled, and workflow concurrency/triggers are unchanged.

Raising compiler parallelism or disabling AOT/qualification work was rejected.
The current build has no explicit one-job Ninja pool; the expensive CUDA units
already run through the normal scheduler. There is no memory/runtime evidence
supporting a different compiler parallelism or NVCC split-compilation policy.

## Validation and follow-up

Policy tests cover timeout ordering, failure-path compiler-cache persistence,
run/attempt snapshot refresh, retained restore prefixes, success-only tool-cache
behavior, and preserved qualification/concurrency. Existing merge-queue policy
tests also run. Full manylinux completion and actual snapshot reuse must still
be established by an ordinary subsequent wheel run; local policy tests do not
claim that result. Revisit the 150/180-minute bounds using complete cold/warm
timings once available.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38003071721/job/114065497728
- https://github.com/actions/cache/blob/55cc8345863c7cc4c66a329aec7e433d2d1c52a9/action.yml
- https://github.com/actions/cache/blob/55cc8345863c7cc4c66a329aec7e433d2d1c52a9/save/README.md
- https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching

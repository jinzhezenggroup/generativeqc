# Decision: reserve bounded time for legitimate cold CUDA qualification

Status: implemented
Date: 2026-10-13

## Problem and observed evidence

The required fast CUDA job's 25-minute job-wide deadline included installation,
cache transport, configuration, production compilation and mandatory probes.
Restoring a cache is not proof that the current combined source has cached objects.

- [Queue job 115135774405](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38348640736/job/115135774405)
  checked out `8dbd30e05940109a2c6270fbcb6dea42de3f7a54`, including the stacked
  #2370/#2393 source. It successfully downloaded trusted master snapshot
  `11788372070` from run `38342838078`: 2,986,967,792 bytes, verified transport
  digest `4d4a816a5cdad0b94ca2c148311577727a27161f3b4d18a8a373200beb9ae8e2`.
  Compilation began at 04:55:25 UTC and was cancelled at 05:18:46, after about
  23m21s, with 614/619 Ninja steps complete. Ccache reported 511 calls, 137 hits
  and 374 misses. No completion or successful remaining probes can be inferred.
- [PR job 115127587193](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38344333312/job/115127587193)
  checked out merge `1d0d8406166f234b22801e49721db417bc8f9ea9`, containing
  #2393 head `f4122a11551001768bb4955974bad57802bbf55e`. Its production build
  completed in 8m34s; the full job took 11m17s. Ccache reported 514 calls,
  467 hits, 46 misses and one uncacheable call.
- [Master job 115137902786](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38349249133/job/115137902786)
  checked out `c4e092c457ababd8d0a985e12a2168ac6d1fe368`. Production compilation
  completed in 19m24s and the job in 21m32s, with 514 calls, 259 hits,
  254 misses and one uncacheable call. The existing resource/packaging probes
  and both host preflight CTests then passed.

The two successful jobs used the same restored Actions cache transport key,
`ccache-linux-cuda-v4-12.9-sm120-78cb0449118cd87d7b3d6c5bebe876c9d26d3fed461582c4a6af0996f41591e7`,
but their compiler hit counts differ substantially. These are different source
builds, not a paired transport experiment. The queue artifact download succeeded;
there is no evidence that changing cache trust or replaying an identical failed
queue build would repair source invalidation. The logs do not attribute every
miss to a particular changed input, and unfinished Ninja counts do not predict
remaining wall time.

## Decision

Set only the fast CUDA job ceiling to 35 minutes and its mandatory production
build step to 30 minutes. The build gets 6m39s more than the observed censored
compile, while five minutes remain outside that step for setup, existing probes,
reporting and runner cleanup. Successful observed non-build overhead was 2m43s
and 2m08s, leaving more than two minutes of reserve. The separate resource and
direct-AOT jobs retain their existing 25-minute budgets.

Keep all targets, compiler flags, parallelism, AO radial/static-resource and
wheel-link probes, Fock/proposal CTests, exact-event checkout, aggregate gates,
trusted-master discovery, cache keys and publication rules unchanged. The
consumer never publishes a partial or branch-scoped cache. A timeout/failure
still fails the required gate; there is no automatic retry.

Always run the existing compiler-cache reporter with detailed counters and its
allowlisted configured compiler/build identities, bounded to one minute. On
failure/cancellation, retain that text and Ninja's per-target completion timings
for 14 days, with a two-minute, non-gating upload. Do not upload the complete
CMake cache, build tree, process environment, secrets or memory dumps. This
retention supports a later source-bound diagnosis without repeating compilation
merely to recover timing information.

## Rejected alternatives and limits

Do not increase every CI timeout, remove a qualification target, weaken compiler
input identity, trust PR-produced snapshots, cancel active master producers or
blindly rerun the failed queue. Independent CLI scheduling in #2402 addresses a
separate dependency and composes with this job-local repair.

This is a bounded cold-path reliability allowance, not a measured speedup or
proof that every future cold source build finishes within 30 minutes. No new
GPU/numerical campaign is implied. Normal required CI on the repair and its
synthetic merge source must still pass. Revisit the bound using completed cold
builds and retained per-target timings if a 30-minute compile is insufficient;
inspect the actual long tail before expanding the allowance again.

## Verification

Workflow controls check finite nested timeouts, mandatory gate ordering,
unchanged resource budgets, no consumer cache save, failure-only allowlisted
retention and rejection of weakened variants. Executed diagnostic fixtures cover
missing/partial configuration and exclusion of unrelated secret-like config and
environment values. Existing cache discovery, trust, cleanup, aggregate and
packaging suites remain part of validation.

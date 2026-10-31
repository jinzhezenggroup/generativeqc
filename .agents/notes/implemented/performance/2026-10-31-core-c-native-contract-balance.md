# Decision: balance stationary native contracts across core shards

Status: implemented
Date: 2026-10-31

## Problem and evidence

PR #2873 run `41165110996` and its D3 dependent #2876 run `41165259305`
exhausted the existing 30-minute core-c test step. The retained native
diagnostics from the first run show no failed reports and all four workers
still active at termination. This differs from the late single-file
runtime-heavy tail repaired in #2903.

The complete `test_native_stationary_shared_kernels.py` and
`test_native_stationary_cuda_binding.py` modules account for approximately
525 and 280 reported worker-seconds. In that same run, the whole core-a job,
including setup, finished in 24m39s. These are observations from the failed
run, not a claim that its incomplete test suite passed.

## Decision and invariants

Use the existing explicit core-a owner pins for those two complete modules.
Preserve their ordinary enumeration slots, every selected test, scientific
assertion, scheduler mode, worker count and job/test deadline. Existing
ownership regressions check unique coverage and unrelated modulo owners
across tracked, missing and untracked files and all five CI event classes.

Do not solve this imbalance by retrying the same overloaded shard or
removing numerical coverage. The runtime-heavy ordering repair remains
separate and is inherited from master.

## Revisit when

Check complete hosted CI after this change. If core-a becomes the critical
path or native fixture costs change, use per-module diagnostics from the
same run to rebalance complete owners without dropping tests. The measured
worker sum estimates available load; it does not establish a wall-time win.

# Decision: transfer the complete ECP Hessian module to the ECP force lane

Status: implemented
Date: 2026-10-31

## Problem

PR #2873 run `41174408456`, job `124462471065`, exhausted the unchanged
30-minute Python step. Its retained JUnit contains 2,422 results, zero failures,
zero errors and 1,386 skips, with an elapsed time of 1,810.058 seconds. The four
workers exited successfully, but the enclosing action had already timed out.
Starting slow files early had eliminated the former single-file late tail;
the remaining runtime lane's total work still left no reporting headroom.

`tests/python/test_wb97mv_ecp_hessian.py` accounted for 585.228 worker-seconds.
The same run's ECP force Python step completed in 973 seconds. Even a wholly
serial addition would have been about 1,558 seconds at those observed costs;
this is a scheduling estimate, not a new hosted execution measurement.

## Decision

Transfer that intact module to `ecp-forces`, which already uses `loadfile`.
Retain its old core enumeration slot so unrelated modules do not change owner.
Keep every parameterization, independent finite-difference oracle, missing-file
failure, shared-fixture boundary, and existing routine/nightly time limit.
Remove the transferred file from the runtime priority list while preserving the
remaining whole-file ordering.

## Rejected alternatives

Reordering the runtime lane again cannot remove its total-work bottleneck.
Dropping Hessian cases would weaken scientific coverage. Raising the time limit
would leave useful capacity on the existing ECP lane unused. Adding a runner is
unnecessary for this measured imbalance.

## Evidence and revisit conditions

The existing shard regression suite exercises unique ownership, unchanged core
slots, routine/nightly parity, tracked-but-missing failures, untracked exclusion,
and actual xdist fixture/scheduling behavior. All 686 cases pass after updating
the expected owner. New-head hosted CI must still establish the resulting wall
times; revisit ownership if either lane again approaches its unchanged ceiling.

The original debug artifact is
`benchmark-debug-41174408456-runtime-heavy`, especially
`pytest-runtime-heavy.xml` and `python-diagnostics/controller-4312.jsonl`.

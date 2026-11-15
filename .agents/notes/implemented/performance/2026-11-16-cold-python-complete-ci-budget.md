# Decision: preserve Python test allowances after a cold native build

Status: implemented
Date: 2026-11-16

## Problem

Issue #3086's qualification head
`11b60e83aba7bebc04f9b4204474909e8043acca` encounters a second, distinct
whole-job budget defect after the cold GCC repair. In run `43757181745`,
Python core-c job `132944299405` exceeds its 40-minute maximum. The GitHub
annotation identifies the whole-job deadline, not the test-step deadline.

Its successful CPU native build takes 22 minutes 4 seconds, from
2026-11-15 20:22:41 UTC to 20:44:45 UTC. Setup and diagnostics put the start
of the coverage suite at 20:45:08 UTC. The job cancels that suite at
21:02:06 UTC, after only 16 minutes 58 seconds of its existing 30-minute
allowance and at 88% progress. There is no completed test summary or JUnit
report; the cancelled gate remains failed. Failure diagnostics and partial
compiler-cache work are retained. No OOM or test correctness failure is
inferred from this truncation.

The P1b core-c job `132942940925`, with the same native source inputs,
finishes successfully in 26 minutes 31 seconds. Optional compiler/probe
caches and trusted master snapshots can reduce setup cost, but they are not
prerequisites for executing the full correctness endpoint. Every Python
shard builds the same native CPU library, so repairing only core-c would
leave the same cold-build assumption in the other 40/45/50-minute shards.

## Decision

Give every routine Python shard the existing finite core-b 60-minute
whole-job ceiling. Preserve the separate 30/35-minute test-step limits,
test ownership/selection, coverage, diagnostics and cache producer/consumer
boundaries. This leaves at least 25 minutes for setup and cleanup, rather
than relying on a cache hit to make the test allowance reachable. Preserve
the existing 100-minute nightly/manual whole-job and 90-minute test limits;
this change does not claim new full-tier or cold-nightly qualification.

Evaluate all shard/event combinations in regression tests, accepting
equivalent timeout expressions. Reject the previous short per-shard
ceilings, an unbounded/expanded routine ceiling, extended test allowances,
disabled diagnostics or optional test execution.

## Boundaries

This is a bounded CI endpoint-budget repair, not a compilation speedup or
a retry-until-green policy. Do not weaken scientific tolerances, omit tests,
relabel acquisition identities or turn partial progress into acceptance.
The GCC/Clang repair remains separate; all native CUDA CLI, GPU scheduler
and resource-qualification deadlines remain unchanged. The earlier failed
and successful job receipts remain in ignored local validation storage.

## Revisit when

Measured cold setup approaches 25 minutes, a complete test endpoint exceeds
its existing step allowance, or compilation work can be reduced without
weakening scientific/ownership/diagnostic coverage. Diagnose those cases
with their actual phase timings rather than blindly extending deadlines.

## Related decision

`2026-11-16-cold-gcc-complete-ci-budget.md` records the independent GCC
configure/build/CTest/ownership endpoint truncation and its finite repair.

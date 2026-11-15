# Decision: budget complete cold GCC qualification, not only compilation

Status: implemented
Date: 2026-11-16

## Problem

Issue #3086's updated qualification-gate head
`f15fb5268567e0be92e7dea413d69b1917390c32` reaches the finite GCC job deadline
in CI run `43751061186`, job `132924907861`. The GitHub annotation explicitly
reports that the maximum execution time of 25 minutes was exceeded. This is
not the earlier merge-group hosted-runner communication failure.

Setup/configuration takes approximately one minute. The actual cold build runs
from 2026-11-15 19:12:58 UTC to 19:35:02 UTC (22 minutes 4 seconds). CTest then
reports all 142 tests passed, including the final method-IR code-generation
case, at 19:37:06 UTC, but the job is already cancelled. Ownership-report
generation/upload does not run. Do not treat the partial success as a green
complete qualification or discard the cancellation receipt.

The corresponding P1b head `6c0e89cfff07881d76f10f27de79b7bc509dd0d1`
has identical native source inputs. Its GCC job `132923823106` in run
`43750703653` finishes successfully in 19 minutes 8 seconds, with a 16-minute
15-second build. Both jobs lack a bounded successful-master CPU snapshot.
These observations show meaningful cold-run variation, not a new C++ failure
or proof of memory exhaustion. Raw job logs and annotations remain in ignored
local validation storage.

## Decision

Use a finite 30-minute job ceiling for both GCC and Clang. This preserves
Clang's existing limit and gives GCC about five minutes beyond the observed
25-minute cancellation to finish its complete configure/build/CTest/receipt
endpoint. Keep all full-work conditions, both compilers, fail-fast disabled,
compiler-cache validation and the trusted cache producer/consumer separation.

Test the evaluated budget for every supported routine event and both compilers,
including equivalent constant/expression spellings. Reject the former GCC
25-minute policy and any shortened event-specific path. Retain the independent
tests rejecting skipped builds, optional CTest, premature publication or
coverage/ownership weakening.

## Boundaries

This is an operational cold-job budget correction, not a compiler speedup,
scientific tolerance change, cache-identity override or permission to skip work.
The 210-minute native CUDA CLI job and every GPU scheduler deadline remain
unchanged. No source, kernel, default, numerical receipt or generated inventory
is relabeled. Thirty minutes is finite headroom, not a guarantee that any future
growth will fit; a later overrun still needs measured diagnosis.

## Revisit when

Fresh cold jobs approach the complete endpoint ceiling, or measured compilation
work can safely be reduced without removing native tests or receipt ownership.

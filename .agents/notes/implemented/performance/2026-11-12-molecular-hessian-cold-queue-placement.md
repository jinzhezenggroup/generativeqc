# Decision: move intact molecular Hessian qualification to the force lane

Status: implemented
Date: 2026-11-12

## Evidence and diagnosis

PR #3057 source b6326d49 and protected queue 8d5b7b64 have the same tree.
Source core-c job 131232342551 passed in 1091.58 seconds with 18,983 passed,
6482 skipped and 67 subtests passed. Queue run 43237971940 core-c job
131250443753 was cancelled: its public annotation explicitly reports the outer
40-minute maximum. The test step also ran from 14:41:01 to 15:11:13, near its
30-minute limit. Raising only the outer budget is not demonstrated sufficient.
No complete queue XML survived. Retained diagnostics contain 18,135 passed and
585 skipped call reports with four workers still active, not a full-suite pass.

The source probe cache recorded 967/1123 hits and 156 misses; the queue recorded
448/1074 hits and 626 misses. The queue legitimately restored a verified
successful-master fallback rather than the PR's cache. Both native builds had
374/408 hits and 34 misses. All 13 molecular Hessian cases passed in retained
queue diagnostics, using 2119.580 worker-seconds versus 376.813 in source XML.
The six UKS Hessian cases used 641.734 versus 60.320 seconds; other matched
calls over one second had median slowdown about 1.107. This strongly suggests
cold compiler work dominating these response tests, not proof of a new
scientific or production-runtime regression.

## Decision and invariants

Move the entire `test_global_hybrid_molecular_hessian.py` to the existing
`uccsd-forces` worksteal lane, whose test step took 273 seconds in this queue.
Keep RKS first and molecular curvature immediately after it, ahead of force
matrices; seed fresh cases across the same four workers with the existing
priority plugin. The molecular module's state is function-scoped. Preserve
all 13 cases, scientific assertions, fresh setups, markers, time limits and
worker counts. It previously occupied a supplemental non-modulo slot, so
exclude it before ordinary core enumeration rather than reserving an ordinary
slot. Every unrelated file retains its owner and relative target order.

The existing UKS placement remains untouched. No source generation, numerical
threshold, cache identity or queue trust rule changes. Do not use untrusted PR
cache output to hide the colder protected-queue workload.

## Limits and alternatives

The force lane has measured spare capacity, and moving this large completed
workload off core-c is a bounded plausible improvement. Worker-seconds are not
elapsed seconds: neither 2119.580 / 4 nor addition to 273 predicts the new wall
time. Different cache content, four-way compile contention and residual core-c
work can still determine the critical path. A hosted run must establish that
both lanes finish; local scheduler regressions do not establish that outcome.
No new GPU, scientific or performance qualification is claimed by this change.
Broad timeout expansion and fixture/threshold weakening are not justified.
Revisit placement if a cold protected-queue run still exhausts either lane.

## Validation

Executable selector tests cover all five workflow events, absent/untracked/
tracked/missing file inventories, exact ownership and no unrelated modulo
migration. Real four-worker worksteal tests exercise all 13 synthetic molecular
cases alongside existing force modules, fresh setup once per case and molecular
starts before force cases on each worker. These model scheduling, not science.
A one-off comparison against b6326d49 preserves every other target sequence and
distribution on the full tracked inventory, without pinning source fingerprints
in permanent CI.

References:
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43237971940/job/131250443753
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43232809405/job/131232342551

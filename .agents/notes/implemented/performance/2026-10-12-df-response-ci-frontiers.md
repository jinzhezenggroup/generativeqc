# Decision: seed fresh A/C frontiers after the remaining DF-defaults tail

Status: implemented
Date: 2026-10-12

Run 38239287435 on reviewed head `47758148` retained the landed native
compiler transfers. Core-c job 114777669031 nevertheless reached the
unchanged 30-minute test-step cap at 11:32:50 UTC, with one corrected-Lambda
qualification case still active and no visible assertion failure. All other
required jobs passed. Observed complete pytest seconds were A 1222.29,
B 879.24, runtime 525.40, posthf 944.89, compiler 678.59 and ECP 913.68.
Completed corrected-Lambda start/result intervals sum to 1537.460 worker-
seconds, not complete fixture costs or a wall-time bound.

Reuse the unchanged reviewed `tools/pytest_ci_fresh_frontier.py` from
#2355 `a03ebbe4`, with its exact 36 property/subprocess regressions. The
original donor run 38236462737 passed all seven shards and all 135 ownership/
plugin tests. That validates the donor's collection contract and actual
run, not the deadline of a different inventory. Some donor margins were thin.

Only existing standard-gradient core-a and corrected-Lambda core-c opt in;
core-b emits no priority metadata. Keep this PR's ordered file owners,
original modulo slots and landed native transfers unchanged. No corrected-
force source or #2355 B/Fock ownership policy is imported. Every scientific,
OOM-recovery and evidence-reconstruction blob remains exact reviewed head.

The plugin only permutes final collected item objects across the existing
worksteal startup queues. It preserves selected/parametrized nodes and ordinary
relative order, rejects unsupported parallel modes before execution, and
uses actual worker counts. It has no runtest/report/error hooks or fixture
replacement. Function-scoped independent owners remain fresh; serial runs
and dedicated loadfile lanes are unchanged. All negative controls, numerical
gates, worker counts and budgets remain.

Selected tracked files alone receive metadata. Untracked/absent files are
not admitted, and tracked-missing files still fail collection. Static
ordered-inventory checks cannot substitute for executing Bash/pytest;
executor ENOSPC prevented local execution. Current hosted CI and the normal
merge queue remain the actual endpoint/deadline gates. Revisit after fixture
scope or scheduler-startup changes, using the retained live behavior tests.

The true master integration also retains landed #2356 at
`c4b455f29d21bd296bebd82f878398aac2c41814`, including its response-to-B
pin, complete automatic-CUDA source and existing ownership probes. The shared
A/C frontier composition changes no ordered target relative to that current
master routing. This is separate from the observed prior-run inventory.

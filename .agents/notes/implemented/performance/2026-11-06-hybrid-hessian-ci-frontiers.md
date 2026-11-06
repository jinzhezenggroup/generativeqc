# Start fresh hybrid Hessian qualifications before the core-b tail

Status: implemented

## Evidence

PR #3017 head 1127c311338da1c26302fc1ffb49e791295db381 passed PR CI
42125152009, including core-b with 18,256 passed, 5,168 skipped and 214
subtests in 1382.77 seconds. The same branch's merge-group
ea2918330702f042fc03c23fd4940a5f58ea0f5f failed CI 42128769441:
core-b job 127599728016 reached 99% and hit the unchanged 35-minute action
limit. This was a timeout, not a reported numerical assertion failure.

Preserved diagnostics artifact 13464093014 (SHA256
5ba7afedee0825259e10ff16f63b8470c25f90f70265ba9ba87f66cad1361dd0)
shows three workers still executing the public and UKS global-hybrid Hessian
suites while the fourth finished. The spherical raw-Hessian and wrong-spin
cases passed in 94.82 and 101.40 seconds; B3LYP public recovery passed in
105.11 seconds. Their Cartesian/PBE0 siblings were still active at termination.
The controller recorded no OOM events. Probe compilation had 932 misses among
938 cacheable calls, and compiler processes were active at timeout.

## Decision and boundaries

Add both complete, function-scoped suites to the existing fresh-frontier
metadata. The existing worksteal plugin spreads their cases at the beginning
of the four worker queues. It changes collection order only, following each
file's existing selected shard rather than assigning a new owner.

Preserve every file, case, fixture lifetime, ordinary relative order, marker,
numerical gate, worker count, timeout, cache identity and cache trust policy.
Do not hide the timeout by increasing limits, moving cases to nightly,
rerunning until green, or cancelling existing work. No LR production source,
GPU qualification or historical racecheck limitation changes. Hosted runtime
improvement remains to be verified on the new head and normal merge queue.

Regression coverage exercises real four-worker startup for all three hybrid
Hessian suites, exact case/setup inventory and existing ownership. Tracked,
untracked, absent and missing files retain ordinary modulo owners and missing
file failures across bucket offsets and PR/queue/nightly modes.

## Separate clean-tree evidence repair

Running the retained LR replay from a fresh checkout exposed an accidental
checksum entry for ignored `__pycache__/replay.cpython-313.pyc`. That transient
file is not in Git and made the documented replay fail before any scientific
checks. Remove only that entry, retaining all tracked receipt digests and gates.
The complete offline replay and all remaining checksum checks then pass.

Local validation: 842 shard/collection regression tests pass. A separate
before/after comparison across all 40 event/shard combinations preserves exact
file order, ownership and other controls; only the two priority arguments are
added to core-b. Ruff, Bash syntax and whitespace checks pass. Initial local
collection failures due to absent test-environment dependencies were resolved
using existing cached packages; no test was removed. This is scheduling and
offline evidence validation, not a new scientific or GPU campaign.

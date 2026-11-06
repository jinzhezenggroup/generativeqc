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

## Follow-up: cold total work still exceeds the core-b budget

The first repair's PR core-b run passes in 761.52 seconds with 18,256 passed,
5,168 skipped and 214 subtests, but this warm run has 838/955 compiler hits.
Do not attribute that entire improvement to priority. Merge-group CI
42137083928, job 127627899607, times out again at the 35-minute action limit
with 886/892 compiler misses. All molecular, UKS and public hybrid Hessian
cases now finish early and pass; four workers remain active on ordinary
cases. Priority fixed the original tail placement but not cold total work.

The normal cache is absent on the queue ref and bounded trusted-master
snapshot discovery correctly finds none. The last three successful master CI
runs (41961283952, 41766814271, 40997389481) have no matching core-b snapshot
artifact. Preserve this fail-closed cold path and its trust boundary.

Transfer only the complete molecular Hessian suite from its supplemental
core-b owner to supplemental core-c. Its 13 passing cases consume 1634.1
worker-seconds in the failed queue run. In that same run core-c completes its
Python step in 806 seconds. The idealized four-worker increment is about
409 seconds, providing substantial measured headroom without another runner.
This estimate is not a hosted timing guarantee; the actual new queue remains
the acceptance check. Preserve its fresh-frontier distribution, all scientific
cases, ordinary modulo slots, markers, fixtures, four workers and time limits.
The earlier opposite transfer preceded fresh-frontier scheduling and responded
to a different historical inventory. Current core-c itself had 802/854 compiler
misses, so its headroom is not inferred from a warm-cache run. Actual new queue
validation is mandatory; this estimate does not promise a successful runtime.

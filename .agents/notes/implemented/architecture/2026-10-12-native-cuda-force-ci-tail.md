# Native CUDA force CI tail scheduling

Agent: dot

PR #2357 source review remains separate from this CI-only repair. Hosted run
38239147055, job 114777166769, executed all 11244 core-C tests and 112 subtests
successfully but crossed the unchanged 30-minute step deadline at 30:05.
The final tail was fresh corrected UCCSD(T) Lambda qualification, failure and
lifetime cases, with four worksteal workers. No numerical test failed.

Reuse the exact landed phase-two shard routing from master 61a0e715 together
with the reviewed A/C-only fresh-frontier orderer from 3deb19c. Executable blobs:
selector bae7f19d6f0f937f1c07ce8033c10f86f41b61ef, ownership tests
913ce33c526a1702c37abec28d3442dbf2754260, plugin
86571da52ec1f01b79a1eacd3e61d3040da7db95.

Against #2357 parent 4cfb3359, an in-memory interpretation of the shell arrays
and branches checks the complete tracked tests/python inventory (1659 files)
for pull_request, push, merge_group, schedule and workflow_dispatch. Every
file has exactly one owner; there are no missing, duplicate or extra targets.
The already-landed routing changes only three actual owners: native TensorIR
CPU and native UCCSD move to compiler-heavy; high-angular CPU HF moves from C
to A. The frontier metadata preserves every ordered file list relative to that
landed router. Existing deselections and coverage controls are byte-identical.
Only selected gradient cases on A and corrected-Lambda cases on C receive
collection-order metadata; B receives none.

Routine file counts are A 528, B 527, C 523, runtime 22, posthf 19, ECP 7,
compiler 33. Full scheduled/manual counts are A 527, B 527, C 523, runtime 26,
posthf 15, ECP 7, compiler 34. Existing nightly Hessian/WB97M-V ownership,
worker counts, fixture scopes, scientific assertions and time limits remain.

The unchanged donor plugin has property and real-xdist subprocess controls
for exact case preservation, selected-only priority, parallel worker startup,
serial invariance and invalid-distribution rejection. Hosted donor run
38236462737 passed all seven shards and all 135 ownership/plugin tests.
Donor timings have thin margins and do not guarantee #2357 timing success.

No local shell or Python tests were executed in this ENOSPC environment.
The in-memory inventory check is not a Bash/pytest pass. Exact-head hosted
CI remains responsible for executable selection, plugin and runtime results.

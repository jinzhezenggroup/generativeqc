# Decision: split complete unrestricted response tails across parallel CI owners

Status: implemented
Date: 2026-10-12

## Problem and measured evidence

The previous gradient and corrected-Lambda pins both selected core-a.
PR #2347 head c7588795559a49b802905d4d6f34383e87ac0eb9 passed run
38209674912 on merge tree dd5564526 with core-a/core-b/core-c pytest
times of 1583.79/1212.70/939.44 seconds. PR #2350 head
7a5752732c440c4d5bbacf445dfe4508a40fb19a used the identical selector,
but run 38212023906 on merge tree 0128302d8d40f53402087ae468a09a40d5b423d7
hit the unchanged 1800-second core-a deadline. Its core-b/core-c finished
in 1434.29/1328.00 seconds; all 19 new Fock host cases passed on core-b.
The five completed molecular gradient calls in the failed shard and its
late corrected-Lambda cases showed no numerical assertion failure.

This is not an identical inventory or runner-speed comparison: the new
Fock test filename shifts later unpinned modulo owners. Some unchanged
molecular gradient calls also took 30-45 percent longer in the later run.
Source identity alone is insufficient evidence that a near-budget shard
will continue to finish.

PR #2351 run 38213658670 tested merge tree
0de1d79a8c0b8f8897f0a0ae09b967234b8f8bbe with the older master selector.
Its core-b timed out with standard gradient cases unfinished; core-a and
core-c completed in 1031.25 and 1186.45 seconds. This test-only publication
migration did not introduce the scientific tail.

## Decision

Keep the complete gradient file on parallel core-a workers, pin corrected
Lambda to core-c, and pin the full-Fock response matrix to core-b. Transfer
the existing complete native-CC TensorIR CPU and high-angular HF files from
core-c to core-a to leave room for corrected Lambda. In the #2350 core-c
log those two files account for approximately 457 summed seconds between
individual test-start/result records. This observation includes only
visible completed intervals; it is not full JUnit phase timing, a whole-file
wall time, or a prediction of the repaired shard duration.

All five files retain worksteal, fresh fixtures, every case and original
core inventory slots. Untracked or absent future files are not selected;
tracked missing files remain selected and fail closed. Dedicated loadfile
transfers continue taking precedence. Other files retain their current
modulo owner, and routine/nightly exclusions are unchanged.

## Rejected alternatives and boundaries

Do not serialize the large UCCSD files under loadfile, share their mutable
function-scoped owners, skip numerical/fault cases, raise timeouts, add
workers, clear caches, or change production CPU/CUDA equations. Do not
blindly frontload all heavy files: contiguous initial xdist chunks can
concentrate them on one worker. These are CI ownership changes, not a
scientific performance optimization or a complete-endpoint speed claim.

The real shell selector and ownership regressions exercise all five event
types and absent, untracked, tracked and tracked-missing inventories.
Complete reconstructed master/PR inventories must each have exactly one
owner per test file. Such checks establish selection and preserved coverage;
actual runtime remains unmeasured until the new full CI run completes.

This supersedes the paired core-a placement, while preserving the earlier
evidence in [the original owner-balance note](2026-10-12-uccsd-ci-owner-balance.md).

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38209674912
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38212023906
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38213658670
- tests/python/test_ci_shard_ownership.py

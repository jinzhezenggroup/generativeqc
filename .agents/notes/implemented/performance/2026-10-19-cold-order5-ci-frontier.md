# Decision: start the complete cold order-five host gate early in core-c

Status: implemented
Date: 2026-10-19

## Problem

PR #2611 CI run 39207592326, core-c job 117913720180, hit the unchanged
30-minute Python coverage step limit with no failed assertion or OOM record.
Per-worker native diagnostics show gw1/gw2/gw3 finished normally. The sole
active node on gw0 was the complete order-five independent-value/native-source
adapter test. It started at 06:31:34 UTC, only about 195 seconds before the
global timeout, before its own 300-second compiler limit expired. The final
printed cache-contract test had passed; log order alone misidentified the tail.

## Decision and invariants

Add the already selected, pinned core-c order-five file to the existing fresh
frontier metadata. Reuse the worksteal collection-order plugin. Do not change
file ownership, collection inventory, worker count, numerical gates, coverage,
compiler options, cache correctness, or timeout limits. The test has one
function-scoped native probe, so early startup does not duplicate module setup.
The selector still emits priority metadata only for tracked selected files;
tracked-but-missing files remain selected and fail closed.

Extend the existing synthetic all-frontier inventory cases to include this
owner, with no new parametrized cases. They cover all CI event types, ordinary
modulo slots, absent/untracked/tracked/missing files, and unique ownership.
Existing actual xdist controls qualify early worker startup and unchanged
selected/deselected outcomes.

## Evidence and rejected alternatives

Exact-source isolated replay on Inspire passed in 158.76 seconds with 109,728
independent coordinate checks (maximum error 6.84331e-10) and 746,496 adapter
comparisons (maximum error 1.66533e-16). This is not hosted-runner timing or GPU
qualification. Verified ccache 4.14 before compilation and retained before/after
statistics: 0 cacheable calls and 1 uncacheable call. The retained test combines
compile and link; restored shard caches cannot establish a hit for this probe.
Do not describe the isolated replay as a complete shard pass.

Relying solely on a warm rerun leaves the uncached late compilation vulnerable.
Raising time limits, dropping the gate or its independent work, weakening
coverage, and changing scientific production code do not address startup order.
A broader compiler-probe caching refactor is unnecessary for this scoped repair.

## Revisit when

Use new complete-shard worker diagnostics if another cold workload becomes the
critical path. Preserve exact inventory and independent scientific acceptance;
measure the complete shard before making hosted timing claims.

## References

- PR #2611 and original issue #2137.
- `.github/scripts/ci_select_python_shard.sh`.
- `tools/pytest_ci_fresh_frontier.py`.
- `tests/python/test_ci_shard_ownership.py`.
- `tests/python/test_direct_order5_sources.py`.

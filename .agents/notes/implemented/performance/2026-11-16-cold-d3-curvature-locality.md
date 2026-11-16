# Decision: keep public D3 curvature qualification file-local

Status: implemented
Date: 2026-11-16

## Problem and retained failure

PR #3091 author head `9e0da797` passed all checks, but its exact-tree protected
group `5f18ae11` failed the unchanged 30-minute core-a test allowance in run
43819312426. The test step lasted 30m13s and ended at 99%. This is a real failed
qualification, not a success inferred from the preceding author run.

Retained native-diagnostic histories show no reported assertion failure before
the deadline. Workers independently started the module-scoped `public_results`
fixture in `test_d3_public_second_order.py` late in the worksteal queue. One
first test took 225.36 seconds from start to report, dominated by setup; another
worker was still setting up the same fixture for the joint-budget failure
case after all three peer workers had finished. The fixture constructs three
complete electronic/dispersion curvature endpoints before yielding detached
results. Core-a also performed 1,023 cold compiler-cache misses. In the same
protected run, runtime-heavy completed its test step in 13m16s.

## Decision

Move this intact module to the existing runtime-heavy loadfile lane and start
it before the ordinary runtime queue. This shares the existing module fixture
within one worker rather than duplicating late cold setup across workers. It
does not share a live electronic owner, alter fixtures, skip cases, or change
scientific oracles, failure injection, admission gates, or production code.

The module was a supplemental core-a owner with no ordinary modulo slot.
Exclude it before supplemental dispatch and retain that no-slot behavior; do
not shift any unrelated core owner. Keep the separate public-hybrid Hessian
suites on their established parallel fresh frontiers. All event selections,
diagnostics, cache rules, worker counts, and finite job/test deadlines remain.

## Validation and limits

Executable selector regressions compare all five CI events and every shard,
preserving every selected file, coverage flag, deselection and unrelated owner.
The existing real xdist loadfile regression includes the D3-priority module,
checking one module-fixture setup and unchanged node/outcome coverage. Actual
new-head and protected CI must still qualify the complete endpoint execution;
synthetic scheduler costs do not certify production timing or numerical science.

Native inputs and the #3086 acquisition remain unchanged. Preserve the failed
protected run and all positive/negative scientific observations. No time limit,
global default, physical tolerance or release policy is relaxed.

Revisit if measured runtime-heavy load loses sufficient bounded headroom or
this module acquires an explicit parallel fresh-fixture contract.

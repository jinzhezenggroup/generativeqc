# Decision: Preserve Python test deadlines after a cold native build

Status: implemented
Date: 2026-11-12

## Evidence

PR #3062 head `c51adb1f5aa76d3ef52d1e82808631b8b107ab66`, CI run
`43052160517`, attempt 2, ECP job `130875676222` reached its 40-minute
whole-job deadline. This was an automatic timeout, not an operator cancel.

- Setup began at 00:10:11 UTC. Native configure/build ran from 00:10:54 to
  00:26:05; installation and probe-cache preparation finished at 00:26:20.
  The build completed with 176 compiler-cache hits and 232 misses. The
  restored trusted cache was valid; changed inputs genuinely required work.
- The new Gaussian ECP graph includes 133 generated translation units.
  Its final compilation completed at 00:26:05. This is additional build
  work, not evidence that the compiler-cache launcher was bypassed.
- Pytest began at 00:26:20 and reached 88% at 00:34:10, then was terminated
  at 00:50:23. It received only 24m03s of its existing 30-minute allowance
  because setup had consumed 16m09s of the whole-job budget.
- `loadfile`, the first public-CPU target, and `--no-loadscope-reorder` are
  retained from the preceding scheduling repair. No ECP test-file change
  occurred between `a154dc140c` and this head.
- No ECP JUnit or debug archive was retained. The debug step required
  `failure()`, while the whole-job timeout produced `cancelled`. Therefore
  the exact last node and whether its numerical gates would pass cannot be
  reconstructed from this run. An absent watchdog message does not prove
  progress: its Python thread can be delayed by native code holding the GIL.

The earlier [file-priority diagnosis](2026-11-11-ecp-force-file-startup-priority.md)
documents the expensive f-force matrix. That earlier run predates this
native build graph and the priority repair; it is not a successful matched
runtime baseline. The first published-priority run at `878b89d1b` failed
in dependency installation before compiling or testing. Do not claim a
complete hosted ECP pass from either record.

## Decision

Increase the routine ECP whole-job ceiling from 40 to 50 minutes, retaining
the 30-minute pytest deadline. The measured setup plus that full test
allowance is about 46 minutes. Increase the routine core-b job ceiling from
45 to 60 minutes, retaining its 35-minute pytest deadline: measured setup
plus that allowance is about 54 minutes. The remaining time bounds cleanup
and artifact retention. Nightly/manual allowances and all other shard
ceilings are unchanged. Scientific cases, tolerances, compiler options,
cache keys, cache ownership and scheduling are untouched.

Retain the existing Python debug directory on failure or cancellation, with
a two-minute upload cap and nonfatal transport outcome. This also retains
already-produced diagnostics after an ordinary operator cancellation; it
does not initiate a cancellation or convert a test failure into success.
Retention is best effort, not guaranteed after a hard whole-job deadline.
The observed timed-out job still ran cache-statistics and JUnit-upload steps,
so the missing debug upload had a concrete condition-level cause. The new
condition admits both a failed test step and a cancelled job when the runner
still executes cleanup, but cannot extend the runner's cancellation grace
period or recover files after that runner has been terminated.

## Separate CPU build limit

The same retry's GCC job `130875677414` and Clang job `130875676159`
continued producing objects until their 15/20-minute job limits, at 497/729
and 555/729 targets respectively. Cache summaries showed 193 hits/226 misses
and 223 hits/254 misses. Neither reached CTest; no compiler failure was
reported before termination. Their old successful `a154dc140c` jobs built
591 targets in 382.5/406.2 seconds and ran 140 CTests in 107.5/66.3 seconds.
Those old totals exclude the new Gaussian graph and cannot establish its
complete current budget. This scoped change leaves CPU budgets unchanged;
they remain separate unresolved timeout failures needing a bounded remedy
and complete current-build/CTest evidence. No CPU optimization is attempted.

Core-b job `130875676239` subsequently timed out at its 45-minute job cap.
Its setup took 19m03s, leaving only 26m08s of the configured 35-minute test
allowance. Tests progressed to 96% just before termination, but the quiet
log contains one actual failure marker at 00:54:33 UTC (92%). No traceback
or completed JUnit survived. That is a real unresolved test failure as well
as a truncated test allowance, not an unqualified missing-evidence result.
The added core-b headroom preserves its full test deadline so normal CI can
finish reporting failures. By itself, this budget repair does not declare
the failed test passed or resolve its unknown cause.

### Follow-up: independently reproduced core-b admission failure

The exact c51 core-b collection includes
`test_hessian_public_rks.py::test_public_rks_second_order_capability_fails_closed[b3lyp-uks-kwargs2]`.
This older test still rejected direct CPU/all-electron/FP64 B3LYP UKS after
`e5c6011017` intentionally admitted that context. On a fresh source-matched
c51 CPU library, the original complete module produced one failure at this
assertion and eleven passes. The test-only repair adds a positive B3LYP UKS
context and retains its refusal with explicit CPU density fitting. PBE0/LDA
UKS, r2SCAN, spherical PBE and the other genuine boundaries remain intact.

The repaired module plus the fourteen changed core-b modules selected 211
routine cases: 193 passed and 18 explicitly required GPUs and were skipped.
The three existing nightly exclusions remain; separate registry consumers
passed 27 tests and compiler/D3 boundary neighbors passed 13. The searched
757-file exact-shard inventory contained no other blanket B3LYP UKS refusal.

These before/after results use native source identity
`52e227f1e90e9a1b78d680aa07fa140efabe9fbca8ca1ba12b791e10c09d1c48`
and retain c51 provenance. They are local Python 3.12/scalar-CPU results, not
the combined master-integration tree, CI's Python 3.11/OpenBLAS wheel, the
full shard, or new molecular/GPU qualification. The old CI log lacks a node
ID, so its historical printed failure cannot be uniquely identified as this
case. This is an actual independently reproduced exact-shard defect and
repair, not an attribution from progress percentages or a later green run.

## Verification and limits

Workflow regressions evaluate all event-specific ECP/core-b job/test deadlines
and failure/cancellation/prose upload conditions. Existing tests retain every
other shard ceiling, required CPU build/CTest work and compiler-cache
ownership. The unchanged whole-file scheduling regression is rerun.

No scientific suite or native build is duplicated for this workflow edit.
A fresh normal CI run is still required to show whether the unchanged
30/35-minute test allowances suffice and to expose complete failure details.
If it is not, use retained per-test diagnostics before considering another
change. Do not remove gates or increase the test deadline merely to pass.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875676222
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875677414
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875676159
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875676239
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/42975834279
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43045778989

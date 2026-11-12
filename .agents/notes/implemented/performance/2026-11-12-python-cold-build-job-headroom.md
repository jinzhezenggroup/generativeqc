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

### Next-run core-a headroom and completed neighboring shards

In [run 43138869481](https://github.com/jinzhezenggroup/generativeqc/actions/runs/43138869481),
source `6667d3722e5f0aa39a5ad8b26bad50f03afbbe20` used synthetic merge
`cc62754d44a5406e910e3a2bd45c7b7645b3b596`, whose tree exactly matches the
source tree. Core-a job `130920737536` reached its 40-minute whole-job cap:

- Setup began at 02:04:56 UTC. The native build took 17m22s, from 02:05:45
  to 02:23:07; remaining preparation finished at 02:23:39. Total setup was
  18m43s. Pytest ran for only 21m29s before cancellation at 02:45:08,
  despite its unchanged 30-minute step deadline, and reached 99%.
- No failure marker appeared in the quiet progress and no retained diagnostic
  report failed. The cancellation debug artifact `13910187745` was uploaded
  successfully. One worker's history reached its bounded 4-MiB cap; no final
  JUnit or pytest summary exists, so this is not full-shard acceptance.
- The retained report explicitly records
  `test_native_stationary_owner_ports.py::test_stationary_owner_port_forwarding_and_refusals`
  passing in 5.1533s. Its preceding compiler-fixture defect is fixed; the
  later whole-job timeout does not reverse or replace that recorded result.

Increase only routine core-a's outer ceiling from 40 to 50 minutes, preserving
its 30-minute pytest allowance. Twenty minutes remain for setup and cleanup:
the measured 18m43s setup plus the full test allowance totals 48m43s, leaving
1m17s against the seven seconds of observed cleanup/upload. This is bounded
planning headroom, not a measured full core-a completion time. Preserve every
test, inner deadline, diagnostic limit, command, coverage path, cache and
scheduling choice. Nightly/manual core-a and other Python shard caps remain
unchanged. Revisit using the next complete run if this margin is insufficient.

At this same source, core-b completed with 17,207 passes, 5,705 skips and 2,730
subtests in 1427.94s; ECP completed with 260 passes and 66 skips in 1738.85s,
inside its unchanged 30-minute pytest deadline. These are actual hosted
results at their recorded identity, not a retroactive pass of the earlier
timeouts or broader Ni, mixed-ECP or molecular-TS qualification. The separate
[GCC headroom decision](2026-11-12-cpu-gcc-ci-build-headroom.md) records its
remaining incomplete build and the unchanged successful Clang ceiling.

Workflow regressions evaluate every Python shard/event job and test deadline,
both CPU compiler ceilings and failure/cancellation/prose upload conditions.
Existing tests retain required CPU build/CTest work and compiler-cache
ownership. The unchanged whole-file scheduling regression is rerun.

No scientific suite or native build is duplicated for this workflow edit.
A fresh normal CI run is still required to complete core-a and GCC. Use
retained per-test diagnostics if either remains incomplete or reports a real
failure. Do not remove gates or increase the test deadline merely to pass.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875676222
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875677414
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875676159
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43052160517/job/130875676239
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/42975834279
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43045778989

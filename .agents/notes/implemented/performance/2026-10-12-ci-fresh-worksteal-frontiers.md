# Decision: seed fresh qualifications at each worksteal worker frontier

Status: implemented
Date: 2026-10-12

## Problem and measured evidence

PR #2350 run 38219795252, job 114713251749, still hit the unchanged
30-minute Python test-step deadline after the first parallel-owner rebalance.
Actual pytest reporting began at 08:08:31 UTC, with four workers and 12149
collected items. Corrected Lambda first appeared at 08:25:14; the step timed
out at 08:33:54 with late cases unfinished. Visible completed cases have no
numerical assertion failure. Their start/result intervals sum to about 1167
seconds for corrected Lambda and 395 for the native UCCSD producer file;
these are not full fixture timings or a prediction of whole-shard runtime.

The later parent transfer of two complete native files to compiler workers
is retained unchanged. The concurrent #2355 force-owner split at
`191909fc4fd9a5f9a2e9d5769523fd36be26246e` is also preserved: standard
gradient stays on core-a, corrected gradient uses core-b, and full-Fock uses
compiler-heavy loadfile. Their ownership/cost policies and actual hosted
runtime remain distinct from the startup-order problem addressed here.

Simply prepending a heavy file is not sufficient: the existing worksteal
scheduler initially divides a complete collection into contiguous worker
queues. One prepended fresh family can concentrate on one worker instead of
starting across the existing workers. This was already a rejected alternative
in the parallel-owner note.

## Decision

Keep the shell selector's file ownership, original modulo inventory slots,
dedicated transfers and scientific selection unchanged after composing both
independent ownership repairs. Define the fresh
function-scoped families once in its frontier arrays, reuse those arrays for
their existing pins, and emit non-filtering metadata only for selected files.
Absent/untracked future paths are not admitted; tracked missing paths still
fail closed through ordinary collection.

An explicitly loaded pytest plugin reorders only the final selected items.
Compute the existing contiguous worker queue capacities, distribute fresh
cases round-robin within those capacities, and fill remaining positions with
ordinary cases in their original relative order. Every item index appears
exactly once. Only the three parallel force/corrected-Lambda files opt in;
the full-Fock file retains dedicated loadfile scheduling. Serial execution
remains unchanged; dedicated loadfile owners do not load this policy.
No module/function fixture is replaced, shared or
upscoped, and there is no reference, native compilation or scientific replay
inside the ordering plugin.

Validate the selected scheduler on the controller, before worker startup.
xdist resets each worker's local `dist` to `no` and `numprocesses` to None;
those local values cannot certify the controller's distribution policy.
Use the worker's actual `workercount` for deterministic collection ordering.
Unsupported parallel modes fail closed on the controller, before any cases
execute. A first actual-plugin probe exposed this reset; its failure is not
counted as successful qualification.

## Evidence and invariants

- Executable inventory properties cover empty, serial, remainder, more-workers-
  than-items, mixed and all-fresh collections, with exact item preservation
  and ordinary relative order. Invalid worker counts are rejected.
- Real four-worker pytest subprocesses observe each worker starting a fresh
  case, every selected node executing exactly once, and ordinary `-k`
  deselection remaining effective. Unsupported load scheduling is rejected
  before case execution; actual serial execution preserves collection order.
  These are behavior tests, not scheduler-body hashes
  or mandated internal call counts.
- The shell regressions retain all five CI event types and tracked, absent,
  untracked and tracked-missing inventories. Every current file retains one
  owner; no unrelated modulo owner moves.
- An opt-in 120-case probe of the actual plugin records all four worker
  frontiers while preserving all cases. Its synthetic wall time is not a
  complete scientific endpoint measurement or a claim of hosted speedup.
- The final composed selector and actual-plugin regression passes all 135
  cases in 13.39 seconds with pytest 9.1.1 and pytest-xdist 3.8.0. This includes
  actual serial execution preserving order, four-worker startup, filtered
  inventory and early rejection of unsupported parallel scheduling.
- An opt-in comparison against the composed selector before frontier metadata
  preserves all 1657 ordered targets, modes and ordinary selection arguments
  across all five events and seven shards. The force split and parent native
  transfers are retained; the ordering policy makes no further owner move.
- Explicit `ty` with the repository root as an additional tools-module search
  path, Ruff check/format, Bash syntax, compiler/runtime/public-docstring
  audits and the standard Sphinx warnings-as-errors build pass. Six scientific
  composition checksums pass, and all UCCSD owner/compiler/scientific-test
  sources remain identical to `6b066d3fa3ed6a333e54303b02ee5e9334fb78b3`.
- All 16 complete corrected-force CPU cases pass in one terminal-success
  four-worker run using the actual frontier plugin, with PySCF 2.14.0 and
  single-threaded BLAS/OpenMP. Elapsed time is 249.49 seconds on this local
  machine, not a hosted-shard speedup or deadline claim. Independent
  all-coordinate finite differences, valid restricted analytic/RCCSD(T)
  parity and transactional admission/failure gates remain unchanged.
  The existing native library is reused; this is not a current-master native
  rebuild or a new GPU qualification. Prior source-matched four-case CUDA
  evidence under finite Slurm 7828 is retained.
- Subsequent integration of master
  `b836b89bfbd6b8e9a71faf59b371dcf9083ec286` retains the same UCCSD scientific
  sources. Its 135 selector/plugin cases pass in 14.77 seconds, and the
  current 1660-target inventory retains exact composed ownership, order,
  mode and ordinary arguments on every event. Compiler/runtime/public-API
  audits, explicit type checks, formatting and Sphinx warnings-as-errors
  pass again. The increase from 1657 targets is inherited new-master test
  inventory, not added scientific work in this scheduling policy.
- Keep four CI workers, existing 30-minute test-step limits, complete numerical
  and failure cases, independent fresh owners and compiler caches. Production
  CPU/CUDA equations, precision, derivative gates and no-fallback contracts
  are unchanged.

## Consequences and revisit conditions

Fresh work begins before a late alphabetical tail without increasing its
semantic scientific work. Ordering adds linear collection bookkeeping, not
new scientific execution. Whole-shard deadline acceptance still requires the
current hosted run, including setup, coverage, compile-cache misses and every
selected test. Do not present startup evidence as proof of that endpoint.

Revisit if selected suites introduce expensive shared/module-scoped fixtures,
if a scheduler's actual startup distribution changes, or if hosted timings
show another complete-shard bottleneck. Preserve live startup and selection
coverage instead of freezing a vendor implementation fingerprint.

## References

- [Parallel-owner decision](2026-10-12-uccsd-parallel-tail-balance.md)
- [Native compiler-lane transfer](2026-10-12-uccsd-native-compiler-transfer.md)
- [Standard/corrected force-owner split](2026-10-12-uccsdt-force-owner-split.md)
- `.github/scripts/ci_select_python_shard.sh`
- `tools/pytest_ci_fresh_frontier.py`
- `tests/python/test_ci_shard_ownership.py`
- `docs/developer/uccsd_physical_response.md`

# Proposal: evidence-preserving closeout of nine requested issues

Status: proposed
Date: 2026-11-04

## Scope and baseline

Requested issues: #1118, #762, #597, #1629, #2681, #2751, #2915, #2937,
and #2293. All nine were OPEN at the initial check. On the final 2026-11-04
check, #762 was CLOSED as NOT PLANNED / superseded tracking, with its unfinished
production-science inventory transferred to #934; this is not scientific audit
acceptance. The other eight remained OPEN. Scientific source for this round is
the clean archive of `9cfb77727661cc64e47938d9d0ba65a157427439`; do not relabel
its measurements as a later master or a different installed library.

This is a bounded work plan, not a replacement authority for the issues'
acceptance criteria, generated catalogs or resource contracts. This evidence
change closes no issue, authorizes no release and relaxes no scientific tolerance.

## Evidence collected

- #762: [24 supplied-MO RCCSD observations and offline replay](../../../benchmarks/results/cpu-rccsd-native-closeout-20261104/README.md)
  support retaining the explicit native backend rather than promoting a slower
  default. Native solver gates: 10 passed; benchmark/retention suite: 80 passed.
- #2751: [actual RTX 5090 emitted-consumer counterexample](../../../benchmarks/results/occupied-xc-finite-input-20261104/README.md)
  returns dense rho=0, occupied rho=2, structural witness accepted, error=0.
  This is not a molecular calculation or a repair. Slurm job 15784 on node1;
  device visibility preserved and finite time enforced.
- #1118: the existing native catalog checker passes on the named source but
  reports **0 observed native contexts, 0 qualified products, stale=True**.
  Checker success means coherent fail-closed reporting, not fresh admission.
- #597: the existing inventory validator passes, retaining four constants:
  three migration-pending choices and the explicit 1024-task staging bound.
- The combined Direct tuning, residency receipt and Libxc capability portable
  tests pass: **51 passed**. They do not qualify production GPU residency.
- #2681: offline publication replay passes for the historical exact CPU slice
  at `bbd41a42764e8d22d15c1348f70a633961d4dcf5`, preserving its 13 native/65
  unit-test receipts. These tests were not rerun by that replay; DF and CUDA
  remain UNAVAILABLE in that capsule. Source PR #2944 was OPEN, conflicted
  (`mergeStateStatus=DIRTY`) and still had a CUDA CLI package check running.

## Remaining work and order

1. **Scientific correctness: #2751 and #2915.** For occupied XC, certify stored-D
   error/amplification or prove a fail-closed restriction, retaining original-D
   fallback and admission cost. For moved/fresh PBE, recover the original task,
   run paired parent/candidate binaries with the original grid and a separate
   tighter-SCF-only control, and retain the original 1e-7 force gate. Do not
   infer a common cause from the two different observations.
2. **CPU ownership: #934, transferred from #762.** Finish the exhaustive current-source production
   scientific inventory and classify genuine duplicates, glue, independent
   oracles and bounded/measured exceptions. Endpoint measurements alone are
   not this audit. Reuse the new negative default-promotion evidence; do not
   reopen redundant tracking or report the terminal consolidation as audit PASS.
3. **Libxc admission: #1118.** Refresh exact source/Graph/binary capability and
   admitted E/F/spin/domain/lifetime receipts through the existing catalog.
   Historical measured rows must not become current merely by changing a SHA.
   Do not reimplement completed children or mistake public-method=0 for an
   absence of structurally selectable methods.
4. **Direct policy: #597.** Move persistent-ERI admission into the resolved
   workload policy and topology/cache identity; make force/Fock angular-domain
   selection agree with arena capacities and compiled queue consumers. Preserve
   the no-profile fallback and qualify legal non-5090 targets. The explicit
   1024-task staging bound is not unclassified debt.
5. **Residency: #1629.** Exercise the existing recorder on the actual host-staged
   DF compatibility round trip, deliberate hot-region transfers and resident
   owners. Exact bytes/waits/roles and coverage or INCOMPLETE remain mandatory.
   The tiny #2751 probe is not production residency evidence.
6. **CBS: #2681.** Resolve the existing implementation PR's conflicts and checks,
   then measure only missing admissible conventional/correlation-DF/full-RI-JK
   CPU/CUDA cells with full original error vectors, refusal/lifetime outcomes
   and offline replay. A PR merge or frozen CPU replay does not fill the matrix.
7. **Public budgets: #2937.** Prove smaller host bounds from retained/serialized
   owners before changing conservative estimates, implement the supported public
   Direct/DF planning contract, and qualify actual E/F and negative lifetimes.
   Native-provider budgets are not equivalent to public planner support.
8. **Complete overhead: #2293.** Run source-matched profiled/unprofiled native
   and Array API H2/water on n1/n4, 10+ interleaved warm samples, separated cold
   JIT, exact work/SCF controls and memory/lifetime accounting. Retain H2 <=0.2 ms
   overhead and the water control; do not optimize from kernel-only timings.

## Environment limitations and rejected shortcuts

The clean archive has no installed native library. Consequently 72 compiled
CUDA-host resource tests fail at setup with the missing-library prerequisite;
they are not 72 scientific failures and are not passes. Build the actual named
source with a verified compiler cache before using those tests for #2937.
Do not borrow an unidentified stale library merely to make this setup green.

Nodes n1/n2/n4/n5 were reachable. Node3 is excluded for GPU work; it was used
only for CPU execution. Keep all actual GPU endpoints/profilers/sanitizers
inside finite-time Slurm allocations and preserve assigned device visibility.

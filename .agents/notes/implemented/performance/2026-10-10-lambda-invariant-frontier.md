# Decision: retain the exact Lambda invariant frontier, not its entire graph

Status: implemented and source-matched qualified in this change
Date: 2026-10-10

## Problem

After mixed-f force-domain scheduling, cold ethane Lambda remains approximately
87.74 seconds. The retained source-matched diagnostic has 61.59 seconds of
GMRES wall and 50.60 seconds of GMRES GEMM kernels. Its existing core reuse is
requested but refused: the native extra retained-core arena requires
1,710,877,464 bytes at nine occupied and 221 virtual orbitals. Increasing the
budget, reducing Q32, changing residual cadence or removing independent audits
is not an acceptable substitute for reducing this storage demand.

## Decision

The same dependency/purity proof identifies 103 immutable core operations, but
only 40 values feed dynamic nodes or public graph outputs. Execute all 103
operations during preparation; persist only those 40 frontier values. Their
exclusive arena is 396,517,056 bytes. Preparation and dynamic temporaries need
578,431,456 bytes and may borrow the already-required matrix core scratch,
which is 712,793,032 bytes for this workload. Symbolic slot-shape containment
proves that reuse does not depend on coincident numerical dimensions.

The runtime must still explicitly compare required transient scratch with its
actually selected matrix arena. Missing capacity refuses only optional core
retention, never Q32, independent audit/replay or mandatory matrix execution.
Both arenas belong to one immutable Problem/T owner and one stream; changing
primal inputs ends the epoch. Prepared contractions bind the two disjoint pools
consistently in both phases. Submission/nonfinite errors retain sticky audit
semantics; partial outputs must not justify a retry.

## Invariants and rejected shortcuts

- Same FP64 adjoint graph, packed-space metric, parameter/factor VJP and physical residuals.
- All preparation operations remain, even when their outputs are only transient.
- Persist every invariant-to-dynamic edge and every invariant published output.
- Existing conventional CCSD reuse and other default emitters remain unchanged.
- No new paired/factored adjoint equation is needed for this first storage step.
- No claim of CUDA correctness, admission or speedup follows from a CPU sizing proof.

## Evidence

CPU complete-versus-boundary graph probes pass for occupied/virtual dimensions
(1,3), (2,3), (3,2), (3,4), with zero, ordinary, tiny and negative unrelated
adjoint seeds. Symbolic transient slot counters are contained by the original
core matrix arena. Local evidence is in
`/home/jzzeng/codes/qc-branch-audit-20260922/evidence-lambda-next-20261010/`.

The qualified baseline is merged master
`7685c251ded6c986b3cb63336e3c1cd31c0cd739`. The measured candidate is that exact
base plus the retained **dirty** reconstruction patch, not a subsequently
invented clean SHA. Both builds use separate initially empty source/build trees,
Release/sm120/portable_cuda AOT, FAST_COMPILE OFF and verified/reused ccache.
Dispatch object references prove both execute the merged mixed-f force path.

Three RTX 5090 hardware-stratified cold ABBA cohorts, Slurm n1 jobs 7215/7216
and n5 job 1461, retain twelve fresh complete endpoints, six per side. They all
pass independent energy, qualified 24-force, four retained coordinate/step FD,
translation, replay, exact Lambda/Z and stationarity gates. Every cohort improves
and the complete saving exceeds three times the combined MADs.

| Complete native endpoint / phase | Baseline median s | Candidate median s |
| --- | ---: | ---: |
| Cold ethane CCSD(T) E+F | 327.989404 | 318.739047 |
| Lambda | 88.312413 | 78.998265 |
| Orbital two-electron derivatives | 67.481965 | 67.519774 |

This is a descriptive scoped 2.8203% complete reduction and 10.5468% Lambda
reduction, not a universal profitability or formal promotion envelope with
complete iteration histories. Q32, cadence 30, independent audit, fresh primal
replay, 21 Lambda iterations, 22 physical solver actions, CCSD/replay work and
both physical force passes remain unchanged. One preparation and 23 reused core
actions replace 23 whole-core evaluations. Each of the 22 skipped preparations
independently predicts 103 scheduled operations, 33 GEMMs, 34,875,793,292
contraction terms, 34,851,307,725 GEMM summands and 777,609,024 logical packing
bytes; all five measured reductions match exactly. These are not hardware FLOPs
or measured DRAM traffic.

Fresh n4 Slurm job 726 passes 49 real-GPU cache/budget/owner/audit/provider tests
in 109.94 seconds, and seven bounded native cache/budget/owner tests under
memcheck in 94.24 seconds with zero errors/leaks. This is not full-ethane
memcheck. A separate complete node-level profile verifies all 70 immutable
non-GEMM nodes launch once and all 216 dynamic non-GEMM nodes launch 23 times.
It also verifies the five mixed-f class domains and residual f consumer launch
twice each after retention admission. Its GMRES wall is 52.6074 seconds and
GEMM kernel sum 41.0003 seconds; this diagnostic is never pooled with cold
timings. Categorize CUTLASS using demangled kernel names, not the generic
`Kernel2` short name. Near-boundary kernels and approximate clock alignment
remain explicitly recorded; node counts use the complete physical trace.

Host validation includes 28 focused frontier/conventional-reuse/default tests,
48 adjacent tests with 35 GPU-only skips, plus publication/oracle recomputation.
The official selected evidence is
`benchmarks/results/lambda-frontier-20261010/`. It preserves the source patch,
actual source/binary/toolchain identities, quantitative errors, raw measurements,
predicted-work oracle, separate profile and bounded sanitizer scope.

## Rejected approaches and failed qualification

Increasing the budget, reducing Q32, dropping audits/replay, or adding another
dynamic arena would trade away stronger existing paths rather than reduce
storage. A new paired/factored adjoint equation is unnecessary for this step.
An incomplete frontier is also rejected by the generic emitter: a storage hint
cannot discard an invariant-to-dynamic edge or bypass the dependency proof.
`StagedCudaState` borrows `response_arena`; it has no generic `arena` member.
The first prototype's compile failure is retained rather than called a pass.

The force PR's initial integration reused one build directory while extracting
Git archives with older commit mtimes. Only its new unused class-domain TU
rebuilt, leaving an old dispatch object and a null-gain observation. That is a
failed source/build qualification, not a valid candidate performance result.
Original raw runs and the Lambda exploratory ABBA against those binaries remain
separate. The corrected isolated-build force campaign retains twelve complete
endpoints, 350.604799 → 328.097170 seconds, with identical semantic work and all
numerical/default/fallback/memcheck gates passing. Its clean source pair is
7f342546d887796e6a92a005ba033029ed73ce2a / 590e076c4d3345b797ed2b2babc0d1778fd793d5;
its production force bytes match the merged implementation, but unrelated later
master changes are not silently included in that timing claim.

The force qualification's first profile exceeded its finite 20-minute Slurm
allocation after the numerical and sanitizer gates had completed. Its partial
trace remains failed evidence. Job 727 completes the same-binary profile in a
separate finite allocation. Exact NVTX force windows retain a 12.704-microsecond
ancillary density-bound kernel straddling the scope boundary; no owned derivative
domain crosses. Rejecting that unrelated overlap as a missing force domain would
confuse asynchronous setup with physical ownership. The original frozen force
campaign and its old-base 6.5923% measurement are unchanged.

## Consequences and revisit conditions

This supersedes the whole-core storage layout, not the scientific equations,
purity proof, independent audit or owner epoch. The existing default requests
retention automatically; its old explicit disable and all resource fallbacks
remain. Persistent values stay owner-local, while transient storage cannot grow
the selected mandatory arena. Future graph changes must preserve every frontier
edge, checked native scratch containment, bitwise cached/uncached small gates,
same-shape epoch isolation and complete-endpoint acceptance.

Further Lambda work should use the fresh physical GEMM/Q/packing attribution,
not assume fewer summands or more storage guarantees a complete-endpoint win.
The f residual and independent audit remain separate measured consumers; avoid
trading away their admission to retain another cache.

## References

- Previous layout: `2026-10-09-df-lambda-core-invariant-reuse.md` in this directory.
- Current behavior: `docs/developer/df_ccsdt_gradient.md`.
- Merged force prerequisite: PR #2210, master `7685c251d`.

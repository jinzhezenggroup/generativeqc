# Decision: opt-in FP64 Lambda batching and original-graph fresh replay

Status: implemented, opt-in; no default performance promotion
Date: 2026-10-09

The source/defaults and measurements below describe the original opt-in epoch.
The later [VRAM-guarded default decision](2026-10-09-df-lambda-vram-defaults.md)
supersedes its default policy, not the historical source or numerical evidence.

## Problem

After #2136's core-invariant retention and independent matrix audit, the strict
W / cadence-thirty 230-AO endpoint still spent about 121 seconds in Lambda.
The qualified profile retained about 18 seconds of mandatory fresh primal replay
and many small Q-batched operator GEMMs. The remaining replay executed the
original virtual residual one auxiliary slice at a time, even when its later
independent Lambda audit had admitted a larger matrix arena and Q batch.

Batching and fresh replay need separate ablations: a faster operator can otherwise
receive credit for a replay change, or vice versa. A capacity-bounded batch is
not automatically faster and reserved capacity is not a measured VRAM peak.

## Decision

Keep Lambda's default Q batch at eight. Add a fresh-process experimental sweep
with a mandatory batch-eight baseline, and compare batch limits independently
from opt-in original-graph matrix replay. Keep CCSD batch eight, strict FP64 W,
cadence thirty, core retention, matrix audit and all scientific gates fixed.

`df_primal_matrix_gemm=false` remains the lower-level and complete-owner default.
The optional replay uses the original virtual residual program, whose only
inputs are `t1`, `t2`, `bov` and `bvv`. The existing generic matrix transform and
ordered batch accumulation consumer lower that graph; no handwritten residual,
derivative equation or cached accepted residual is substituted. The original
retained-core replay and energy/residual acceptance check still run.

Share the fresh replay matrix arena with the later independent matrix audit:
the two stages have disjoint lifetimes on the same owner stream. Charge the
maximum of their arenas, plus conservatively reserved prepared descriptors.
Replay currently requires both ordinary matrix and matrix-audit admission. If
the enlarged reservation is refused, retry the original audit-only arena and
retain scalar fresh replay. Dropping the audit arena or provider clears replay
admission. Preserve the existing CUDA allocation-shortage fallback order, sticky
finite checks and failure-without-publication behavior.

This deliberately does not make replay admission independently own an additional
arena. A future independent replay owner would need its own complete budget,
lifetime proof, bounded allocation fallback and matched endpoint evidence.

## Rejected alternatives

- Removing fresh replay or borrowing accepted CC residuals would weaken the
  independent primal acceptance boundary. Optimize the original graph instead.
- Reusing staged solver cuts or core-cache values for the independent audit
  would destroy its original-equation independence. Those remain excluded.
- Globally promoting batch 32 from one screen would confuse a shape-specific
  throughput/capacity tradeoff with a general production policy.
- Adding replay and audit arena sizes would charge storage that cannot be live
  concurrently; omitting either capacity would undercount actual admission.
- Raising evidence budgets or publishing external release backups is not an
  optimization. Retain compact local evidence under the unchanged 64 MiB cap.

## Invariants

- Strict FP64 operators, accumulation, W and original acceptance tolerances.
- Fresh original primal replay plus independent expanded Lambda audit.
- Identical Lambda iterations/actions across every timing comparison.
- Original Q order, including whole batches and partial tails.
- Owner-local immutable inputs, prepared bindings and explicit scalar fallbacks.
- Oracle energy/finite differences read only after production finishes.
- Finite Slurm allocation, assigned device visibility and no node3 GPU use.
- No pooling old-binary screens, clean timings or instrumented profiles.

## Evidence and source identity

The initial batch 8/16/32 screen used the previously qualified #2136 binary.
Its 518.321/494.727/483.359-second complete endpoints and
120.840/96.565/84.739-second Lambda samples were screening observations, not
repeated performance qualification. They are not pooled into the final cohort.

The follow-up source is pinned master
`d5a3173cc89b399ed105b0750124473eb782e2e5`, plus #2156's dependency diff through
`3f79e2a93d28c5ed1d9817871f365473e5710e17`, plus this follow-up. It is not the
original #2136 qualification source and must not be described as a continuously
moving latest-master build. The source-only reconstruction patch includes all
measured source/driver changes while excluding historical evidence/docs/notes.

The compact numerical publication is
`benchmarks/results/df-lambda-batch-replay-20261009/`. It separates eight clean
forward/reverse cold endpoints from four instrumented endpoints, all in Slurm
job 2767 on one frozen GPU/library/executable/input/driver cohort. All retain
21 Lambda iterations and 22 actions. Medians/ranges are descriptive, not
confidence intervals or the shared five-pair promotion gate.

Final clean median seconds are:

| Cell | Complete E+F | Lambda | Lambda capacity GiB |
| --- | ---: | ---: | ---: |
| Batch 8, scalar fresh replay | 516.922 | 120.465 | 9.390 |
| Batch 32, scalar fresh replay | 481.469 | 84.327 | 16.646 |
| Batch 8, matrix fresh replay | 507.606 | 110.613 | 9.913 |
| Batch 32, matrix fresh replay | 471.299 | 74.271 | 19.280 |

The combined candidate saves 8.826% complete endpoint / 38.346% Lambda time.
Separate parent-consistent profiles put b8/b32 GMRES at 74.085/49.282 seconds
and independent audit at 20.881/11.137 seconds. Scalar fresh replay remains
17.996/18.000 seconds; the replay flag reduces it to 8.261/8.085 seconds without
changing GMRES/audit time or action counts. This is a capacity tradeoff, not a
claim that the extra arena or packing is free, nor a physical DRAM/peak-VRAM
measurement. All twelve endpoints pass the independent original gates.

Native/library qualification passes 95 cases; ten whole/tail replay/batch
memchecks report zero errors. Fresh replay tests compare all response arrays
bit-exactly with scalar replay, with and without energy sources, including
audit-disabled fallback. Three independent NumPy tests compare the original
primal graph with packed whole/partial batches at the original tolerance.
The final executable received only a usage-text correction after the native
suite, passed all 39 early CLI tests, and was then frozen for the full cohort.
Its library is unchanged from the GPU/memcheck qualification.

An earlier screening journal failed after native execution because logging used
the wrong time-field name. A candidate test run also exposed an obsolete CLI
argument-count test. Neither incomplete attempt contributes a completed summary;
the logging and boundary tests were repaired without relaxing numerical gates.

## Consequences and revisit conditions

Batch 32 trades more reserved memory for fewer small GEMMs and packing/batch
overheads. Original-graph matrix replay changes the fresh-replay lowering, not
the iterative operator or number of solver actions. These mechanisms require
separate phase/work/capacity reporting and explicit actual-admission fields.

The ordinary batch remains eight and fresh matrix replay remains disabled.
Revisit defaults only after the shared repeated complete-endpoint, compilation,
measured-memory and wider scientific-shape gates. Consider an independent replay
arena only if another consumer needs replay without matrix-audit admission.

## References

- [Current gradient composition](../../../../docs/developer/df_ccsdt_gradient.md)
- [Core retention rationale](2026-10-09-df-lambda-core-invariant-reuse.md)
- [Independent audit rationale](2026-10-09-df-lambda-independent-audit-gemm.md)
- [Compact matched evidence](../../../../benchmarks/results/df-lambda-batch-replay-20261009/README.md)

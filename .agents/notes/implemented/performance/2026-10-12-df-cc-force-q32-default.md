# Decision: use the bounded automatic primal Q32 cap for CUDA DF forces

Status: implemented
Date: 2026-10-12

## Problem

#2197 promoted automatic Q32 only for energy calls. Complete analytic-force
owners retained Q8 even when their correlation budget admitted Q32. An explicit
Q32 feasibility run would not qualify changing the actual automatic selector.

## Decision

Resolve zero/omitted `ccsd_batch_limit` to 32 for both complete energy and force
calls, before passing a positive cap to the solver. Keep positive overrides,
standalone solver defaults and Lambda/response tile policies unchanged. Existing
dimension, complete-owner budget and allocation fallbacks still determine the
actual admitted primal tile. Do not change precision, incoming-residual mixing,
physical replay, contraction equations or response owners.

## Evidence and source boundary

Frozen clean parent `34dd51ed3ebc5b03b17f095de2cd7b39903923c8`, with one
production-owner selector overlay. The candidate requests real automatic zero;
the control explicitly requests Q8. This isolates the change from incoming-
residual #2298 and batched physical-replay #2299. Timed endpoint binaries and
all generated scientific kernels are byte-identical across arms. A separately
rebuilt validation adapter exposes zero through a test-only cap selector so the
small independent-force suite also exercises the actual automatic policy.

Fresh cold ethane230/o9/v221/Q488 complete-force ABBA runs on node2's assigned
RTX PRO 6000 in finite Slurm 2902. Complete median wall decreases
275.668013 -> 257.361953 seconds, **6.6406%**. Both candidate samples are faster
than both controls. Exact process medians, all samples, scientific gates and
work receipts are retained in the referenced publication. This is
descriptive scoped evidence, not statistical/global performance or timing of
the latest whole library.

Actual admitted Q32 reduces CC GEMMs 54,862 -> 25,792 and Q tiles 2,806 -> 1,096.
Evaluations remain 38; semantic contraction summands 26,799,066,988,432 and
physical GEMM summands 26,308,647,544,608 remain unchanged. This is a physical
tiling/launch/data-movement improvement, not reduced scientific arithmetic.
Lambda actions remain 22 and orbital actions 13. Actual per-graph compiled
queries independently reconstruct every published CCSD work aggregate.

CCSD logical numeric capacity increases 4,392,162,937 -> 8,231,194,433 bytes:
**+3,839,031,496 bytes (~3.58 GiB)**. Its device arena increases
3,981,577,984 -> 7,820,543,744 bytes. Complete endpoint logical capacity remains
19,509,839,897 bytes because a later force phase already determines that maximum.
The unchanged complete maximum must not conceal the CC-stage memory increase;
physical peak VRAM is not measured. Caller-provided caps and bounded admission
remain essential for different workloads and budgets.

Three compiled host selector/default tests cover omitted/zero force and energy
caps, positive 1/3/8/16/32 overrides, argument positions and unrelated selectors.
All 38 independent complete-force/public cases and four large independent energy,
fresh residual, Lambda/Z and force-direction gates pass. Water memcheck reports
zero errors. The unchanged baseline's prior independent suite is reused.

## Alternatives and consequences

Keeping Q8 globally leaves a complete endpoint gain unused. Unconditionally
allocating Q32 would violate resource contracts; this change alters only the
requested cap and retains all admission and failure behavior. An explicit Q32
benchmark alone cannot qualify the actual auto selector, hence zero is passed
to the rebuilt candidate in every matched timing process.

## Revisit when

Requalify broader numerical/workload domains or a different memory policy with
independent force gates and complete endpoints. Reconsider the cap if a retained
case shows an endpoint regression or unacceptable resource tradeoff; do not
remove positive overrides or bounded allocation fallbacks.

## References

- #2197; independent force PRs #2298 and #2299.
- `benchmarks/results/df-cc-force-q32-default-20261012/publication.json`.
- `tests/python/test_df_lambda_default_policy.py` and force Q32 evidence tests.

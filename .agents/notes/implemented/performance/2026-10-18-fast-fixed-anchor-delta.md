# Decision: default fixed strict anchors for qualified Fast WB97M-V cold solves

Status: implemented
Date: 2026-10-18

## Problem

The isolated [anchor experiment](../../proposed/2026-10-18-wb97mv-anchor-delta-jk.md)
showed useful complete-endpoint work reduction for application-budget Fast on
water-96, but no convincing FP64 benefit. Promoting its environment-only patch
literally would also admit unqualified domains and make optional extra memory
a mandatory preparation failure.

## Decision

Default to fixed FP64 J/K/range-K anchors only for already-admitted relaxed
application-budget Fast WB97M-V restricted exact-Direct cold VV10 outer solves.
This does not enable the generic incremental controller or change its strict
precision eligibility. It does not admit new mixed arithmetic or modify the
initializer, frozen-potential schedule, convergence gates or force arithmetic.

The first full build and subsequent physical targets use full-density strict
FP64 linear operators. Provisional initializer/frozen-potential stages apply
the existing policy arithmetic to total current-minus-fixed-anchor density,
then add the anchor before nonlinear XC. Delta builds do not advance it.
Physical validation commits or invalidates the state; each new solve resets
anchor validity and same-owner warm replays do not use this cold optimization.

Allocate a separate optional lease after mandatory preparation: five FP64 AO
matrices and one scalar, charged to native state capacity. Device OOM retains
the ordinary Fast path; unrelated CUDA and host-bookkeeping failures propagate.
Resource-ledger owners, device chunks, UKS, DF, other models, FP64 policy and
strict caller controls remain outside this promotion. Explicit
`GENERATIVEQC_KS_FAST_ANCHOR_DELTA=0` rolls back; unset/`1` cannot force unsupported
domains. Successful work and fallback/invalidation counters remain internal to
the native owner without extending the public C transport ABI.

## Evidence

- [Frozen experiment](../../../../benchmarks/results/wb97mv-anchor-delta-20261018/README.md):
  water-96/768 AO, WB97M-V/def2-SVP, RTX 5090, complete cold energy plus analytic
  forces. Fast off/on was 580.152/461.533 seconds, 18/16 target Focks and 3/2
  VV10 sweeps. FP64 work counts were unchanged. These are single observations,
  not repeated statistical speed guarantees.
- Fast maximum-force error improved from 6.46e-6 to 3.19e-6 Eh/Bohr, but did not
  attain FP64 accuracy. Independent policy gates remain unchanged. Water-12 was
  3.42% slower in its single experimental Fast pair; no universal win is claimed.
- [Production qualification](../../../../benchmarks/results/fast-anchor-default-20261018/README.md)
  retains source-authenticated build, host admission, actual GPU owner lifecycle
  and independent molecular endpoint evidence for the promoted implementation.

## Rejected alternatives and invariants

- Do not promote FP64 or generic incremental J/K based on this Fast experiment.
- Do not chain previous-iterate deltas or use delta density in nonlinear XC/VV10.
- Do not interpret strict final arithmetic as FP64 convergence or observable error.
- Do not fail an otherwise admitted endpoint solely for an optional device lease.
- Do not silently expand to budget-ledger owners without extending their planner.

## Consequences and revisit conditions

At 768 AO the extra lease is 23,592,968 bytes (22.5 MiB plus eight bytes), with
O(N^2) differences, reconstruction and copies. It can change provisional
trajectories and need not improve every size. Revisit after repeated complete
endpoint measurements or independent qualification of additional domains;
preserve explicit rollback, final full-target audits and finite fallback bounds.

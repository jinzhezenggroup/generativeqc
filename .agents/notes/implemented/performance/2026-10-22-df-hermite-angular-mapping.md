# Decision: give low-order DF targets more source lanes

Status: implemented
Date: 2026-10-22

## Problem and decision

The native tensor-free DF contraction originally assigned eight source lanes
and sixteen target-component lanes to every angular class. An auxiliary s
target has just one Hermite component, so most component lanes wait while
the source lanes perform radial work. Select 32 source lanes for up to four
target components, 16 for up to ten, and eight otherwise. Keep 128 threads,
power-of-two subwarp reductions, sequential angular-class launches and one
writer per target interval. Raised derivative targets select from their actual
raised order. High-order targets retain the original shared-memory bound.

This changes summation order, not mathematical work, screening, precision,
quadrature, metric rank, convergence or the optional-owner admission policy.
Padding lanes participate in every barrier/shuffle. It is not evidence for
FP32 DF integrals or a reduced recurrence census.

## Evidence

The clean baseline combines master `7522994e2`, the qualified native DF
charge/force owner, default-grid Fast gradients and final-state-aware stopping
at `91800c93d`. The angular candidate changes only this kernel mapping.
Ignored `df-md-unified-v1` and `df-md-angular-v1` artifacts retain complete
source manifests, core/AOT packages, scripts and endpoint receipts, locally
and under `n1:/data/jzzeng/scf-performance-20261021/`.

All ten complete populations (cold plus three frozen warm calls each) pass
the unchanged independent README gates, including separately traced runs.
The clean r2SCAN-3c DF/96 Standard comparison is 83.209 -> 70.330 seconds
cold and 17.750 -> 16.064 median warm, both 15/1 iterations. Fast is
82.563 -> 68.252 cold and 16.767 -> 14.727 median warm. The resident PBE
control does not use this streamed owner and changes within ordinary timing
variation. These samples do not establish universal or reference leadership.

All 178 standalone libcint charge/projection/derivative populations pass with
three changed/zero-input replays each. All 16 compiled-core metric/J/complete
Coulomb-force populations pass, including capacity and truncated-rank fallbacks.
Integrated memcheck (16 cases) and racecheck (4 cases) report zero errors/hazards.
The first standalone attempt failed before GPU execution because snapshot
copying had not finished; its receipt remains separate from the successful run.
The candidate core SHA-256 is
`00a6d1c41647a4918944bdcb802255efa9d607b6cbd271856478ae78afedb888`.

## Revisit when

Further mapping changes require the same independent angular, geometry,
changed-input, force and sanitizer gates, plus complete endpoint evidence.
Avoid treating low-order efficiency as a reason to expand high-order shared
storage or to change the source screening/error contract.

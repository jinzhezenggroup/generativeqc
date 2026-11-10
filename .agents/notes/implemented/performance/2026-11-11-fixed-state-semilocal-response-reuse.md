# Decision: reuse fixed-state compiler-AD point response with a bounded fallback

Status: implemented
Date: 2026-11-11

## Problem

The asymmetric default-grid NiH3 experiment described in
`../../proposed/2026-11-11-ni-uks-hvp-first-integral-work.md` has a coupled
945-dimensional orbital-response problem. Each matrix-free action previously
re-evaluated the same semilocal scalar AD Hessian. Only the feature tangent
changes between actions at one stationary state; geometry, weights, primal
features and the same/cross-spin point derivatives are invariant.

An isolated spin-gap-conditioned trial completed the response solve after
46 actions in 635.076 seconds, but had not returned a complete molecular HVP
when this decision was prepared. That observation identifies reusable work;
it does not establish complete endpoint performance or Ni qualification.

The spin-gap experiment subsequently hits its 3,600-second endpoint timeout
without an HVP (reported peak RSS 2,081,064 KiB). A separate frozen checkout
with the compiler-only point panel records 46 response actions in 442.130459
seconds, then also reaches its 3,600-second endpoint timeout without an HVP
(reported peak RSS 2,258,664 KiB). The explicit timeout receipt is
`/tmp/gqc-ni-hvp-fxc-frozen-probe/timeout-receipt.json`. Do not turn those
overlapping phase observations into a paired endpoint speedup claim.

## Decision

The CPU `_CompositeGrid` owner can build its point-local feature Jacobian by
evaluating the existing compiler-AD program on explicit feature directions.
This preserves the exact production expression, branch policy, grid weights
and cross-spin coupling. Subsequent fixed-state density actions multiply the
retained panel and use the existing AO potential assembly. No independent XC
algebra, reference engine or displaced SCF enters production.

Reserve twice the panel bytes before construction, conservatively covering
candidate/retained overlap. Subtract the same reservation from both
`feature_budget_bytes` and `model.budget_bytes`; reducing only the feature
budget would let scalar AD overspend the lease. Keep the panel private to one
state/geometry owner, publish only a complete finite candidate and restore
both budgets on every failed build. Allocation/tile-capacity failure selects
the original bounded action. If retaining the panel displaces the minimum AO
feature workspace or a subsequent moving-geometry AD tile, evict it and
restore both leases before replaying the original action. Do not borrow a
panel across a consumer that may evict it: otherwise the old buffer can stay
live after its reservation is restored. Other scientific/runtime failures propagate
after restoring the reservation. Geometry and moving-weight partials do not
use this fixed-state cache.

The CPU composite UKS nuclear driver separately composes both spin gap blocks
into the existing diagonal right conditioner. Preserve alpha-then-beta,
occupied-major order, signed gaps and the common global relative cutoff.
Singular gaps decline conditioning for the entire coupled system, without
clamping denominators. The true-residual solve gate remains unchanged, and
diagnostics report the actual selected policy rather than an outer default.

## Rejected extensions

- A first scratch cache also attempted arbitrary unit-feature tangents through
  native PBE point providers. Two PBE0 public endpoint regressions returned a
  native numerical failure in density tails. Both scratch runs retained their
  failures: `/tmp/gqc-grid-fxc-reuse-probe.log` and
  `/tmp/gqc-grid-fxc-reuse-final-probe.log` (20 passed, 2 failed each).
  Native providers therefore keep their physical-direction contract; no
  density floors, finite-difference bridge or weakened gate repairs the cache.
- VV10 is nonlocal in the point index and cannot use this local panel.
- Do not persist a point panel by functional name, array shape, geometry hash
  alone or object identity. Those keys do not authenticate the complete state.
- Do not report response-only elapsed time as complete Hessian performance.

## Validation boundaries

`test_composite_point_kernel.py` exercises restricted/unrestricted actions,
cross-spin coupling, multistep differences of primal coefficients, capacity
fallback, late allocation/runtime failure, reservation recovery, distinct-state
ownership, nonfinite tangents and native/nonlocal exclusion.
`test_orbital_gap_preconditioner.py` exercises dense independently solved
coupled equations, empty spin blocks, signed/singular gaps and packing.
Complete independent moving-grid Hessian, public PBE0/B3LYP and lifecycle
regressions remain required; lightweight panel tests alone are insufficient.

The sealed-source regression run reports **72 passed**, two existing JUnit
warnings, in **789.67 seconds**. It includes both Cartesian/spherical coupled
UKS independent moving-grid Hessians, public PBE0/B3LYP results and failure
recovery, lifecycle tests, compiler grid derivatives, spin-gap dense solves
and point-panel resource/failure tests. Runtime/test source hashes are identical
before and after the run; the log, XML, time and hash receipts are under
`/tmp/gqc-response-reuse-sealed-*`. This is CPU light-element/regression
qualification, not an installed Ni Hessian or a molecular TS receipt.

After preserving the concurrent raw CUDA ECP admission fix, the integrated CPU
ECP capability/numerical CTest group passes **2/2** in **10.52 seconds** and the
integrated Python smoke group passes **144 tests**, with two expensive
raw-Hessian cases deselected, in **190.40 seconds**. Changed-file pre-commit
checks pass. These receipts are under `/tmp/gqc-response-reuse-integrated-*`.
The rebuilt native library has SHA256
`79076cfbed7f07176c46f935a6c3366bc017959b0763b5e265240afd4ac20533`;
the earlier sealed 72-test run used the distinct library SHA256
`2debc36c38f0ed570e5a175fd6da942b2b57cd9b33019839c4e8d934a5d72769`.
Do not restamp the original regression run as using the rebuilt binary.

The frozen Ni experiments remain separate scratch installations. Their
directional-first/shared-spin lowering is not implemented by this change.
No public capability, mixed-ECP numerical acceptance, molecular saddle or
installed Ni full-Hessian acceptance follows from this decision.

# Decision: retain strict-anchor delta-J/K as a bounded Fast experiment

Status: proposed
Date: 2026-10-18

The guarded Fast-only production decision superseding this experiment is in
[the implemented note](../implemented/performance/2026-10-18-fast-fixed-anchor-delta.md).
The frozen experiment and its FP64 results remain historical evidence.

## Problem

The user asked whether applying J/K to delta density improves Fast accuracy,
reduces SCF iterations, and accelerates both FP64 and Fast on water-96. A
fixed-density probe had shown that a strict FP64 anchor plus mixed delta J/K
reduces arithmetic error as the delta shrinks. That observation alone says
nothing about nonlinear XC, final forces, actual iterates or endpoint speed.

The working checkout is older and dirty. The actual application-budget Fast
facade must not be confused with ordinary `Calculator(precision="auto")`.
Freeze `5d1dbbcaf2f04fb83980c8f988795df28b41e392` in an isolated source
archive instead of mixing current upstream revisions or editing user work.

## Experimental decision

Use separate experiment state in archived `src/dft/cuda_ks.cpp`, selected by
`GENERATIVEQC_EXPERIMENT_ANCHOR_DELTA=0|1`; do not promote it into defaults.
Restricted exact-Direct nonlocal KS alone is admitted, without DF, an active
public resource ledger or the generic incremental controller.

The initial full seed and every full target build strict full-density
J/K/range-K anchors. Provisional semilocal/frozen-VV10 iterations build on
the total current-minus-fixed-anchor density in ordinary policy arithmetic,
then add the strict anchor before nonlinear XC. Delta iterations never
advance the anchor. Failed iterations invalidate it. Every full physical
target remains fresh and full-density; policy acceptance gates are unchanged.

Keep the existing optional source residency, precision-specific source
identity checks and bounded nonresident fallbacks. Add arena storage through
the existing partition accounting, not hidden allocations. Require observed
delta and full-target events so an alternate execution route cannot silently
turn this into a comparison of two labels running the same algorithm.

## Rejected shortcuts

- Removing the generic mixed/incremental guard is not sufficient. Activating
  that controller also changes nonlocal initialization and residency behavior,
  confounding the intended delta-only schedule comparison.
- A mixed anchor retains its full-density arithmetic error. Successively
  advancing approximate anchors also chains omitted screening/roundoff work.
- Computing XC or VV10 on delta density changes nonlinear science; only the
  linear J/K components admit this reconstruction.
- Looser final checks, oracle densities, a different grid, disabled FP64
  reuse, kernel-only timers or first-use subtraction cannot establish the
  requested complete-endpoint win.

## Evidence

Finite Slurm jobs on node4's RTX 5090, water-96/def2-SVP, 768 spherical AOs,
Direct WB97M-V/RKS, grid 48/16/32, native Modified Huckel seed:

| Policy | Delta off/on cold E+F (s) | SCF Focks off/on | VV10 pair sweeps off/on |
| --- | ---: | ---: | ---: |
| FP64 | 996.437 / 982.914 | 28 / 28 | 4 / 4 |
| Fast | 580.152 / 461.533 | 18 / 16 | 3 / 2 |

The four water-96 endpoints share allocation 1133. Fast shows a 20.45%
observed endpoint reduction and explicit semantic work reduction; FP64's
1.36% single-pair difference with identical Fock/sweep counts is not an established win.
These are one observation per arm, not repeated statistical guarantees.

All arms pass independent retained GPU4PySCF geometry/basis/grid gates, with
no imported seeds, discarded attempts, retries or native fallbacks. The
water-96 reference is the completed geometry-zero accuracy receipt, not the
timed-out reference population endpoint, in the frozen README summary.
Fast maximum-force error improves from 6.46e-6 to 3.19e-6 Eh/Bohr, but remains
above the strict independent FP64 gate of 1e-7. FP64+delta has maximum-force
error 2.31e-11 and physical residual RMS 2.32e-13; Fast+delta has residual
RMS 1.77e-7. Strict final arithmetic is not an FP64 convergence guarantee.

Fast+delta executes four strict J applications and twelve mixed ones, versus
four strict and fourteen mixed in the control. Its mixed-stage Fock count
does not describe every operator's arithmetic: two provisional Focks contain
strict anchor J/K. The saved strict refinement iterations offset anchor
replacements here. Avoid claiming an increased strict-J total for water-96.

Four preliminary water-12 endpoints qualify the implementation. There Fast
stays at seventeen SCF Focks, strict J applications increase from two to four,
and the single complete delta endpoint is 3.42% slower. The unusually slow
first FP64-off qualification is retained without subtraction and is not a
small-system speed claim. The benefit is not size-independent.

## Consequences and revisit conditions

The 768-AO owner needs five extra FP64 matrices plus one scalar: 22.5 MiB
plus eight bytes. Copies, differences and reconstruction add O(N^2) work;
strict anchors can cost more per provisional call. Peak whole-endpoint
memory and isolated anchor-overhead percentages are not measured.

Do not promote from one molecule and one ordered pair. Revisit after
reverse-order/repeated complete endpoints, more difficult hybrid/nonlocal
fixtures, explicit resource-budget admission and fallback tests, execution
schedule coverage, and independent force/precision gates. If FP64 work does
not decrease, do not justify its enablement using Fast's result.

## References

- `benchmarks/results/wb97mv-anchor-delta-20261018/README.md`
- `benchmarks/results/wb97mv-anchor-delta-20261018/evidence.json`
- `benchmarks/results/wb97mv-anchor-delta-20261018/experiment.patch`
- Raw local/remote bundle: `.artifacts/scf-delta96-20261018/` and
  `/data/jzzeng/scf-delta96-20261018/` on n1/n4.

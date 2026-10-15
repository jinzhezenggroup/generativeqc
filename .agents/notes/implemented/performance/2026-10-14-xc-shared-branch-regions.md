# Decision: share lazy XC branches across derivative outputs

Status: implemented; endpoint qualification recorded with the benchmark evidence
Date: 2026-10-14

## Problem

The master README's complete energy/force endpoints expose slow large-system
M06-2X and r2SCAN-3c calculations. A separate 96-atom M06-2X DF Nsight diagnostic
on the retained October 13 source attributes 22.4% of device kernel time to
SCF equal-spin points and 14.8% to stationary geometry points. Both consume
the same scalar expressions. The r2SCAN-3c direct profile is instead dominated
by integral derivatives; point lowering alone cannot close that entire gap.

Scalar AD produces energy and partials guarded by identical predicates. The
incumbent emitter gives each select a separate lexical scope, repeating primal
work in each output. Structural CSE cannot reuse definitions after that scope
ends. Nested occurrences of an already decided predicate repeat work again.

## Decision

Add an explicit multi-output branch schedule to the shared scalar C emitter.
Discover the simultaneously required select frontier without traversing unknown
arms. Group equal predicates on that frontier, emit their required values
together in each arm, and carry the predicate fact only into that lexical scope.
Reuse the existing arithmetic emitter and graph nodes. Enable the schedule in
curated semilocal and split-hybrid equal-spin point emitters. Preserve the
incumbent general-point lowering and the existing equal-spin opt-out.

No new scientific formula, reassociation, precision, threshold, quadrature,
SCF stopping rule, or method-specific runtime admission is introduced. The
producer's existing equality witness remains required. Default scalar emission
for unrelated consumers is unchanged.

## Rejected alternatives

Hoisting common expressions out of unknown branches can evaluate singular
inactive expressions. Replacing spin-dependent algebra by approximate equality
would violate the producer contract. Tuning block sizes alone would retain the
repeated scalar work. A point-kernel speedup does not establish an endpoint win.

## Invariants and qualification

Executable strict C++ comparisons preserve every scalar output bit, lazy
singular-arm behavior and floating exceptions; nested predicate dependencies,
NaNs, signed zero and emitter scope reuse are covered. Actual generated
equal-spin bodies remain compared to their general siblings and independent
Libxc. GPU qualification retains complete cold, frozen-seed warm, moved and
moved-warm energy/force observations, independent error gates and semantic work.

Revisit this schedule if a new expression family causes register pressure or
code-size regressions. Retain the general lowering for controlled comparisons;
do not change a scientific tolerance to rescue an optimization.

## Measured endpoint evidence

The [paired campaign](../../../../benchmarks/results/xc-shared-branches-20261014/README.md)
binds master `0e6dfb453` and implementation `786c31a48` to independently rebuilt
Release sm_120 target-specific AOT libraries. At 96 atoms, complete M06-2X
direct cold/warm changes from 149.912/20.193 s to 123.065/17.347 s; DF changes
from 149.425/36.209 s to 122.755/29.699 s. Complete r2SCAN-3c DF changes from
213.073/21.591 s to 197.695/20.441 s. All 12 calls per arm pass unchanged
independent E/F gates, and each of these comparisons has identical SCF work.
The campaign retains moved/moved-warm phases, every repeat and small PBE0
controls. Direct r2SCAN-3c/96 has differing cold SCF work (32 → 30 builds) and
a slower moved endpoint (296.815 → 403.783 s; 14 → 23 builds). At candidate
moved iteration 14 the physical residual is already 1.31e-13, while energy
changes remain above the unchanged 1e-12 threshold until iteration 23. Preserve
this negative endpoint observation, full SCF histories, timeouts and incomplete
populations; do not normalize iterations or loosen convergence to rescue a
timing. A strict local scalar optimization does not establish identical SCF
trajectories or an all-phase direct endpoint improvement.
Both direct/96 populations hit the unchanged 1800-second limit, returning
11 baseline and 10 candidate calls. Their complete five-call warm medians are
103.830 and 100.647 s (one Fock build each), but neither has a complete
moved-warm series. All 237 returned calls across the paired campaign pass
1e-8 Eh / 1e-7 Eh/Bohr gates; maximum errors are 3.956e-11 / 1.573e-10.

Standalone point improvements (about 2.1× M06-2X and 1.8× r2SCAN at 256 points)
are diagnostic. The M06 probe reduces registers from 255 to 192 and removes
spills. Both probes preserve all eight output lanes bitwise and pass partial-block
memcheck. Molecular endpoints remain governed by independent numerical gates;
their atomics and converged SCF trajectories are not claimed to be bitwise equal.

The broader GPU qualification passes 14 tests after repairing two stale test
fixtures and packaging the UKS force profile. Moved replays must supply moved
coordinates explicitly, and historical auxiliary basis fixtures require the
checked namespace adapter. These repairs preserve the native implementation and
all acceptance tolerances. Initial failures and successful reruns are retained.

## Promotion decision — 2026-10-15

The paired campaign above is **historical evidence** for original source
`786c31a48`, which enabled branch coalescing in both split-hybrid and
semilocal equal-spin points. It is not a source-matched benchmark of the
current-master integration (which includes merged #2505).

This integration keeps the general lazy multi-output scalar emitter but
promotes branch sharing **only for FP64 split-hybrid equal-spin codegen**.
Semilocal (including r2SCAN-3c direct/DF) retains current-master lowering
without new branch sharing. FP32 retains the established lowering. The
#2505 qualified force-only integer-power policy is preserved. This is a
compiler scheduling boundary, not a change to any functional, tolerance,
grid or convergence criterion. The historical direct r2SCAN-3c moved
296.815 -> 403.783 s observation, unequal SCF work and timeouts are
retained rather than assumed fixed.

Before merge, perform a source-matched current-master-versus-this-branch
M06-2X complete E+F campaign (direct/DF, cold/five warm/moved/five
moved-warm), retaining SCF work and independent E/F gates. A new benchmark
is required before claiming combined #2464 + #2505 speedups.

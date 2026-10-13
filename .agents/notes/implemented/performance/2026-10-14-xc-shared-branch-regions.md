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

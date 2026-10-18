# Decision: retain opt-in angular-three reciprocal MD-J reuse

Status: implemented as an opt-in experiment, not a default promotion
Date: 2026-10-18

## Problem

The [contraction-priority investigation](../../proposed/2026-10-18-md-j-residual-contraction-priority.md)
falsifies candidate count alone as the explanation of residual cost. The user
requests an actual implementation/timing experiment rather than another source
comparison. Both admitted residual contraction and duplicated higher-order
uniform radial sources are plausible bounded targets.

## Decision

Two residual experiments are rejected: register accumulation/fixed Hermite-loop
expansion is effectively neutral, and task-local density contraction before
atomic J publication is slower by about 2.8% on the common accepted density.
The latter keeps every AO-oriented mask and reads the incumbent integral
triangle for identical shell pairs, but loses enough other execution efficiency
to outweigh reduced publication. Do not retain either production implementation
or infer an achieved-occupancy/bandwidth/atomic bottleneck from these timings.
This is not an implementation or rejection of masked Hermite density itself.

The retained experiment extends the existing generated reciprocal source to
uniform pair-angular `(0,3)`, `(1,3)`, `(2,3)` and `(3,3)` classes. It does not
introduce new equations, radial recurrences, global storage, CPU work or a new
tile-size sweep. The existing tile's bounded shared scratch is sufficient for
these classes. Angular-four work keeps its oriented provider.

`GENERATIVEQC_MD_J_RECIPROCAL_HIGH=1` opts in at resident-owner preparation.
Unset/0 keeps the incumbent path; parent `GENERATIVEQC_MD_J_RECIPROCAL=0`
disables the extension as well. Invalid values reject active MD preparation.
Neither environment variable is polled during replay. Optional-storage failure
and unsupported basis admission still select normal J; K and force/response
ownership do not change. The new option stays off by default.

## Invariants

Each density direction retains its own conservative admission predicate and
aggregate error allowance. Partially screened quartets retain the exact public
AO orientation masks. FP64, nonsymmetric total-density behavior and independent
RKS/UKS numerical gates are unchanged. Uniform density-consumer/summand counts
and residual task/product counts are conserved, although fewer uniform radial
sources are evaluated. The default-off regrouping of class launches preserves
the per-output accumulation order because different bra-angular classes own
disjoint oriented-potential outputs.

The seven-event profile ABI retains its historical intervals. The opt-in
angular-three kernels run in the higher-order interval named `oriented_seconds`;
kernel-name timeline classification is required to distinguish the extension
from the remaining oriented work. Do not silently compare that field as if its
kernel inventory were identical across selectors.

## Evidence and limits

The [source-matched trial record](../../../../benchmarks/results/md-j-residual-20261018/contraction-trials.md)
retains identities, Slurm jobs, independent gates and rejected measurements.
Two fresh processes per mode, four clean prepared-resident J replays each, give
4.121193 seconds off and 3.923415 seconds on: about 4.8% lower J time. Uniform
radial evaluations decrease from 2,225,962,468 to 2,038,308,216 while every density
consumer and residual primitive product is preserved. GPU4PySCF remains much
faster and has a different screening contract. No hardware-counter attribution
or complete-endpoint speedup follows from this prepared-J observation alone.

Independent scientific tests, failure/ownership host probes and small isolated
CUDA sanitizers pass. Full energy/force timing and actual SCF work are recorded
separately: the completed off/on pair preserves 24/1/13/1 Fock builds and all
independent gates, lowers complete cold 270.418484 to 264.922447 seconds, and
lowers the four-phase sum 557.010250 to 547.387625 seconds (1.73%). A single pair
cannot complete the existing draft PR's broader default-promotion qualification.
Raw experiments remain ignored local evidence rather than new tracked run
bundles or external publications.

## Revisit when

Consider promotion only after repeated clean complete energy/force endpoints,
actual Fock counts and independent numerical gates justify it. Revisit residual
masked-density representation or cooperative contraction with a genuinely
different data/work representation, not just the rejected publication rewrite.

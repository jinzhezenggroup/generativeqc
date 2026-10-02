# Decision: bound independent coupled-cluster verification work

Status: implemented
Date: 2026-10-02

## Problem

The native independent expanded CCSD replay and Lambda transpose deliberately
preserved primitive reduction order. They consequently retained degree-eight
contractions even though the shared iteration/operator used bounded-degree
contractions. In the 28-AO STO-3G water cluster (20 occupied, 8 virtual), expanded
replay consumed about 28.4 s of a 40 s warm CPU CCSD(T) energy endpoint. Merely
reclaiming dead scratch did not remove this work.

## Decision

Emit both the original strict graph and a reassociated graph from the same
separately expanded equations, using the existing TensorIR contraction lowering.
Execute the reassociated graph in output-driven depth-first dependency order.
Program logical serialization and equation identity keep their canonical order;
this execution order evaluates shared definitions once and excludes dead nodes.
All outputs remain pinned through return. The schedule metadata is provenance,
not a change to mathematical identity.

Select reassociation separately for physical replay, independent Lambda RHS and
independent Lambda transpose only when its exact checked arena is no larger than
the original schedule's arena at the runtime occupied/virtual dimensions. Retain
the original strict schedule for extreme ratios and optional-intermediate size
overflow. The CPU header owns the selector used by both CPU and CUDA admission
and execution. Replay variants use the replay arena, and Lambda variants use the
response arena. Lambda diagnostic hashes identify the actual selected graph;
the physical replay equation hash continues to identify the expanded equations.

## Invariants and acceptance

- Preserve the separately expanded verification equations; do not substitute the
  shared iteration residual or remove an independent gate.
- Do not relax convergence, independent residual, response or oracle tolerances.
- Keep CPU/CUDA graph and slot-plan equality, ordered-stream lifetimes, checked
  arithmetic, and the prior strict arena ceiling at every shape.
- Reassociation changes floating-point reductions. Compare all outputs against
  unlowered expanded TensorIR and public endpoints against independent PySCF and
  energy finite differences before qualification.
- Retain strict fallback even if common balanced shapes favor reassociation.
- This change leaves the public <=12-AO analytic-force qualification boundary,
  conventional MO preparation and dense response weights in place.

## Evidence

Compiled tests execute strict, reassociated and selected variants against
unlowered expanded TensorIR at (o,v)=(1,2),(2,3),(3,2),(20,1). They cover every
replay/RHS/transpose output with a 2e-11 mixed tolerance, exact admitted capacity,
both allocation canaries and rejection with one fewer element. Selection tests
include (20,8), (20,80), extreme ratios and optional o^4 overflow at (100000,1).
Compiler analysis bounds reassociated replay and transpose contraction degree at
six versus eight for the strict graphs. Production graph liveness tests require
CPU/CUDA schedule equality and exclude overlapping live intervals.

At o=20,v=8 the replay arenas are 21,748,528 B strict and 8,959,912 B reassociated;
independent Lambda transpose is 26,599,680 B versus 3,213,952 B. These are generated
arena capacities, not whole-process memory peaks.

## Rejected alternatives

Removing physical replay would hide a failure instead of optimizing it. Reusing
the shared residual would weaken its algebraic independence. Reassociation with
the canonical breadth/depth-level serialization order increased scratch because
many branches remained simultaneously live. Rewriting sums into add chains added
an unnecessary algebra change and did not resolve that retention under the old
order. A depth-first schedule addresses branch retention without rewriting sums.
Always selecting reassociation would increase admission at extreme o/v ratios;
the explicit strict fallback avoids that regression.

## Consequences and revisit conditions

Generated code includes both schedules. Larger conventional MO preparation,
triples and force-response work remain separate bottlenecks. Revisit the fallback
when a shape-aware contraction planner can prove both work and capacity benefits
without losing independent verification or admission guarantees. Preserve the
full endpoint, oracle, resource and device qualification when changing schedules.

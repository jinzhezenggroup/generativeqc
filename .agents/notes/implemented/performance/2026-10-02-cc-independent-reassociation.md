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

The exact emitted representative replay graph, evaluated at o=20,v=8,
reduces the compiler arithmetic-work estimate from 105,782,717,764 to
1,003,074,884. Independent Lambda transpose decreases from 266,224,876,320 to
2,176,717,600. These evaluate the emitted schedule, not a separately optimized
graph rebuilt at the benchmark shape. Endpoint iteration/replay calls and MO
work counters remain identical; internal replay contraction work decreases.

The [CPU evidence](../../../../benchmarks/results/ccsdt-replay-20261002/cpu-summary.json)
records a single-thread comparison pinned to CPU 45 on the same workstation:

| Endpoint | Strict | Selected |
| --- | ---: | ---: |
| 28-AO energy, cold | 139.304 s | 109.885 s |
| 28-AO energy, warm mean (2 samples) | 39.844 s | 13.166 s |
| 28-AO energy, changed geometry | 139.030 s | 109.928 s |
| 14-AO energy, warm mean (2 samples) | 0.301 s | 0.197 s |
| 7-AO forces, warm mean (2 samples) | 2.578 s | 2.561 s |

Replay alone decreases from about 28.4 s to 0.577 s in the 28-AO case. The 3.03x
warm and 1.27x cold energy endpoint improvements do not imply a force speedup;
the force change is within timing variation. All energy/force outputs are
bitwise equal to the strict baseline. Matched PySCF 2.14.0 errors are at most
8.7e-13 Eh energy, 1.5e-14 Eh triples and 1.3e-8 Eh/bohr force. Independent public
PySCF gradients, three-step energy finite differences, exact-budget tests and
nested allocation probes pass. The shared workstation and small sample count
limit precision of timing ratios.

The [CUDA evidence](../../../../benchmarks/results/ccsdt-replay-20261002/cuda-summary.json)
records node1 Slurm job 5321 (`main`, `gpu:5090:1`, 20-minute limit): all 37 public
CPU/CUDA tests passed, and the complete 7-AO force endpoint passed memcheck with
zero errors. Cold, twice-warm and changed-geometry 14/28-AO energy results pass
the same PySCF gates. This does not demonstrate a GPU endpoint speedup: the
28-AO warm endpoint remains about 65 s, dominated by reference and MO source
preparation. The node was shared; sanitizer timing is instrumented.

Reproduce complete endpoints with `benchmarks/ccsdt_prepared_endpoint.py`, setting
`GENERATIVEQC_LIBRARY`, `PYTHONPATH=python:.`, and `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to 1. Use `--atoms 3 --forces` or
`--atoms 6`/`--atoms 12`, plus `--output <ignored-artifact.json>`. GPU runs add
`--device cuda` and must use the local Slurm allocation contract. Both summaries
retain the exact basis digest, geometry/settings context and loaded-library
hash; keep cold, warm, repeat and moved rows together when comparing revisions.

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

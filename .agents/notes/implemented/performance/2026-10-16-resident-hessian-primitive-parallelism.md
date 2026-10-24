# Decision: parallel primitive AD with ordered contraction

Status: implemented; selected-closure device qualification complete
Date: 2026-10-16

## Problem

The Python-free native endpoint removed interpreter/JIT dependence, but that
alone did not make its integral derivative schedule practical. The matched
single-GPU first-observed water Hessian took 557.0842585428618 s at source
`7c2a0b95a81f6db51d117dd9810ecb356e93d050`. Its inter-engine numerical gate
failed, so neither this latency nor the lower H2 latency qualifies a speed ratio.
The complete records are retained in the shared native H2 publication.

Slurm 3176 profiled one complete water HVP on the same binary, with native
interpreter/compiler discovery unavailable. Nsight Systems recorded
62.222819175 s of GPU kernel execution, including 60.833134650 s (97.77%)
in 5,168 AOT ERI derivative launches. This is diagnostic HVP evidence, not an
unprofiled Hessian speed measurement. The HVP completed one direction, ten
induced Fock actions and 777,924 exact primitive-component evaluations.

The old emitter put at most sixteen components in one warp and evaluated every
primitive record serially inside each participating thread. Even the ssss
full-range kernel averaged 3.73 ms across 1,024 launches. Launch batching alone
would leave that serial primitive/quadrature work intact.

## Decision

Keep the existing compiler scalar-AD graphs and native integral ABI. Emit one
64-thread block per selected raw component, evaluating independent primitive
records concurrently. Each lane writes its scaled value/gradient/gradient-JVP
fields to bounded shared storage. Lane zero performs the same FP64 additions
in input-record order, including across successive batches. Thus no derivative
rule, recurrence, quadrature, normalization, primitive work count, or contraction
order changes. There is no floating-point atomic or tree reduction.

The common emitter serves one-electron, ERI and raw DF second-order programs.
Its largest block needs 25 × 64 doubles (12,800 bytes) of shared memory and
no additional global allocation. All lanes, including an inactive tail, reach
the failure barrier. An invalid primitive rejects the entire component without
divergent-barrier deadlock; the native owner still gates whole-result publication.

## Rejected alternatives

- Do not replace full/SR quadrature or tune precision to improve this benchmark.
  Those are separate numerical changes needing their own independent gates.
- Do not tree-reduce primitives just because they evaluate concurrently. Severe
  signed contractions can change cancellation under reassociation.
- Do not infer a speedup from occupancy or profiled kernel time. Complete
  unprofiled endpoints and actual semantic counts remain the acceptance boundary.
- Do not equate a faster selected package with complete orbital-f/auxiliary-g
  compiled closure or migration of separately exposed Python numerical consumers.

## Validation

The native AOT tests compare all development integral families and component
tiles to independent quadrature. Additional runtime tests compare contractions
of 1/31/32/63/64/65/129 records bit-for-bit with ordered single-record fields,
including severe signed cancellation, partial blocks and multi-batch lists.
Late invalid records before/at/after the batch boundary must fail without output
or deadlock. Independent molecular energy, CLI and sanitizer reruns are required
before promoting this scheduling change.

The first native reruns completed 52 AOT tests (Slurm 3181, 62.85 s), including
all the ordered-contraction and late-failure controls, and both actual CLI
oracle cases (Slurm 3179, 70.83 s). Two host native contracts passed in 29.47 s;
the local compiler/queued-failure gate passed 25 cases with 52 explicit GPU
skips. Compiler structure checks covered 563 modules with zero dependency
errors; all repository hooks passed. Slurm 3185 completed the 21 molecular
energy-oracle cases: all 16 spin/backend/layout/ECP combinations, four
independent spherical-d auxiliary cases, and the exact H2 benchmark full matrix.
Four disjoint pytest subprocesses shared the single Slurm-visible GPU; all
passed (6/5/5/5 cases). The earlier serial Slurm 3180 was canceled, not counted
as a passing receipt. Slurm 3182 completed synccheck, racecheck, initcheck and
memcheck on accepted UKS/DF/ECP with the independent spherical-d auxiliary:
zero errors, and zero racecheck hazards or warnings.

Slurm 3183 additionally profiles the same complete water HVP using the parallel
schedule. Its diagnostic internal endpoint time is 3.658911347 s; the returned
HVP differs from Slurm 3176 by less than 9e-15 Eh/Bohr squared, with identical
reported semantic work counts and unchanged response convergence. This run is
not an exclusive unprofiled Hessian measurement and is not a comparative
speedup qualification.

Slurm 3186 subsequently measured complete, unprofiled H2 and water Hessians
from clean source `74c90701cfc95a2f7faa0f8e0397692197180a6a`. An exclusive
four-GPU allocation exposed exactly one GPU to the measurement step. Both
engines completed first-observed use plus two interleaved artifact-primed
fresh-process/fresh-SCF repeats. Native/GPU4PySCF primed medians were
1.717712493846193/5.59893289511092 s for H2 and
13.43672386649996/10.781149864429608 s for water. The unchanged 3e-6
Eh/Bohr-squared inter-engine gate still failed, at
9.326707032836223e-5 and 0.0010456709790797603 respectively. Qualified ratios
remain null: the independent native energy oracles do not establish why the
two engines disagree. Small 8-by-4-by-8 grids are not grid-converged chemistry.
The measured script still included an obsolete native-only 30 s deadline
margin (630 s native, 600 s reference); no sample approached either limit.
The subsequent runner cleanup gives both engines the requested deadline and
is not retroactively part of this measured source.

Full matrices, semantic counts, all twelve endpoint samples, exact selected
inventory and reproduction are retained in
`benchmarks/results/wb97mv-hessian-native-parallel-20261016/`. The first
deployment attempt (Slurm 3184) failed importing a benchmark helper before
measurement and is not timing evidence. Complete f/g closure and migration of
the separate legacy Python Hessian numerical API remain outside this receipt.

## Revisit when

Complete endpoint profiling shows remaining source duplication, low occupancy,
instruction/cache pressure or shell-group launch overhead. Any further batching
must preserve bounded resource accounting, source work counts, immutable owner
authentication, deterministic contraction and whole-result failure behavior.

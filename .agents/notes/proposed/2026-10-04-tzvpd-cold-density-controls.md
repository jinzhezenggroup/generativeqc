# Full def2-TZVPD cold-density controls

Status: proposed performance policy; implemented benchmark controls only
Date: 2026-10-04

## Decision

Extend the existing OMol25-level water-cluster comparator with explicit
`--preliminary-provider none|lda16|pbe16`, `--reference-full-fock`, and
`--force-active-ao` controls. These are experiments, not production defaults.
The original and displaced geometries and all fixed-density replays retain
the independent 1e-8 Eh / 1e-7 Eh/Bohr gates and complete force host export.
The reference Fock policy remains part of the shared scientific protocol.
Native seed and AO choices are recorded separately as engine policy.

The private source uses the target's exact loaded basis snapshot, including
diffuse primitives and f shells. Source construction, preparation, solve,
export, admission, destruction, and bookkeeping have disjoint timers; the
entire wrapper is included in complete cold preparation. Only density and
source coordinates cross owners. Target functional, tolerances, DIIS, and
Fock state are not replaced. Source AO discovery is disabled because the
experimental native discovery contract currently admits WB97M-V only; the
target environment is restored even if source preparation fails.

## Boundaries

The earlier def2-SVP source helper hard-coded both the basis and looser target
controls. Reusing it unchanged would invalidate a TZVPD cold comparison.
The new helper reads the actual target identity and controls. Missing work
counts remain null, and point-AO-square domains are not FLOP estimates.

The native checkpoint transfer is a benchmark-private experiment. It does
not qualify a public initialization API or a combined source/target memory
budget. A nonconverged source leaves the ordinary target initial guess intact;
its spent lifecycle is still charged. Preparation and import errors fail the
experiment instead of silently reporting a successful seeded run.

Neither past SVP speedups nor old through-f measurements qualify the current
full-grid TZVPD endpoint. Publish only complete, current-source comparisons
with their actual XC backend, scheduler, binary identity, and all-repeat gates.
The >1024-AO force admission limit requires separate resource qualification;
it is not changed by this benchmark.

## Validation

Host guards verify exact diffuse-basis reuse, policy restoration on failure,
and null unmeasured work. GPU qualification must retain original and displaced
energy/force comparisons for both seeded and unseeded trajectories before any
policy promotion or README performance claim.

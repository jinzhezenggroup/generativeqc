# Combined forces consume the materialized dddd recurrence

Status: implemented default-off candidate; frozen-source device qualification passed
Date: 2026-10-06

## Problem

#2030 selected `Full` for Combined total forces, while the optional materialized
pair derivative scheduler from #2020 admitted only `FullSources`. Combined
therefore used per-AO recurrence even when the materialized derivative switch
was enabled. The same mismatch affected bounded and angular scheduling.

## Decision

Template the compiler-owned materialized force consumer on the existing
`DirectForceOutputMode`. Its common `DirectForceSources` contract provides
one signed J/K weight and one output array for Combined, or two independent
weights/arrays for Separate. Both consume the same scalar order-9 Coulomb
simplex and incumbent Dual3 Hermite responses. There is no additional spin or
exchange factor, new derivative recurrence or precision change.

Both native launchers admit the materialized specialization for full-range
forces, and their generic drains skip exactly the dddd tasks that it consumed.
The original default-off preparation switch, resident-cache requirements and
reachable/convolution exclusions remain. Other shells and radial operators
retain their existing consumers. Connecting the candidate is separate from
promoting it based on complete endpoint measurements.

## Invariants and evidence

The native gate checks signed combined forces against an independent host
symmetry-orbit contraction of retained raw derivatives, alongside Separate.
It covers both spins, repeated atoms, same-pair triangular domains, Schwarz
screening, inactive/zero-density tasks and a zero second-output canary.
The existing public Libcint gate also exercises Combined and Separate at the
same final density with materialized derivatives enabled.

Host production-dispatch qualification covers Combined, Separate and long-range
requests, every resident/cache admission state, workspace selection, disjoint
order ownership and submission error propagation. The initial targeted host
run passed 11 tests; compiler structure checked 480 modules with zero errors.
Frozen-source device results and the minimal endpoint comparison are recorded below.

## Frozen-source qualification and minimal endpoint evidence

Source `1b601e705368bf99e751f1b5e4c83703d52efedd`, RTX 5090/sm_120,
finite Slurm allocations, ccache C++/CUDA launchers: job 6263 passed the native
recurrence/force gate (1 test, 37.48 s including compilation) and all eight
PBE0 Libcint cases (48.91 s). The cases cover RKS/UKS, Cartesian/spherical,
bounded/angular scheduling and both output layouts at a fixed final density.
These results precede the later master integration commits; they do not claim
new GPU qualification of those commits.

Job 6265 compared the same source with materialized derivatives disabled/enabled.
Each geometry has one instrumented diagnostic replay and then one plain replay.
The independent reference is an explicitly attributed two-repeat subset of the
existing five-repeat reference; no reference numerical values were changed.

| Measurement | Disabled | Enabled |
| --- | ---: | ---: |
| Plain warm complete endpoint (s) | 31.454813 | 25.780899 |
| Plain moved-warm complete endpoint (s) | 25.999660 | 25.879465 |
| Diagnostic warm J / K (s) | 2.455698 / 1.970216 | 2.456153 / 1.944387 |
| Diagnostic moved-warm J / K (s) | 2.409523 / 1.971027 | 2.396846 / 1.948154 |
| Diagnostic warm two-electron force (s) | 7.225707 | 7.067407 |
| Diagnostic moved-warm two-electron force (s) | 7.225568 | 7.088373 |
| Profiled dddd worker (ms) | 526.038 | 387.067 |
| Registers / thread | 255 | 255 |
| Local bytes / thread | 87,240 | 87,952 |
| Theoretical active blocks / SM | 1 | 1 |

The disabled plain warm sample is an outlier. With only one plain sample per
phase, do not infer a robust endpoint improvement from it. The repeatable
force-region change is about 2%, and the isolated dddd worker improves about
26%. Cold/moved iteration counts differ between runs, so those endpoints are
numerical evidence only. Every record passed the independent-reference gates;
maximum energy and force errors over both modes are approximately
1.06e-10 Eh and 3.04e-11 Eh/Bohr.

Both modes admitted exactly 92,233,228 shell quartets, 1,263,186,780 AO quartets
and 5,944,643,268 primitive-AO quartets. These are measured admission counts,
not measured weighted-recurrence or force-atomic counts. The native gate checks
its explicit preparation/contraction/publication counters; no production
counter or achieved-occupancy measurement is inferred from those synthetic
checks. The retained generic private frame still limits this materialized
specialization; the cooperative follow-up removes it from its dedicated worker.

Local receipts are retained under `.artifacts/materialized-force/evidence/`
(`gates-6263`, `pbe0-endpoint-6265`) in the author's checkout and under
`.artifacts/force-jk/` in the frozen n1 checkout. They include source/binary
hashes, Slurm visibility, ccache statistics, reference provenance, all numerical
records, launch resources and diagnostic work ledgers.

## Consequences

The disabled candidate keeps its prior specialization and resource footprint.
The enabled Combined specialization has one density weight per AO component
instead of two. Materialized recurrence publication is still prepared by one
CTA lane, and component-level global force atomics remain; cooperative
publication and shell-level reduction are separate follow-up work.

## References

- #2020: materialized pair derivative consumer and fallback.
- #2030: Combined stationary total-force layout.
- `docs/developer/direct_pair_recurrence.md`: current contract and GPU gates.

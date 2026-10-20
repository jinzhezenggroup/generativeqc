# Decision: qualify default README endpoints by common observables

Status: implemented
Date: 2026-10-20

Supersedes the earlier
[stock endpoint campaign](2026-10-20-readme-default-endpoints.md), whose original
receipts and rationale remain unchanged.

## Problem

Replacing historical grid and solver overrides with engine defaults reversed
several apparent 96-atom speed rankings. Treating this as a native performance
regression, or copying residual thresholds numerically between engines, confuses
different scientific and iterative work. Density RMS and orbital-gradient norm
are not interchangeable stopping criteria.

## Decision

The new campaign keeps GPU4PySCF's default XC/VV10 quadratures, convergence,
screening, initial guess and incremental Fock behavior. Native Standard/Fast use
their public defaults and the matching Treutler/Lebedev/NWChem quadratures. Both
include complete moving-grid force response. DF auxiliaries come from the stock
GPU4PySCF selection and are frozen before native timing. There is one reference
profile, no GPU4PySCF Fast, and no displaced-geometry timing.

Common acceptance gates, declared before this campaign, are energy error per
atom <= 1e-5 Eh, maximum force-component error <= 1e-4 Eh/Bohr and force RMS
error <= 3e-5 Eh/Bohr. Fresh tighter direct references at XC levels 4/5 and VV10
levels 2/3 establish a common target including the DF approximation error.
Reference-grid differences and CPU/GPU small-system spot differences must meet
one quarter of these gates. Every timed warm repeat is checked. Missing or
unstable references leave results unverified; failed populations retain their
actual measurements but do not become qualified plot points.

This qualification is limited to the sampled static geometries. It does not
establish displaced-geometry force accuracy, a globally optimal configuration,
or universal Fast speedup. Native WB97M-V DF uses exact long-range exchange while
stock GPU4PySCF fits it; shared direct-target error gates do not make those two
approximate operators identical.

## Controlled configuration evidence

Twenty-nine diagnostic configurations use the same production source
`e5fa4d2f0cd13775327261a0b6ced1eab6006c62` and binary, on one n1 RTX 5090 per
Slurm job. These are diagnostic single cold/warm pairs, not the new README
population medians. Native absolute latency improved under defaults, while the
reference improved much more:

| Method | Native historical cold/warm (s) | Reference historical (s) | Native default (s) | Reference default (s) |
| --- | ---: | ---: | ---: | ---: |
| PBE | 42.06 / 8.84 | 79.63 / 11.88 | 40.72 / 6.72 | 15.38 / 3.61 |
| PBE0 | 61.16 / 10.45 | 79.58 / 13.84 | 49.33 / 8.37 | 19.15 / 4.88 |
| r2SCAN-3c | 118.56 / 24.79 | 146.25 / 33.83 | 83.35 / 18.94 | 29.06 / 10.91 |

The historical grid contains 2,359,296 points. The default pruned quadrature has
about 1,078,000 points. Historical reference settings also forced energy 1e-12,
orbital gradient 1e-10, screening 1e-14, full-current-density Fock rebuilding,
zero density pruning and grid response. The diagnostic reference-default column
uses its default `grid_response=False`; the new README explicitly enables it.

PBE0 leave-one-setting-out measurements distinguish important interactions:

| Configuration | Cold / warm (s) | SCF cycles |
| --- | ---: | ---: |
| All historical overrides | 79.58 / 13.84 | 37 / 3 |
| Restore default grid only | 66.49 / 10.44 | 37 / 3 |
| Restore default convergence only | 29.24 / 10.27 | 9 / 1 |
| Restore default screening only | 76.32 / 13.17 | 37 / 3 |
| Restore default grid response only | 84.39 / 6.40 | 42 / 1 |
| Restore incremental Fock only | Failed | 100-cycle limit |

These are not additive timing contributions: changing a setting can change the
SCF trajectory. A later pilot at energy 1e-10 and orbital gradient 1e-8 with
default incremental Fock failed PBE/PBE0 warm and r2SCAN-3c cold convergence at
50 cycles. Artificially tightening the timed reference or disabling its
optimization would not resolve the intended comparison. Full-density rebuilding
belongs only to the separately labeled strict accuracy oracle.

## Fast work and remaining performance limits

At frozen source `02337e0cf3d94915a352d5056f6d09d27fe353f2`, 96-atom Direct
Fast PBE/PBE0/WB97M-V warm receipts perform two mixed and two strict Fock builds;
r2SCAN-3c performs two mixed and three strict builds. Standard warm performs
one. The public Fast profile restarts its mixed/refinement sequence, so its
looser threshold alone does not imply less work for an already converged seed.
HF Fast additionally runs a complete independent tighter audit. The CUDA warm
density admission and separate audit-seed fix in PR #2647 removes the previous
host eigensolver/cold-audit overhead without deleting those scientific checks.

Default auxiliary g support removes an admission gate; it does not qualify all
sizes for every resource budget. The 96-atom WB97M-V DF force still reports a
prepared response resource limit. Other 96-atom DF force endpoints can dominate
latency despite a one-iteration warm SCF. These must not be described as
over-convergence or hidden behind substituted auxiliaries.

## Invariants and revisit conditions

- Freeze measured production source/binary and retain exact executed drivers.
- Preserve original failed and superseded receipts rather than relabeling them.
- Bind qualification to complete receipts, geometry, versions and reference
  identities; reject malformed force shapes before NumPy subtraction.
- Keep README prose short and place controls, diagnosis and failure details in
  the evidence report. Restore the earlier 2x4 figure arrangement; leave CBS alone.
- Consider new stopping/precision policies only after independent E/F gates and
  complete endpoint work demonstrate the benefit. Do not tune defaults to win
  selected water sizes or suppress genuine slower Fast results.

## References

- README measurement/qualification drivers under `benchmarks/readme_defaults*`.
- Original default campaign: `benchmarks/results/readme-defaults-20261020/`.
- PR #2647: default auxiliary admission and bounded warm/auditor density reuse.
- `docs/maintainer/performance_engineering.md` for production qualification scope.

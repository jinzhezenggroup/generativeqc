# Decision: reject extending the mixed semilocal initializer

Status: rejected
Date: 2026-10-16

## Problem

Allowing one frozen-nonlocal-potential proposal to finish in mixed precision
requires more than the incumbent twelve-Fock warmup budget. Extending that
budget without independently bounding the initializer also oversolves the
incomplete semilocal seed operator. Fewer strict Focks do not imply less work.

## Rejected candidate

The first candidate allowed sixteen initializer iterations, one mixed target
and a bounded mixed frozen-potential proposal. Completed proposals handed off
directly to a fresh strict target; final physical and observable gates remained
unchanged. The small 3/12-atom independent energy/force checks passed.

Slurm 1638 nevertheless measured this clean, same-card pair on one RTX 5090:

| Complete 96-atom endpoint | Incumbent Fast | Extended mixed initializer |
| --- | ---: | ---: |
| Seconds | 858.517744 | 1065.228812 |
| Total Focks | 19 | 24 |
| Mixed / strict Focks | 12 / 7 | 20 / 4 |
| Nonlocal pair sweeps / exact cache hits | 3 / 1 | 3 / 1 |
| Evaluated mixed AO-ERI recurrences | 17,717,684,496 | 29,529,474,160 |

This is a 24.08% endpoint regression, not a successful precision optimization.
Both arms start fresh processes/calculators and hcore densities, request full
moving-grid forces, use direct RKS, spherical def2-SVP (768 AO), identical
`GridSpec(48,16,32)`, four CPU threads and a 180-iteration ceiling. Neither arm
reuses a converged density. This pilot is not a repeated-sample speedup estimate.

The candidate also passes the authenticated retained independent 96-atom
accuracy oracle: energy error `2.22e-9 Eh`, maximum force error `6.48e-6 Eh/bohr`,
RMS force error `1.91e-6 Eh/bohr`. Historical timing is not used. Correctness is
necessary but does not qualify this scheduling change as faster.

## Consequences

- Keep the initializer's incumbent eleven-step allowance independent of the
  extra mixed frozen-potential proposal budget. Do not move extra work into
  the wrong fixed-point problem merely to avoid strict Focks.
- Diagnose mixed J/K's work as well as its scalar type: the incumbent canonical
  mixed traversal lacks the generated strict traversal's density screening.
  Preserve FP32 recurrence / FP64 contraction-products-scatter rather than
  removing the generated weighted-FP32-product guard without new qualification.
- Retain failed candidates and instrumented profiles separately. The traced
  849.518-second endpoint is diagnostic, not this clean 858.518-second baseline.
- A first-use 12-atom observation of 56.965 versus 15.392 seconds is not a fair
  steady-artifact comparison: it has asymmetric first-use preparation. Retain
  explicit artifact warmups and compare repeated fresh-owner endpoints instead.

## Evidence identities

The incumbent source identity is
`eda2388f5b64f1209c85498d2f5220139989a3bc2cddc1508bf96d5127780bc5`;
its core SHA256 is
`3099c3fc3f1868fa86724253f11529b67a56e53420a7a450fbe9c22acdf4d8b2`.
The rejected source identity is
`70f54b107038cac039e9e755eb3daccf57f54e5c9bc883840f1b056b0908e52a`;
its core SHA256 is
`1ca8be6f6585bd72ad074cce62024c4cf94b99a7598f3d62ba24692843f95c82`.

Raw receipts, the original freeze, scheduler logs and independent-oracle
authentication are retained outside the repository under
`../evidence-fast-followup-20261016/`, notably `*-clean-96-r1.json`,
`candidate-proposal-freeze.json`, `clean-pair-r1-slurm.log` and
`proposal-clean-96-r1-accuracy.json`. Do not relabel these as later candidates.

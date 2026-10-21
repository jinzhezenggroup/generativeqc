# Decision: stream the short strict tail of relaxed Fast hybrids

Status: implemented
Date: 2026-10-21

## Problem

Fast PBE0/96 already takes nine target SCF iterations, matching the reference
count. Its cold deficit therefore cannot be explained by excess iterations.
Automatic source residency constructs 1,476,473,708 FP32 integral values for
seven mixed Focks, then reconstructs the same count in FP64 for two closing
Focks. On the counterfactual baseline, the second preparation takes 8.130 s,
inside an 8.139 s precision-handoff interval. These nested intervals are not
independent costs.

## Decision and boundaries

After auditing and retiring the mixed lease, use the available generated strict
provider for full-range through-d hybrids when application-budget stationarity
is active. This retains the useful mixed source while avoiding the second
quartic construction. The measured handoff becomes 0.00645 s; construction
produces only the original FP32 values, and the cold solve still reports seven
mixed Focks, two strict Focks and a final residual audit.

This applies only to automatic source selection. Explicitly tighter stopping
controls, forced residency, range corrections, higher angular momentum and
missing generated capability retain their existing admission/fallback policies.
The mixed lease is retired before selecting the strict provider, including
warm replay and failure paths. Grids, energy/density/residual thresholds,
precision arithmetic, force response and final-state acceptance do not change.

## Complete-endpoint evidence

Both binaries use master `9d1943f3c80b495c42595e41a327e578658c78ee` plus the
provider-capacity/pure-J fix, local base
`751c5d8424f6dcea3a816cba27f66c1e53c23c82`. The counterfactual changes only
`src/dft/cuda_ks.cpp` back to that base; other source/asset inputs match. Both
complete CUDA Release builds use verified sccache 0.16.0 and retain compiler
commands and cache statistics. All GPU runs use finite Slurm allocations on
n1/node1 RTX 5090 with CUDA 12.9.1.

Each population includes constructor/preparation, cold SCF, complete analytic
forces and export, followed by three warm E/F calls from its own frozen cold
density. Each arm contains PBE0/B3LYP/M06-2X at 3/12/48/96 atoms, traced PBE0/96,
and unchanged r2SCAN-3c/96 and WB97M-V/96 controls.

| Fast Direct case | Baseline cold (s) | Candidate cold (s) | Target iterations |
| --- | ---: | ---: | ---: |
| PBE0/48 | 13.110 | 10.932 | 9 |
| B3LYP/48 | 13.335 | 11.058 | 9 |
| M06-2X/48 | 13.957 | 11.617 | 10 |
| PBE0/96 | 38.753 | 29.521 | 9 |
| B3LYP/96 | 38.732 | 29.955 | 9 |
| M06-2X/96 | 40.657 | 32.661 | 10 |

For the three 96-atom hybrids, warm medians are respectively
7.985/8.073/8.160 s before and 7.785/7.930/8.337 s after. Thus the removed
construction is not deferred into warm replay. Small differences are not a
statistical speed claim: M06-2X/12 cold, for example, changes from 2.538 to
2.624 s. Preserve every point, including slower samples.

Independent README qualification uses unchanged gates of 1e-5 Eh/atom energy,
1e-4 Eh/Bohr maximum force and 3e-5 Eh/Bohr RMS force, with quarter-gate dense
reference stability and CPU/GPU spot checks. The original oracle identities
are preserved through the explicit reference-reuse manifest. All 30 populations and 120 E/F calls complete; 28 populations qualify. The
known M06-2X/3 default-grid failure remains in both arms and is not ranked as a
win.
These measurements do not establish universal leadership; the three 96-atom
hybrid cold endpoints still trail the existing GPU4PySCF observations.

The executable regression compares automatic, forced and disabled residency
against independent converged PySCF PBE0/B3LYP energy and full moving-grid
forces on both original and displaced water tetramers. It checks native Fast
mixed work, strict closure, final auditing and frozen warm replay without
importing reference densities. Host source-policy/admission tests pass (11).
The initial test omitted the displaced coordinates on its warm replay and
compared different geometries; that failed test and the corrected test bytes
are both retained. The corrected two-method test passes. The corrected PBE0
test also passes Compute Sanitizer memcheck with zero errors.

## Provenance and retention

Candidate core SHA-256:
`27d4f3514f8d9930b8026b844df3827cea3a3463b9a6ff044f46cb043b0199fc`.
Counterfactual core SHA-256:
`3a42dbd6d699d1a850c31390dc99d5842eb47a75fa72a57c5af256289fa6077f`.
The candidate source patch SHA-256 is
`32cf092efc8423897b8bed3fde69bfc85e0bb0171378b138bfc830b26dcbcbcc`;
the manifest verifies 5,853 source/asset inputs in the build and frozen runtime
trees. The corrected regression test is a separately hashed test-only revision;
the measured native source and library are unchanged.

Ignored local `.artifacts/strict-tail-{baseline,residency}-v2/` and remote
`n1:/data/jzzeng/scf-performance-20261021/strict-tail-{baseline,residency}-v2/`
retain manifests, builds, binaries, complete receipts, outcomes, trace summaries,
qualifications and test/sanitizer logs. The initial broader v1 predicate and its
measurements remain separate; they are not relabeled as the guarded v2 source.
No README or CBS point is replaced by this diagnostic campaign.

## Rejected alternatives and remaining work

Disabling all source residency is substantially slower despite identical target
iterations and passing accuracy: the mixed source is valuable. The preceding
[cold diagnosis](../../proposed/2026-10-21-cold-scf-iterations-and-setup.md)
retains these ablations and separately identifies Standard's strict residual
tail, Huckel budget refusal and the source-only DF generation bottleneck.
Revisit strict-tail residency if a source generator removes construction cost
or a qualified consumer demonstrates enough strict reuse to recover it.

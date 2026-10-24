# Decision: do not perturb converged Fast warm seeds with approximate XC

Status: implemented
Date: 2026-10-14

## Problem

The actual README prepared runner exposed a regression hidden by fresh Hcore
singlepoints: application-Fast PBE0 started every frozen warm replay with mixed
XC. That perturbed a strictly converged resident density, then required another
complete strict refinement. Native FP64 did one Fock build, but Fast did 13 at
24 AOs and 14 at 96 AOs. This was real extra SCF work, not Python audit overhead.

Slurm 8378 retained all 96 complete direct/DF PBE0 attempts at 3/12 atoms, using
five frozen post-cold and post-move replays, automatic initialization and explicit
GridSpec(48,16,32). All independent CPU PySCF observable gates passed. Direct warm
FP64/Fast ratios were only 0.1698/0.1939; moved-warm 0.1695/0.1954. DF already used
strict SCF plus FP32 geometry and did not have this regression: warm ratios were
1.1057/1.1649. The losing population remains in `readme-prepared-8378`.

## Decision

For the budget-admitted application XC warmup only, start a **resident,
same-model** warm replay in the existing strict arithmetic phase. Still evaluate
the real physical Fock, proposal, energy change, density change and both residual
norms. No cache entry bypasses convergence or final-state/gradient-token checks.
The gradient retains its independent budget-selected FP32 consumer.

Cold, external and changed-geometry seeds keep the qualified mixed trajectory.
The resident source is already owned by the same native plan and has passed its
strict publication gates. Ordinary AUTO and experimental precision controls are
unchanged; this is not an unqualified global mixed-SCF scheduling promotion.

The iteration limit must distinguish strict-start from mixed refinement:
`refinement_iterations` only advances after actual mixed execution. A strict-start
trajectory therefore uses the ordinary total-iteration limit. Checking the strict
dispatch flag alone would permit an unbounded nonconvergent replay.

## Invariants

- Warm energy remains anchored to the incumbent provider. Existing source-lease
  and energy-epoch guards remain unchanged.
- Report no mixed SCF builds/refinement if none ran. Positive FP32 gradient work
  remains independently visible; output float64 arrays do not identify kernels.
- Frozen updates cannot overwrite the seed; failures cannot certify convergence.
- Retain independent same-model E/F gates and complete endpoint work/timing.
- Do not fix this by converting all prepared Fast SCF to FP64, changing stopping
  tolerances, hiding warm costs, importing auditor densities, or cold-starting
  the FP64 comparator. Those would change the intended comparison or remove
  useful mixed cold work.

## Evidence and retention

The regression fixture checks PBE and PBE0 at 24/96 AOs, retains positive cold
mixed work, verifies frozen warm strict work with a bounded Fock count, checks
physical residuals and positive FP32 gradients, and compares against independent
PySCF observables. Ordinary AUTO warmup regressions remain in the GPU test set.

The 8378 binary is
`5b7fc0ef88c97ee56246c91c55428b3cb8df972ca5d68fe576a1ce5792bd9599`;
its runtime overlay is retained as `prepared-8378-python.tar.gz` with SHA256
`21df0c82cc3d02962b2184e456cca285ca3ff9f3248dc421213e11a34336c7bb`.
The bounded native change rebuilds with verified ccache on Slurm 8379: six cache
hits and two misses; final source/binary hashes are retained with the log.

Slurm 8380 remeasures all 96 attempts with exclusive-node allocation and the
same complete cold/frozen/moved protocol. Final binary SHA256 is
`f993f2ba55c378b4c8102a682226ad8ed7193903d1dc674632445b3fec886bf6`.
All independent CPU PySCF gates pass; all 48 Fast calls have positive FP32
geometry work, one candidate fleet and no audit replay. Both direct warm
populations now use one Fock build. Maximum Fast errors are 9.67e-13 Eh and
5.82e-7 Eh/bohr. FP64/Fast complete-endpoint ratios:

| PBE0 population | Cold | Frozen warm | Moved | Frozen moved-warm |
| --- | ---: | ---: | ---: | ---: |
| Direct, 3 atoms | 0.9881 | 1.1636 | 0.9634 | 1.1584 |
| Direct, 12 atoms | 1.0719 | 1.1845 | 1.0301 | 1.1710 |
| DF, 3 atoms | 0.9873 | 1.0836 | 1.0035 | 1.0868 |
| DF, 12 atoms | 1.0138 | 1.1623 | 1.0239 | 1.1649 |

Warm entries retain the full five-call populations in raw JSON; cold/moved are
single observations, not statistical speedup guarantees. The small cold and
direct moved regressions remain visible. This is an 8–18% observed warm gain,
not 2–7x complete-endpoint acceleration or large-system qualification.

The measurement uses the actual README native CLI. GPU4PySCF is absent in this
environment, so raw rows deliberately remain `completed_unchecked`; the separate
summary records exact-geometry independent **CPU PySCF** gates, not fabricated
GPU4PySCF timings. The first population uses a shared node; the final population
retains `scontrol` evidence of its exclusive allocation. This is
not a universal speedup, all-size qualification or a replacement of README's
historical FP64 publication.

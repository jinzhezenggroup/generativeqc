# Decision: qualify Becke source owners on converged physical clusters

Status: implemented
Date: 2026-10-06
Related: #1894; draft PR #1996

## Problem

The synthetic native-owner cohort establishes execution, invalidation, counters
and sanitizer behavior, but is not an independent complete molecular force
oracle. Small H2/H3 fixtures cannot select the primitive: the compiler and native
phased cache both require more than 32 atoms, independently of the automatic
48-atom policy threshold. Changing that policy in a test would not bypass the
cache admission guard.

## Decision

Use neutral 36-atom closed-shell and 35-atom doublet hydrogen clusters, retaining
the existing cache guards, native arithmetic and production thresholds. PBE0 and
an IR-composed PBE0 with a deliberately different public label use separated H2
fragments and, for UKS, one distant H atom. The nonlocal UKS gate instead expands
the existing asymmetric H3 doublet with sixteen H2 fragments, retaining 35 atoms
and the same point/pair domain rather than retreating to an unselected small
system.

The alternative nonlocal fixture is a convergence precondition, not a solver
optimization or a relaxed force gate. Preserve the failed isolated-H experiment
and both primitive-off/on cold controls. Both controls exhaust 200 iterations;
neither executes a force. Their final states/trajectories differ, so do not claim
matched solver work beyond the observed iteration count. Both H3 controls
converge in eighteen iterations with unchanged energy/density/screening
thresholds and no reference seed. Do not infer the cause of the isolated-H SCF
failure from these controls or claim that it is fixed.

## Evidence

- n1 job 6170: four PBE0/PBE0-label-independent RKS/UKS force cases pass, with
  actual primitive selection. The PySCF gradient oracle uses the returned native
  state; reconverged native directional differences and moved/fresh-owner
  comparisons supplement it. This is not independently converged reference SCF.
  The same job's initial nonlocal RKS case passes, but its isolated-H UKS case
  fails cold SCF. Retain that terminal failure, not a successful aggregate job.
- n1 job 6175: the four cold controls above retain complete native histories,
  actual iterations/Fock builds and null stationary force work. No force or
  Becke primitive execution is attributed to these controls.
- n1 job 6179: two nonlocal RKS/UKS physical cases pass independent GPU4PySCF
  cold/moved energy and force gates, warm replay and all three reconverged
  directional differences. Both semilocal and nonlocal owners actually select
  the primitive and report the complete dense point/pair counts.
- n1 job 6183: repeat the same two cases to retain complete oracle/force vectors
  and actual source-owner linked/object metadata, including generated source,
  included headers, compiler options and binary identities. This is additional
  observation of two cases, not two new independent physical cases.
- Reviewed numerical publication:
  `benchmarks/results/becke-physical-forces-20261006/publication.json`. Its
  accepted decision is restricted to the specified converged nonlocal force
  cohort, not performance or completion of #1894. Complete 48/96 PBE0 endpoints
  and physical phase attribution remain separate acceptance evidence.
- Reconstruct the published source revision plus its dirty test patch in an
  isolated directory: all 1402 scientific/build/test input checksums pass. This
  verifies source restoration, not a fresh execution of the entire reproduction
  recipe. Host qualification reports 83 passed and 10 GPU-only skips; evidence
  retention, shared compaction and the configured local hooks pass separately.

All runs use finite `main/gpu:5090:1` Slurm on n1 with unchanged assigned device
visibility. The core Release library hash is
`806f28a2191ecbdaea5964e4ab8e3b53629ca1ea44b66aee1cda069df8db3f1d`.
All 1400 scientific/build inputs match the later qualification source revision
byte-for-byte; gate-source changes do not change compiler/native mathematics.
The first build's `/tmp` space failure and the first GPU harness's missing
`ptxas` sibling remain archived separately. Ccache invocation, actual cached
artifact identities and before/after statistics are retained; shared aggregate
statistics do not identify individual owner hits/misses.

## Invariants and limits

Do not lower production/cache thresholds, normalize actual solver work, inject
oracle densities, weaken independent energy/force/FD tolerances, or turn an SCF
precondition failure into a numerical force pass. Nonlocal cold/warm/moved
iteration and Fock-build counts are RKS 8/1/8 and UKS 18/2/10; UKS warm is not
reported as one iteration. The source owners stay separate in telemetry.

Profiling is disabled in this cohort. Raw zero-filled ABI event arrays do not
measure zero elapsed time. Report logical dense-domain traffic separately from
unmeasured hardware transactions. Full force-vector errors are retained for
cold/moved oracles; scalar warm infinity-norm/FD records must not be relabeled as
unmeasured force-vector RMS errors.

Default selection remains off. Preserve generic/bounded fallback and the losing
route publications. This fixture decision does not promote the experiment,
close #1894, fix general UKS convergence, or merge unrelated #1892/#1893 gains.

# Decision: preserve and validate the affine constraint of normalized KS DIIS

Status: implemented
Date: 2026-10-24

## Problem

The Direct DFT cold investigations in PRs #2714 and #2715 exposed two
numerically distinct observations. Fast r2SCAN-3c/96 paired forces differed by
more than the optional 1e-8 Hartree/Bohr comparison threshold. A Standard
screening-disabled displaced-geometry campaign also contained a cold solve that
ran 13 iterations instead of 10, with a transient density change of 7.58 and an
83.86 Hartree energy excursion. All original samples remain retained.

A launch interposer copied only the small physical-slot Gram, final live ring
window and coefficients after the actual production CUDA DIIS update. Four
fresh Standard/96 cold captures, using both pre-XC and tile-replay binaries,
accepted late nearly dependent solves with these affine residuals:

| Capture | Retained histories | max(abs(c)) | abs(sum(c)-1) |
| --- | ---: | ---: | ---: |
| baseline round 1 | 4 | 1883.43 | 3.416e-7 |
| tile replay round 1 | 3 | 158.49 | 9.557e-6 |
| baseline round 2 | 4 | 7443.24 | 4.027e-7 |
| tile replay round 2 | 3 | 4543.22 | 7.648e-8 |

Independent 80-decimal solves of the captured FP64 inputs satisfy sum(c)=1 to
better than 1e-75. An independent scalar replay with libc FMA reproduces the
captured FP64 constraint errors. The augmented-system pivot cutoff alone does
not establish that its solution remains a valid affine combination. In
particular, an invalid solution changes even a Fock shared by every history.
This is a concrete preexisting failure mode consistent with late SCF excursions;
the original 13-iteration sample was not itself instrumented, so it is not proof
that every such excursion has this cause.

## Decision

Normalized KS solves on CPU and CUDA now share partial-pivot forward
elimination and back substitution in `solver/diis_affine.hpp`. The previous
Gauss-Jordan scheme repeatedly updates the affine row with ill-conditioned
intermediate rows; forward elimination and a final back substitution avoid that
source of error growth. The augmented matrix, common metric scaling and 1e-14
pivot cutoff are unchanged. After solving, an allocation-free scalar constraint
check accepts only finite arithmetic satisfying, for n Fock coefficients,

```text
epsilon = DBL_EPSILON
gamma = ((n+1)*epsilon) / (1 - (n+1)*epsilon)
tolerance = 8 * gamma * max(1, sum(abs(c)))
abs(sum(c)-1) <= tolerance < 1
```

The tolerance supplies eight conservative FP64 summation-error budgets for the
coefficient sum and subtraction of one. It scales with cancellation rather than
imposing a fixed coefficient-magnitude cap. An uncertainty reaching one cannot
resolve the constraint and fails closed. The Lagrange multiplier is not a Fock
coefficient and is excluded. This checks the defining affine constraint, not the
full backward error of the augmented system; it is not a general condition-number
guarantee.

A failed check uses the existing chronological retirement and retry policy.
At the two-history floor, preserve current Fock. No convergence, force, precision
or independent-reference gate changes. The guard applies equally to serial,
cooperative and cached normalized CUDA routes. Unnormalized HF and the generic
host DIIS policy retain their prior admission behavior. The SCF dependency audit
admits exactly the scalar header, without opening access to host solver drivers.

## Rejected alternatives

- Renormalizing coefficients would hide an inaccurate solve and its potentially
  inaccurate coefficient ratios.
- Tightening or loosening SCF stopping criteria would conflate output accuracy
  with a broken linear constraint.
- Guarding the original Gauss-Jordan solution alone retires otherwise usable
  histories. The fully qualified first attempt increased PBE/96 cold iterations
  from 11 to 13 in both XC variants. Keep that attempt in `parent-endpoints/`
  and `affine-endpoints/`; it is not the shipped implementation. Scalar replay
  shows forward/back substitution retains the captured 3/4-history windows
  while reducing affine errors from 1e-5--1e-7 to about 1e-13.
- Changing eigensolvers, stopping criteria, or introducing a coefficient cap is
  broader than the observed small-system solve failure requires.
- Treating all Fast cold pair differences as a tile/transport regression ignores
  the same-image controls below.

## Independent controls and scope

Two rounds of the original same-image Fast serial/ordered control have cold
force differences of 2.752e-8 and 2.973e-8. Repeating the same frozen binary and
policy produces 2.345e-8 (serial) and 3.376e-8 (ordered). Thus the optional 1e-8
cold comparison already fails without changing the implementation or policy.
With one identical native density seed, serial/ordered differences shrink to
5.970e-11 and 1.270e-11. This seed comes from a native solve, never an oracle.

For each exact live final state, the diagnostic repeats the force calculation
three times without another SCF. XC components are bitwise identical; integral
components differ at approximately 1e-13. The larger Fast difference therefore
arises between final SCF states, rather than from repeated force contraction at
one state. These controls locate the sensitivity upstream of the fixed-state
force calculation; they do not identify a unique source inside initialization
or claim bitwise cold SCF reproducibility.

The Standard original-binary baseline/tile controls give cold force differences
of 8.329e-12 and 1.043e-11 and identical-seed differences of 5.294e-12 and
6.348e-12. These intrusive runs are numerical diagnosis, not timing evidence.
Every diagnostic process executes two SCF endpoints and six force evaluations.

## Validation and retained evidence

The shared host tests include an analytically solvable diagonal Gram, the actual
captured metric at three common scales, bounded fallback, finite cancellation,
overflow and nonfinite rejection. The CUDA regression replays the captured Gram
with both spin counts and three wrapped ring heads. The captured nonsingular
window is preserved. A common Fock is an independent exact affine oracle; an
inactive neighbor stays untouched and poisoned metrics retain current Fock. The test
fails against the frozen pre-fix library and passes against the repair. Existing
ordered/cache parity and ring-lifetime coverage remains in place.

Raw receipts, snapshots, interposer, scalar replay, input manifests, cache stats,
compiler commands and all failed setup attempts are retained under
`n1:/data/jzzeng/direct-dft-followup-20261024/`, with local mirrors in the XC PR
worktree's ignored `.artifacts/followup/`. Every real-device run uses finite Slurm
allocations on node1 and preserves scheduler visibility. Builds use verified
sccache 0.16.0; the local host check uses verified ccache 4.5.1. Instrumented
captures and force repeats are excluded from latency claims. Existing benchmark
publication archives and their failed/slow observations are not replaced.

The final forward/back-substitution builds qualify 46 complete SCF energy/force
endpoints against the retained independent references: 16 same-seed diagnostic
endpoints, 12 cold/warm/moved endpoints, ten PBE/PBE0 Direct/DF endpoints, and
eight instrumented PBE/3c endpoints. Diagnostics add 32 fixed-state force repeats,
for 78 total force evaluations on the final builds. Across the original,
rejected guard-only and final cohorts, 116 endpoints and 212 force evaluations
remain retained and independently qualified. Qualification uses the existing
1e-5 Eh/atom, 1e-4 max-force and 3e-5 RMS static gates, and the unchanged
2.5e-6 / 2.5e-5 / 7.5e-6 displaced-geometry gates. Dense-reference grid stability,
CPU/GPU small-system agreement, geometry and basis/spin identities are rechecked.

Both final XC variants retain PBE/PBE0/3c cold iteration/Fock counts 11/9/10;
all standard warm calls use one. The four 3c moved runs each use 10/1/6, with
maximum independent force error 2.005e-6 Eh/Bohr. Paired final XC variants differ
by at most 1.322e-11 over those cold/warm/moved forces. Actual late GPU solve
captures have maximum affine error 4.000e-13. Native DIIS and ordered-cache
suites, the large staged route, host coefficient tests and 165 SCF ownership
tests pass. The new GPU regression fails with the original binary and passes
with the repair; memcheck and synccheck report zero errors. Public host launch
admission covers 37 rejections, three original admissions, failure injection and
300 independent byte-boundary cases without initializing a GPU.

Fast cold paired differences remain 2.774e-8--3.715e-8 on the final builds and
still fail the optional 1e-8 comparison. Identical-seed differences are at most
7.296e-11. Same-image repeat differences also remain above 1e-8. This issue is
explicitly unresolved; neither the guard nor the changed elimination method is
presented as a fix for deterministic Fast initialization or as a relaxed gate.

Final pre-XC library SHA-256:
`9291d4230ceb4c6f735b5cfa4b8c60857c7fd7a59d134e99f3337244828113db`.
Final tile-replay library SHA-256:
`ca0af859fdb34a9257dca06af4bf4969b0747081273bf1fe15bd4f6fe9d5b6d6`.
Their complete build input manifests are `lu-parent-inputs.json` and
`lu-inputs.json` at the retained campaign location. The lossless numerical text
bundle `numerical-evidence.json.xz` has SHA-256
`32c7b83c387c3e6627445fcec42bb900c5845b711835cf95ce05e822bd7551bb`;
large state/seed NPZ files remain alongside the individual receipts. The bundle
includes the 80-decimal/scalar replay and offline qualification scripts. The
original 13-iteration receipt remains in the preceding XC campaign archive.

## Revisit when

A valid affine solve still causes excessive extrapolation, or broader methods
need a condition-aware solver. Such work needs independent residual/solution
oracles and endpoint qualification; it should not weaken this affine invariant.

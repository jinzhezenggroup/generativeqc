# Decision: compensate the shared generated Direct-J owner

Status: J-only component qualified; complete coupled outcome recorded in the follow-up
Date: 2026-10-15

## Problem

The [Direct-J root-cause investigation](../../proposed/2026-10-15-r2scan-direct-j-roundoff-root-cause.md)
identifies unordered FP64 J scatter as the first differing physical operation
in repeated r2SCAN-3c direct/96 reconvergences. SCF feedback amplifies its rounding
above the unchanged 1e-12 Eh energy tolerance, even when density and residual
are already stationary. A generated-only diagnostic correction improved the
first three iterations but did not qualify native DDDD, full convergence,
forces, optional capacity, or complete endpoint cost.

## Decision

The method-neutral `GeneratedCoulombPlan` retains two contiguous Cartesian
planes in its existing Coulomb allocation: sum followed by rounding correction.
Generated shell workers receive the existing `runtime::CompensatedOutput`;
native DDDD, its materialized consumer and bounded higher-angular completion
receive that correction through launch-local copies of `DeviceBatch`.
The shared batch itself is never rebound, so K and derivative borrowers keep
their own ownership contract. No structure layout or generated launch ABI changes.

Clear both planes at every build. Fold exactly once after all generated,
native and bounded J writers, immediately before either public-AO projection.
Use the existing magnitude-ordered atomic rounding-residual primitive and fold
kernel, not a new recurrence or functional-specific solver policy. Compensation
improves accuracy but does not promise bitwise reproducibility.

Both actual preparation and the shape-only capacity envelope charge the extra
`batch * direct_nbf * direct_nbf * sizeof(double)` bytes. The actual Cartesian
plane inventory is now density, J sum, J correction and Schwarz. A rejected
optional owner retains the existing bounded provider fallback; production does
not borrow CPU/PySCF/oracle work.

## Rejected alternatives

- Relaxing the SCF tolerance or adding an r2SCAN exception hides an upstream
  error and changes the scientific contract.
- Compensating only generated classes leaves native DDDD and admitted bounded
  completion on a different numerical contract.
- Folding before bounded completion loses that completion's recovered low words.
- Rebinding persistent shared batch metadata lets K/force borrowers write into
  J's correction plane.
- Ordinary-FMA/CSE and complete-grid XC reduction experiments did not identify
  this first differing operation; their limits remain in the
  [rejected investigation](../../rejected/2026-10-15-r2scan-cse-rounding-and-xc-totals.md).

## Invariants

- Preserve screening, class masks, operator coefficients, RKS total-density
  semantics, UKS spin summation, AO conventions and force ownership.
- Preserve energy/density/screening tolerances 1e-12 / 1e-10 / 1e-12 and the
  independent molecular gates 1e-8 Eh / 1e-7 Eh/Bohr.
- Keep one clear/fold lifetime per J build, including mixed generated/native/
  bounded source portfolios and dense-zero-dense replay.
- Do not infer determinism or universal performance from three repetitions.

## Evidence

Production base: `5f80ba629b5f20412215cdc690f8fa458ac3fdf8`.
Candidate library SHA256:
`3d48661a7d953ebe4b8f84f844744deb4363e439194b6285020eefce1a74a6cf`.
Only the two changed host C++ objects are rebuilt; the existing correction-aware
CUDA objects and AOT modules are reused read-only. The build receipt records
commands, replacement source/object hashes, dependency hashes and every reused
link input. Final production source hashes match the qualified source copy.

- Focused host allocation/admission/stack tests: 1,539 passed. Existing generated
  scatter/optional-allocation/lifetime tests: 16 passed. Ruff, changed-line
  clang-format and `git diff --check` pass.
- Slurm job 8630, node1/main/gpu:5090:1, finite 20-minute allocation:
  `--compensated-j-only` passes with independent CPU integral/J oracles. Coverage
  includes standalone and shared prepared J entry points, generated-only s/p,
  native-only d/d, mixed s/p/d and generated/native/bounded s/d/f portfolios;
  Cartesian/spherical, RKS/UKS, two unequal systems and both DDDD value policies.
  Both planes are poisoned before each call. Zero reset, finite/nonzero recovered
  low words, unchanged input densities and unchanged shared batch bindings pass.
  Compute Sanitizer memcheck reports zero errors and zero leaked bytes.
- The broader existing `--canonical-values-only` assertion
  `range fallback displaced an independently generated J/K channel` fails in
  both the frozen baseline and candidate. This pre-existing routing/count gate
  is retained as a known failure, not weakened or reported as passing.

Slurm job 8627 runs six fresh processes/owners on node1/main/gpu:5090:1,
with a finite 03:15:00 limit and balanced order B0,C0,C1,B1,B2,C2. Each uses
the unchanged qualified driver, water-96, r2SCAN-3c, spherical basis, direct
no-DF, grid 48x16x32, eight CPU threads and the same original native checkpoint.
The independent reference enters only the post-endpoint energy/force gate.
All six complete, reconverge from that seed without fallback and pass both gates.

| Arm | Complete seconds, repeats 0/1/2 | Fock builds and XC evaluations | Median seconds |
| --- | --- | --- | --- |
| Baseline | 359.814 / 387.876 / 277.904 | 19 / 21 / 13 | 359.814 |
| Compensated J | 284.646 / 284.529 / 284.681 | 13 / 13 / 13 | 284.646 |

Complete seconds include owner preparation plus the entire SCF+force execution,
not an isolated J kernel. Importing the common experimental checkpoint is setup
outside these timers in both arms. The measured complete-endpoint median ratio
is 0.79109 (20.9% lower); execution-only median ratio is 0.78626. The baseline's
fastest 13-build observation is faster than the compensated 13-build observations:
the extra scatter/clear/fold work is not free. The benefit here is avoiding
stochastic extra SCF work, not speeding up a fixed amount of J work.

Candidate energy error is 3.775e-11 Eh or less and force error is 3.913e-11
Eh/Bohr or less. Conservative energy-only blocked iteration counts are 9/11/3
in the baseline and 1/1/1 with correction. These counts require both recorded
maximum density change and maximum residual <=1e-10 while energy change >1e-12;
they do not reinterpret maximum norms as the solver's RMS criteria. Complete
Fock counts equal observed XC evaluations. This build does not expose complete
per-channel J/K application counters, so their unavailable values remain null.

### Cold endpoint negative

Do not promote the moved result into an all-phase performance claim. Three
additional fresh cold calls per arm pass the same original-geometry independent
E/F gates, but retain this negative:

| Arm | Complete seconds, repeats 0/1/2 | Fock builds and XC evaluations | Median seconds |
| --- | --- | --- | --- |
| Baseline | 432.653 / 558.039 / 432.057 | 24 / 33 / 24 | 432.653 |
| Compensated J | 475.643 / 474.880 / 474.565 | 26 / 26 / 26 | 474.880 |

The cold median regresses 9.76%. Slurm 8632 runs B0,C0 and Slurm 8647 runs
C1,B1,B2,C2, both on node1/main/gpu:5090:1 with Slurm-assigned visible device 2
and finite 01:05:00 / 02:05:00 limits. Every cold call has a fresh owner, no
checkpoint input and its own new checkpoint output. The original common moved
seed remains byte-identical. The standalone J correction therefore reduces
J's rounding and warm variability but does not eliminate every energy-only
blocking tail. Numerical success alone is not closure of the entire performance
request. Further component controls must keep the corrected J fixed rather than
repeat the previously inconclusive XC-only trial with noisy J.

## Evidence retention

Raw journals, E/F arrays, histories, source/build receipts, sanitizer logs,
driver wrappers and summary audit remain on n1 at
`/data/jzzeng/qc-r2scan-j-compensation-20261015` and in the local ignored
`.artifacts/r2scan-j-compensation-20261015/` campaign. In particular,
`evidence/complete-endpoint-comparison.json` refuses partial/failed populations
before computing medians and verifies common protocol, seed, oracle, scheduler
and per-arm library identities. Native test v2 is separately source-bound and
does not relink or mutate the library after endpoint measurements start.

## Revisit when

Independent complete endpoints show a different provider dominates, optional
capacity materially changes admission, or an equally accurate bounded reduction
can remove the extra scatter cost. Stronger reproducibility needs a separately
qualified deterministic reduction, not a claim that correction makes atomic
ordering deterministic. Other J providers and K are not promoted by analogy.

## Coupled follow-up (2026-10-15)

The [coupled J/XC decision](2026-10-15-coupled-j-xc-compensation.md) holds this J
correction fixed, tests the remaining scalar-total roundoff, replaces a costly
serial control with a bounded warp reducer, and qualifies all twelve complete
cold/moved calls with real source-matched force modules. It supersedes the
all-phase-pending outcome, not this J-only experiment's cold regression or its
measured fixed-work cost. Keep the separate populations and source identities.

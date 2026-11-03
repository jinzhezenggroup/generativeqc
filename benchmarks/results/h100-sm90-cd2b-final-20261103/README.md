# H100 fixed30 three-pair acceptance, frozen CD2B

The original product target is met on source
`cd2b4514be0a38c4920934481b1f02d157a694b4`, tree
`c76f5bd3cfa62416255c5d6914cf701d34e10a8d`, production core SHA256
`37b8bd02c1dddbf42385c494ae21ff51b50d874dae1dd0c01ab2b75fb0eb194e`:
**30/30 scientifically qualified, 180/180 fresh processes, 720 complete E/F,
20/30 wins, complete-set geometric mean 1.2869162091880366.**
DF wins are13/14; Direct wins are7/16. The ten non-winning cases remain below.
These are frozen-source observations. Later main revisions, including native
compiler, electronic-property and tile-policy changes, do not inherit this
numerical/performance qualification. Current-main integration is separate.

## Independent CPU replay

From a full local Git checkout, with Python3.12+ and NumPy2.2.6:

```sh
uv run --no-env-file --no-project --python 3.12 --with numpy==2.2.6 \
  python -m benchmarks.h100_final_cohort > final-replay.json
```

The command recovers the existing build/DF/support/public-guard prerequisites,
recomputes their accepted and failed proofs, reconstructs the exact1,239,836-byte
final archive from two authenticated parts, losslessly decodes the original
manifest and verifies every member. It then recomputes every final number from
full-precision raw observations. The final proof must match the original SHA256
`5d3a1528ef81a00e7e2dc3afa37d7484d91249126d99b3d396ca4cf8bfb852e6`.
No GPU, solver, network fetch or compute allocation runs during replay.
The public receiver changes only reference-root configuration for the isolated
CPU wrapper; every imported helper is checked against the frozen Git revision.

The existing supporting Nsight archive parts are read through their explicit
local-Git historical snapshot. Their exact bytes, all three SQLite captures and
all12 intrusive E/F remain available. This reserves current-tree space without
raising the80MiB aggregate or1MiB file limits. No Release or external backup is
created. A shallow checkout may lack required historical objects.

## Fixed scientific and timing contract

The original preregistration SHA256 remains
`2391b71fb49660f682f1b60056b2892d5516f01871d7479319f0be2cf925fdaa`.
Thirty physical HF/DFT water-cluster cases cover48/96 atoms and Direct/DF.
WB97M-V DF at both sizes was preexcluded for different long-range fitting
contracts and contributes no equivalent-task vote. Precision arms, repeat calls,
controls and failed precompute launches add no votes.

One documented recommended policy was fixed before measurement:
`integrated-auto-rys-public-tile1024`, Standard, explicit KS tile1024, no DF
environment overrides. This is not automatic promotion of every default.
Basis/grid/fitting/correction inputs, HF matched fitting preflight and independent
oracle identities are retained. The gates remain E/atom<=1e-5 Hartree,
Fmax<=1e-4 and FRMS<=3e-5 Hartree/Bohr; both engines must pass them, as must
all16 cross-engine full-vector comparisons within each process pair.

All180 processes used H10080GB SM9.0/MIG-disabled UUID
`GPU-7c9f7fcf-a830-d861-356c-499f2be3e6ef`,20CPU/200GiB/16GiB shared memory,
one BLAS/OpenMP thread, driver595.58.03 and the pinned CUDA12.9 environment.
Reference GPU4PySCF1.9.0/CuPy13.4.1 and native CuPy13.6.0 use the declared
qualified package contexts. Native execution blocks PySCF/GPU4PySCF imports.
The parent alternates first arm by case and pair. Each fresh process returns
one cold plus three warm E/F calls seeded only from its own cold density;
synchronization and full force export are included. Cold/preparation cost stays
visible; the declared primary statistic uses warm complete-endpoint medians.

A case wins only when the median of its three paired GPU4/native ratios is>1.05,
each pair is>1, and all scientific gates pass. The aggregate uses all30 case
medians, including every non-win. Screening, profiling and supporting guards
contribute zero final timing votes. Raw seconds, iterations/Fock or effective
potential counts, settings, route, warm/fallback state and numerical errors are
retained; unavailable hardware counters are not reported as zero.

| Method | Atoms | Route | Three paired ratios | Median | Win |
| --- | ---: | --- | --- | ---: | --- |
| hf | 48 | direct | 4.220963, 3.705612, 3.825621 | 3.825621 | yes |
| hf | 48 | df | 1.659306, 1.625531, 1.589858 | 1.625531 | yes |
| hf | 96 | direct | 1.071680, 0.999198, 0.992007 | 0.999198 | no |
| hf | 96 | df | 0.835099, 0.816738, 0.844772 | 0.835099 | no |
| pbe | 48 | direct | 1.555909, 1.520606, 1.470054 | 1.520606 | yes |
| pbe | 48 | df | 1.753324, 1.747845, 1.747785 | 1.747845 | yes |
| pbe | 96 | direct | 0.941320, 0.928865, 0.928237 | 0.928865 | no |
| pbe | 96 | df | 1.268269, 1.230576, 1.225791 | 1.230576 | yes |
| pbe0 | 48 | direct | 1.777981, 1.642868, 1.515372 | 1.642868 | yes |
| pbe0 | 48 | df | 1.950802, 1.778019, 1.798536 | 1.798536 | yes |
| pbe0 | 96 | direct | 0.835693, 0.820413, 0.821709 | 0.821709 | no |
| pbe0 | 96 | df | 1.145833, 1.131454, 1.142950 | 1.142950 | yes |
| b3lyp | 48 | direct | 1.564593, 1.562262, 1.627335 | 1.564593 | yes |
| b3lyp | 48 | df | 1.758822, 1.740069, 1.812737 | 1.758822 | yes |
| b3lyp | 96 | direct | 0.812131, 0.846221, 0.782638 | 0.812131 | no |
| b3lyp | 96 | df | 1.129946, 1.150785, 1.130801 | 1.130801 | yes |
| r2scan | 48 | direct | 1.643484, 1.617430, 1.614367 | 1.617430 | yes |
| r2scan | 48 | df | 1.856483, 1.851227, 1.878193 | 1.856483 | yes |
| r2scan | 96 | direct | 1.093578, 1.090166, 1.102363 | 1.093578 | yes |
| r2scan | 96 | df | 1.568507, 1.604284, 1.453907 | 1.568507 | yes |
| r2scan-3c | 48 | direct | 0.805351, 0.957721, 1.413330 | 0.957721 | no |
| r2scan-3c | 48 | df | 1.326938, 1.275319, 1.520688 | 1.326938 | yes |
| r2scan-3c | 96 | direct | 0.545163, 0.542028, 0.624049 | 0.545163 | no |
| r2scan-3c | 96 | df | 1.307999, 1.189326, 1.428070 | 1.307999 | yes |
| m06-2x | 48 | direct | 1.753033, 1.713741, 1.717497 | 1.717497 | yes |
| m06-2x | 48 | df | 2.019914, 2.143884, 2.056843 | 2.056843 | yes |
| m06-2x | 96 | direct | 0.949776, 0.964147, 0.969320 | 0.964147 | no |
| m06-2x | 96 | df | 1.397430, 1.386706, 1.369068 | 1.386706 | yes |
| wb97m-v | 48 | direct | 0.882087, 0.886786, 0.884132 | 0.884132 | no |
| wb97m-v | 96 | direct | 0.892961, 0.903983, 0.897678 | 0.897678 | no |

## Retained failures, identity and review boundaries

Two earlier final launches failed before any molecular E/F or primary attempt:
a missing already-completed Nsight admission file, then a reference-runtime
receipt written into the old output namespace. Distinct original failure records,
platform status/logs and old producer bytes are bundled in
`precompute-failures-v1.tar.xz`; its manifest specifies the exact retained scope.
The full second failed precompute archive remains in private task retention;
common scientific inputs/admissions also appear in the complete successful final
archive. No failed launch substitutes or removes a primary physical case.

All33 core/AOT package artifacts are byte-bound before/after endpoints and the
actual post-endpoint native core mapping is verified for every native process.
Those mappings are sampled after owner close/cache clear and do not prove every
AOT DSO was loaded during timing. Loaded-AOT attribution and achieved occupancy
are not inferred from package identity or elapsed time. The inherited oracle
reuse report retains historical14 producer-byte gaps; fresh known-source
reference checks provide corroboration rather than recovering those scripts.

The fresh actual-result audit independently recomputed all720 E/F,1440
cross-engine comparisons and all ratios, plus36 oracle/grid checks. Eight CPU
controls passed: seven corruptions were rejected, and an untrusted stored-summary
spoof did not change the recomputed result. All1061 sealed originals stayed unchanged. Its scoped
fixed30 verdict is PASS; overall WARN retains the identity/scope limitations.
`actual-result-audit.json.xz` losslessly retains the full report. Review is
same-family/provisional (requested gpt-6.1-sol, actual backend model unverified).
This packet establishes the bounded frozen-CD2B matrix result; it does not
certify universal chemistry, other GPUs, current-main performance or the whole
project's unrelated electronic-property methods.

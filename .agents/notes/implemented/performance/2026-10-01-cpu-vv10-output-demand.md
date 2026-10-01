# Decision: omit unused CPU VV10 radial derivatives inside the finite envelope

Status: implemented; qualification recorded below
Date: 2026-10-01

## Problem

After the exact-zero-weight optimization in #1645, the CPU VV10 pair loop remains
most of the measured medium-grid WB97M-V energy endpoint. Every remaining pair
also calculates `dphi/dr²`, even though the molecular energy/potential consumer
has no geometry output and never reads that field. Its three divisions and
associated arithmetic exist only to preserve the old pair finiteness check.

## Decision

Specialize the existing pair evaluator on radial-derivative demand and choose
the matching pair-loop specialization once per execution, outside all pairs. Select the
reduced evaluator only when the **final, whole-domain** `mask_zero_weights`
finite-arithmetic envelope from #1645 is admitted. That already requires VV10,
weighted molecular feature outputs, no geometry request, and bounded finite
coordinates, scales, local derivatives, coefficients and weighted densities.
The molecular caller retains its separate AO-pullback envelope.

The reduced evaluator uses the identical `phi`, `dphi/domega` and `dphi/dkappa`
expressions, then leaves its unused, value-initialized radial field zero. No
reciprocal rewrite, reassociation, symmetry reduction or precision change is
introduced. Every retained row, partner and accumulation stays in the original
order. Pair evaluations remain A² for A active weights, and full-index weight
checks remain unchanged. The changed work is radial-derivative evaluations:
A² to zero in the admitted molecular E/V consumer.

The full evaluator remains the original scientific/failure owner outside that
envelope. In particular, raw fixed-grid feature requests, energy-only calls,
geometry/weight derivatives, rVV10 and out-of-envelope molecular inputs still
compute and validate every legacy pair field. CUDA is untouched. There is no
new API, array, allocation, cache, workspace admission or persistent state.

## Why omitting the failure-only computation is safe here

Let m=1e-12 and M=1e12 be the admitted lower/upper omega and kappa bounds.
Nonnegative finite r² gives gi, gj >= m. The previously established coordinate
bound also keeps their products and denominator finite and away from underflow.

- |phi| <= 3/(4m³) = 7.5e35
- omega_i/gi + omega_j/gj + (omega_i+omega_j)/(gi+gj) <= 3M/m = 3e24
- |dphi/dr²| <= 9M/(4m⁴) = 2.25e60

The implementation uses the conservative bound <3e60, leaving ample binary64
rounding margin. Each intermediate division, sum and product is finite. Thus
omitting this unused calculation cannot hide a failure inside the envelope.
The existing validation and full fallback are essential; checking only
`!want_geometry` or the caller opt-in is insufficient.

## Regression gates

The native fixture checks exact raw/reduced parity for all-active signed weights
at |coordinate|=1e6 with a steep but admitted gradient, across four tile sizes.
A companion zero-weight case observes 16 actual pairs rather than 25, proving
that the wide fixture really enters the reduced envelope.

A discriminating fallback fixture uses coincident points, rho=1,
gradient=(1e16,0,0), b=1e-70, C=.01 and zero weights. Its local scales, beta,
phi≈-3.81e208, dphi/dkappa≈2.12e278 and dphi/domega=0 are finite, while
`dphi/dr²` overflows. Both raw and opted-in feature calls must still report the
pair-kernel numerical failure and zero successful pair count. This would catch
an incorrectly broadened demand-only omission even though all requested E/V
outputs could otherwise remain finite.

Existing signed/subnormal weights, raw zero-row derivatives, geometry/rVV10,
energy-only, resource, inactive-input validation, overflow/recovery and finite
huge-AO failure regressions remain in force.

## Qualification

Baseline: merged #1645, master commit
`47d2f1938d128df3f007da7bfcaa89654777930f`, tree
`6d90ffa76f7af812f7058840701a3d288daa2f46`. The retained baseline library was
built from local commit `97b8d6ded09819f37cab7697c65ed48989f54877`, whose
complete tree is identical. These are incremental gains after both #1643 and
#1645, not a repeat of their earlier optimizations.

Both builds use GCC 14.2, RelWithDebInfo (`-O2`), identical CPU/AOT/OpenBLAS CMake
settings, and one OMP/OpenBLAS/MKL thread on a shared Intel Xeon Platinum 8573C runner (nine
exposed vCPUs, about 10 GB RAM). Three alternating baseline/candidate fresh-process
pairs per case each record first-call cold, two process-warm one-shot calls and
a 1% changed geometry. Every sample includes preparation and complete energy SCF;
imports, loading and builds are excluded. There is no plan/density reuse. No
builds, tests, oracles or profilers run during the final timing campaign.

| Case | Grid points | Warm baseline (s) | Warm candidate (s) | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Water/STO-3G, 12/4/8 | 1152 | 0.32888 | 0.31318 | 4.77% |
| Water/STO-3G, 16/6/12 | 3456 | 1.46888 | 1.20191 | 18.18% |
| Water/STO-3G, 24/8/16 | 9216 | 7.83543 | 5.92608 | 24.37% |
| Water/def2-SVP, 16/6/12 | 3456 | 5.13051 | 4.58757 | 10.58% |
| OH doublet UKS/STO-3G, 16/6/12 | 2304 | 1.18134 | 0.91864 | 22.24% |

Warm medians contain six samples per build. All 60 corresponding energies and
complete KS diagnostics match exactly, including iterations, Fock builds and grid
work. No sample was removed for a timing outlier. Shared-host variation remains;
these results describe only the measured cases, not a universal speedup.

| Case | Cold baseline -> candidate (s) | Changed geometry baseline -> candidate (s) |
| --- | ---: | ---: |
| Water/STO-3G, 12/4/8 | 0.32962 -> 0.32600 | 0.31157 -> 0.29706 |
| Water/STO-3G, 16/6/12 | 1.51135 -> 1.29445 | 1.64520 -> 1.16368 |
| Water/STO-3G, 24/8/16 | 9.44464 -> 5.67817 | 8.08160 -> 6.14887 |
| Water/def2-SVP, 16/6/12 | 5.25327 -> 4.89346 | 5.12381 -> 4.12417 |
| OH doublet UKS/STO-3G, 16/6/12 | 1.09145 -> 0.92619 | 1.11472 -> 0.92841 |

Cold and changed-geometry medians each contain three samples per build and improve
in every tested case. Cold means first endpoint after loading, not a filesystem
page-cache measurement. These explicit grids do not establish production-grid
quadrature convergence. No default-grid, large-molecule, multithread, GPU or
force-endpoint performance claim is made.

Native library SHA-256:

- Baseline: `dff79c6e144d0cb0e4add0aba92c882e500135e6b61fbf8a18e5282ba182fd92`
- Candidate: `aa4acae8250b28c7b6077e1c8420409972de9915d91dfe03a06d2e6038c7fb85`

Final-code validation:

- Native CTest: 57/57 passed
- Focused Python nonlocal/domain/composition gates: 57 passed, 6 CUDA skips
- Complete WB97M-V gradient and reconverged energy finite differences: 7 passed;
  correctness coverage only, not a force-performance claim
- Five independent PySCF 2.14.0 / Libxc 7.0.0 RKS/UKS fixtures: energy-at-density
  error <=3.1e-14 Eh, complete Fock error <=4.6e-14, density error <=8.6e-12
- Separate same-grid PySCF reconvergence for water/STO-3G and water/def2-SVP
  agrees within 1.6e-13 and 2.8e-13 Eh at 1e-12/1e-10 native tolerances
- Independent source/diff review found no blocking issue; full Python-suite and
  real GPU execution are not claimed

Separate instrumentation confirms that every case preserves pair and AO work:

| Case | VV10 calls | Pair kernels (both) | AO calls / AO-point pairs (both) |
| --- | ---: | ---: | ---: |
| Water/STO-3G, 12/4/8 | 11 | 8,504,480 | 165 / 266,112 |
| Water/STO-3G, 16/6/12 | 11 | 76,107,148 | 462 / 798,336 |
| Water/STO-3G, 24/8/16 | 11 | 545,150,324 | 1188 / 2,128,896 |
| Water/def2-SVP, 16/6/12 | 15 | 106,452,024 | 630 / 3,888,000 |
| OH doublet UKS/STO-3G, 16/6/12 | 17 | 51,863,225 | 459 / 705,024 |

Only the radial-derivative calculations disappear: the pair count in each row
above becomes the number of omitted `dphi/dr²` evaluations. The pair counter
still counts actual complete E/V pair evaluations, not a lowered pair count.
Instrumented component times are retained separately and are not substituted
for the complete endpoint medians. Memory accounting and dense capacity bounds
are unchanged.


### Publication rebase

The three-file change was subsequently rebased onto master
`737f3481fcc5a82039bd59b676f14d5e963a517d`, which adds #1644's separate CUDA
exchange-dispatch fix. None of its four changed files participates in this
CPU-only library's scientific compilation. The build did regenerate the embedded
source-identity string in `c_api_tuning.cpp`; its CPU branch only exposes the
identity string and returns NOT_IMPLEMENTED for CUDA tuning. This changes the
binary hash without changing CPU scientific sources, flags or generated kernels.
The timing table above deliberately retains the actual measured binary identity.
The rebased publication library hash is
`82b71748c4e32c78707c755eb538f208bad0b622b373ae83a8720bd07d4e71ff`.
After this rebase, native CTest again passes 57/57; the combined focused Python
selection passes 72 tests with 6 CUDA skips, including all 7 complete-gradient/FD
cases and the newly merged dispatch-owner regressions. All five final-head
nominal-geometry endpoint replays exactly match their measured candidate energy,
iteration and complete KS diagnostic. No new timing claim is inferred from the
replays. Architecture, complexity, ownership, evidence and format gates also pass.

### Reproduction

Build the baseline and candidate with identical settings, then use
`benchmarks/cpu_wb97mv_endpoint.py` in alternating fresh processes:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
GENERATIVEQC_LIBRARY=/absolute/baseline/libgenerativeqc.so \
  python benchmarks/cpu_wb97mv_endpoint.py --case water --basis sto-3g --grid 16 6 12
GENERATIVEQC_LIBRARY=/absolute/candidate/libgenerativeqc.so \
  python benchmarks/cpu_wb97mv_endpoint.py --case water --basis sto-3g --grid 16 6 12
```

Repeat three pairs, reverse order on alternate pairs, and repeat for the other
four cases in the tables. Check every paired energy and complete work diagnostic
before reporting medians. Run `generativeqc_wb97mv_scf_tests` with a JSONL output
path followed by `tools/verify_wb97mv_scf.py` for independent native SCF checks.
Raw JSON, build/test logs, profiler source and rejected pilot results are retained
in the local qualification directory; none of the pilots enter the final table.
No release, binary archive or external evidence host is created.


## Rejected alternatives

- Omitting all unrequested derivatives without a finite-domain proof changes
  legacy failure behavior, as the new discriminating fixture demonstrates
- Reciprocal reuse or pair symmetry can change floating-point arithmetic or
  reduction order and require a separate numerical/performance qualification
- Active-index compaction adds storage and resource policy; this patch removes
  only demonstrably unconsumed work in the existing bounded schedule

## References

- [Previous exact-zero-weight decision](2026-10-01-cpu-vv10-zero-weight-work.md)
- `benchmarks/cpu_wb97mv_endpoint.py`
- `tests/native/test_wb97mv_scf.cpp`
- `tools/verify_wb97mv_scf.py`

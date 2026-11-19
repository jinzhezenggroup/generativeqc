# H100: analytic forward-mode Becke partition derivative

The stationary DFT force differentiates the normalized Becke partition. The
phased route did this in reverse mode through a retained pair panel, in seven
launches per tile. The analytic route forms the same first derivative in
forward mode, with no pair panel, in five launches per tile:

1. **Products:** a (points, atom) lane forms each atom's product.
2. **Normalization:** one warp per point normalizes.
3. **Motion:** a (points, atom) lane sums `dP_B/dG` over every neighbor.
4. **Owner:** one warp per point gives the moving owner `-sum_G`.

Owners with phase storage now use it by default.
`GENERATIVEQC_STATIONARY_BECKE_ROUTE=phased` restores the pair phases.

In a README Direct r2SCAN-3c96 warm call, the Becke kernels take about 185 µs
per 1024-point tile. The phased route took about 410 µs, over 1052 tiles.

## Measurements

The runs used fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-opt-1118-qb`, all from one build.
- **Phased arms** set `GENERATIVEQC_STATIONARY_BECKE_ROUTE=phased`.
- **Analytic arms** leave it unset, which is the new default.
- **Main A/B:** an untimed prewarm pair per case, then two rounds in the order
  phased, analytic, analytic, phased.

| Case | phased cold (s) | analytic cold | phased warm | analytic warm |
| --- | --- | --- | --- | --- |
| Direct r2SCAN-3c96 | 9.635, 9.604 | 9.369, 9.366 | 3.769, 3.752 | 3.494, 3.489 |
| Direct PBE0 96 | 7.047, 7.065 | 6.728, 6.760 | 2.060, 2.061 | 1.760, 1.769 |
| DF B3LYP 96 | 5.144, 5.136 | 4.840, 4.863 | 1.516, 1.526 | 1.219, 1.223 |

The sweep covers every other README DFT method/mode combination at 96 and 48
atoms. Each case ran one phased process, then one analytic process. Cells give
phased → analytic, in seconds.

| Case | cold 96 | warm 96 | cold 48 | warm 48 |
| --- | --- | --- | --- | --- |
| Direct B3LYP | 7.126 → 6.874 | 2.081 → 1.799 | 3.921 → 3.874 | 0.788 → 0.739 |
| DF B3LYP | (main A/B) | (main A/B) | 2.437 → 2.425 | 0.428 → 0.382 |
| Direct M06-2X | 8.320 → 8.037 | 2.119 → 1.834 | 4.770 → 4.711 | 0.863 → 0.818 |
| DF M06-2X | 5.924 → 5.609 | 1.575 → 1.280 | 3.095 → 3.048 | 0.508 → 0.458 |
| Direct PBE | 4.498 → 4.199 | 1.355 → 1.065 | 2.641 → 2.590 | 0.470 → 0.422 |
| DF PBE | 4.188 → 3.897 | 1.169 → 0.872 | 2.415 → 2.335 | 0.390 → 0.344 |
| Direct PBE0 | (main A/B) | (main A/B) | 3.890 → 3.812 | 0.781 → 0.731 |
| DF PBE0 | 5.081 → 4.818 | 1.501 → 1.210 | 2.436 → 2.360 | 0.417 → 0.371 |
| Direct r2SCAN | 5.214 → 4.905 | 1.457 → 1.166 | 3.134 → 3.087 | 0.574 → 0.528 |
| DF r2SCAN | 4.858 → 4.602 | 1.274 → 0.982 | 2.858 → 2.826 | 0.496 → 0.454 |
| Direct r2SCAN-3c | (main A/B) | (main A/B) | 3.690 → 3.634 | 1.252 → 1.209 |
| DF r2SCAN-3c | 7.718 → 7.394 | 1.760 → 1.465 | 2.951 → 2.910 | 0.744 → 0.705 |
| Direct ωB97M-V | 33.981 → 33.375 | 4.865 → 4.261 | 13.276 → 13.228 | 2.527 → 2.477 |
| DF ωB97M-V | 21.341 → 20.768 | 5.582 → 4.982 | 8.890 → 8.839 | 1.880 → 1.818 |

Warm values are medians of three calls.
- **Counts and qualification:** all 68 processes complete and qualify against
  the frozen references. The two routes have identical SCF iteration and Fock
  counts in every case.
- **Force agreement:** same-case phased-vs-analytic forces agree within
  5.4e-13 Eh/Bohr for the equal-radius functionals. They agree within 1.2e-11
  for Treutler r2SCAN-3c, which matches the same-route process-to-process
  spread: 1.2e-11 phased and 7.8e-12 analytic.
- **ωB97M-V DF96:** its single pair differs by 2.2e-10. Its two processes also
  differ in energy by 1.9e-12 Eh, a difference the derivative route cannot
  produce.
- **Files:** `endpoint.json` holds every arm, sample, verdict and pairwise force
  difference. `results.json.xz` adds all force vectors and the full
  qualification records.

For context only, the 2026-11-18 three-process README cohort measured GPU4PySCF
at 3.676 s for Direct r2SCAN-3c96 warm. A new cohort, not this record, decides
the paper comparison.

## Gates

- **GPU tests (`gpu-tests.txt`):** the shared-owner, Treutler and binary-scaled
  owner tests pass on the measured source (798 cases), with probe owners built
  for sm_90.
  - Each owner compares the phased-storage owner, under both routes, with the
    bounded generic owner on identical inputs, at rtol 5e-12 and atol 2e-11.
  - It checks the route-specific work metrics, including the new
    `stationary_becke_analytic_metrics_v1`.
  - Six restricted-point cases construct an sm_120 owner and stop at the
    architecture check on this sm_90 device. They do not reach Becke work.
- **Sanitizers (`sanitizers.txt`):** compute-sanitizer memcheck, racecheck,
  initcheck and synccheck report 0 errors and 0 hazards on 96-atom analytic
  owners, both equal-radius and Treutler.

## Source

The measured tree is `a6d5a3c00` plus this change and several other changes:

- pre-merge versions of the #3115–#3120 changes;
- a rejected 32-load variant of the #3121 `geometry_reduce`;
- the #3122 512-thread RHF energy kernel.

It lacks #3083, the CPU Hessian. None of the extra changes touches the Becke
route, and both arms of every pair ran the same binary.

Against this head, the measured route sources differ only in two ways:

- clang-format whitespace in `stationary_gradient_cuda.cuh`;
- one lint-only string in `grid_native.py`, which generates byte-identical
  CUDA source.

A rebuild with that string, run as Direct r2SCAN-3c96, gives phased 9.758 /
3.761 s and analytic 9.318 / 3.482 s (cold / warm).

These measurements do not qualify any other GPU. The launch bound targets a
96-atom, 1024-point tile on a 132-SM H100.

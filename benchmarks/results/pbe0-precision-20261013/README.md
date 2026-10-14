# PBE0 application-precision endpoint measurements

Measured 2026-10-13 on Slurm node4, job 804, one RTX 5090 (32 GB), driver
580.178.04. This is a negative acceleration result, not FP32-gradient
qualification or a default-promotion benchmark.

## Identity and protocol

- Base: `e0ad22036fe83d9bcaed277d0be9a1dbcb453e1c` (then `origin/master`),
  plus the application-precision facade and its KS-constructor replay fix.
- Native source identity:
  `8e24cab8ead5a7aa2f15f317b520977bdc993dd85f754416622b35bf99208082`.
- Native library SHA-256:
  `c8cc3ee3213ee8e3aed44fea2d14026d9e103c5156e8d4dfe34d43fdd361314c`.
- CUDA 12.9.1, Release, official `sm_120` shell AOT, fast-compile OFF,
  native stationary CUDA ON, `pbe0_rks` stationary AOT, CPU derivative AOT ON.
  CMake uses ccache; `CMakeCache.txt.xz` retains the complete configuration.
- Spherical def2-SVP, PBE0/RKS, no density fitting. The shared benchmark cases
  fix the water monomer and WATER27 hydrogen-bonded tetramer geometries in Bohr.
- Moving quadrature: 48 rational-Legendre radial points, 16 polar nodes,
  32 azimuthal points; unit atomic radii, equal-radius Becke partition with
  three iterations, no pruning. This is a fixed discrete target, not a
  grid-convergence claim.
- Every endpoint starts from Hcore, not MINAO or a recycled converged density.
  Energy/density/screening controls are `1e-10` / `1e-8` / `1e-12`;
  the Fast audit tightens these to `1e-12` / `1e-10` / `1e-14`.
- Fresh calculator construction, SCF, complete analytic forces, host export
  and cache release are all timed with CUDA synchronization. Native production
  has no PySCF dependency; the reference runs separately on the CPU.
- First calls after the device/build probe are retained but excluded from the
  summary. Three subsequent alternating-order repeats use warm process caches,
  not warm densities. Imports, compilation and the metadata probe are outside
  timing. These small-case observations do not establish larger-system speed.

## Complete endpoint medians

| System | AOs | FP64 (s) | AUTO (s) | Fast including audit (s) | FP64/AUTO | FP64/Fast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Water | 24 | 2.160638 | 2.770549 | 5.002412 | 0.779859 | 0.431919 |
| Water tetramer | 96 | 6.124801 | 8.901217 | 15.513401 | 0.688086 | 0.394807 |

Ratios below one are regressions. AUTO takes about 28%/45% longer; Fast takes
2.32/2.53 times the baseline time. No 2–7x endpoint acceleration is observed.
The preserved samples include every first call and repeat, without filtering
slow or failed samples from a timing distribution.

## Work and arithmetic

| System | FP64 Focks | AUTO mixed + strict Focks | Fast candidate + audit Focks | FP32 recurrence evaluations per AUTO endpoint |
| --- | ---: | ---: | ---: | ---: |
| Water | 13 | 13 + 6 | 19 + 14 | 688675 |
| Water tetramer | 14 | 16 + 8 | 24 + 16 | 151540976 |

These counts repeat across the measured samples. Every ordinary endpoint
evaluates forces once; Fast evaluates them twice and performs two complete
SCFs. No budget rejection or third-solve fallback occurred. Its retained
`fp32_gradient_not_qualified` reason describes admission, not a numerical
failure of the candidate.

All public SCF precision inventories pass `require_public_precision`.
The only FP32 entries are `coulomb_j` and its `coulomb_recurrence`; these are
two levels of the same work, not independent work to sum. Storage,
accumulation and reduction remain FP64. The mixed recurrence scalar prevents
double literals from silently promoting its arithmetic. Gaussian preparation,
density products and primitive contraction sums also remain FP64.

Exchange K, XC point algebra, matrix products, DIIS, eigensolver and density
build records are FP64 in these executions. Although the schedule permits
qualified density-contraction lowering, it did not execute here. The SCF
inventory does not cover individual gradient instructions: source inspection
of the existing native derivative and stationary owners establishes that this
facade has not introduced a FP32 gradient route. No instruction-mix profile was
performed and no phase-specific speedup is inferred from the endpoint totals.

## Independent numerical gates

PySCF 2.11.0 independently reconstructs the same atomic quadrature and owns
integrals, SCF, XC, Becke partition and analytic grid response. It consumes the
same explicit basis coefficients, not native density or force results.
Its convergence controls are `1e-12` energy, `1e-10` orbital gradient and
`1e-14` direct screening. All native endpoints converge on CUDA and pass
`1e-7 Eh` energy and `1e-6 Eh/bohr` maximum force-component gates.

Across measured modes/repeats, maximum errors against this independent oracle
are `4.55e-13 Eh` and `3.17e-10 Eh/bohr`. AUTO versus ordinary native FP64
differs by at most `2.96e-10 Eh/bohr`. These are **post-FP64-refinement endpoint
errors**, not standalone FP32-kernel errors or transferable error bounds.

## Reproduction and retained failure

Run `python -m benchmarks.pbe0_precision --engine reference --output <path>`,
then run `--engine native --reference <path> --repeats 3 --output <path>` on
an allowed GPU node through finite Slurm `srun`. The measured allocation used
`--partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=4
--time=00:30:00`, with `OPENBLAS_NUM_THREADS=4` and `OMP_NUM_THREADS=4`.
Preserve scheduler-assigned device visibility. See the executable's docstring
and the application-precision guide for the full timing contract.

`native.json.xz` and `reference.json.xz` retain complete measurements and
scientific protocols. `native-failed-composition.json.xz` retains the earlier
failed run: Fast reconstructed a discovered PBE0 selector with resolved KS
composition outputs, which conflict with that selector's model ownership.
The facade now replays the immutable constructor input instead. That run has
no performance summary and is not mixed into job 804's measurements.

The older failed source identity is
`c7aee7e2f8545464c4aa826b5a6c1e23b4aaa2d114d74eaa4ce4296cc136fd5f`.
The constructor-only fix changes no native scientific kernel. Subsequent
CPU regression tests cover discovered PBE0/B3LYP and explicit MethodIR input;
job 804 exercises the repaired Fast constructor and full CUDA audit.

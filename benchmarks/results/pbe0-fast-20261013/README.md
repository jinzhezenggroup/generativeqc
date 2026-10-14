# PBE0 Fast: single strict endpoint and exact source reuse

Measured 2026-10-13 on node1, one Slurm-assigned RTX 5090, job **7980**.
These are complete energy plus analytic-force endpoints, not kernel timings.
The improvement is **FP64 work reuse**, not new FP32 arithmetic qualification.

## Source and build identity

- Base master: `4f4c1aeb0bd32baf4bbdda276947b7ce911c453d`, plus the retained
  application-precision implementation and the small-source policy change.
- Native source identity:
  `b72ce4fd6ac8046b6b35a8eeb4641ab44f65113b71320025ee9bc5012b7731a2`.
- Native library SHA-256:
  `c6cd6e18f6a00a0669fd51e4d74b435b22c99717299530b1c0ed3f2b90e2f7df`.
- Uncompressed `runtime.patch.xz` SHA-256:
  `4924ef22f75cd9f1eb802a06dcc3bc68c7af5b717ffc48a5a879e0798accad5f`.
  It binds native/public runtime changes and both benchmark drivers to master.
- `benchmarks/pbe0_fast.py` SHA-256:
  `172810f6e0fa4b3847215eca88093ebd6aa3caaa000b70ff6d66d6422c870636`.
- `python/generativeqc/precision_policy.py` SHA-256:
  `6d31d2d4027accd31af63c04140427930d62df71286623cbab230d66cf3f780d`.
- Release, CUDA 12.9.1, official SM120 AOT, fast-compile OFF; native stationary
  CUDA and PBE0 stationary AOT enabled. The device probe reports driver API
  13000 and runtime API 12080. Host compilation/linking uses node1 GCC 13.3;
  repeated builds use ccache. The full CMake cache is retained.
- The relocated build initially failed on node4-specific GCC 11 paths absent
  on node1. Compilation used node1's GCC 13 after correcting PATH; the CUDA
  host linker uses the retained `link_host.cmake.xz` through CMake's
  `CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA`. No failed build produced a timing
  sample. Local/remote build logs retain these failures; a fresh build with
  the local toolchain does not need that relocation override.

## Protocol and comparator

PBE0/RKS, spherical def2-SVP, no density fitting, Hcore initial density on
every solve. The moving grid is `(48,16,32)`, unit radii, three equal-radius
Becke iterations, no pruning. Energy/density/screening tolerances are
`1e-10`/`1e-8`/`1e-12`, with max_iterations 180. Scientific protocol and complete
results are in `native.json.xz`.

The reference is the unchanged independent PySCF 2.11.0 SCF and full moving-grid
analytic gradient in
[`../pbe0-precision-20261013/reference.json.xz`](../pbe0-precision-20261013/reference.json.xz).
Its uncompressed SHA-256 is
`f764b2350565182729fab3a58dac3b928baf6f781ecd7187f6bc1b9a0b833bcd`.
The driver compares the exact scientific protocol before using this older
reference; it does not relabel that record as a new reference calculation.
The oracle never supplies density, integrals or gradients to production.

Every timed endpoint includes calculator/facade construction, SCF, all analytic
force contributions, host export, cache release and CUDA synchronization.
Imports, the build/device probe and first calls are excluded from medians.
All six first calls are retained, followed by five alternating-order repeats
per mode/case (30 measured endpoints). Process caches are warm, densities are
not. There is no JIT compilation in the timed native force path.

`fp64_no_reuse` sets only `GENERATIVEQC_KS_RESIDENT_VALUES=0`; this reproduces
the former small-case schedule, whose source was rejected below 128 Cartesian
AOs (and, for water, the former 64-public-AO allowance). It is **not the new
FP64 default**. `fp64` is ordinary Calculator with the new bounded default;
`fast` is PrecisionCalculator without an explicit observable target. All three
retain the same convergence controls, precision and scientific model.

## Complete endpoint medians

| System | Public AOs | Former no-reuse FP64 (s) | Current ordinary FP64 (s) | Fast (s) | Former/Fast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water | 24 | 2.114368 | 1.758106 | 1.761276 | 1.200475 |
| Water tetramer | 96 | 6.120576 | 3.401016 | 3.394788 | 1.802933 |

Fast and current ordinary FP64 are effectively equal: the exact optimization
is shared rather than withheld from FP64 callers. Fast measured ranges are
1.758073–1.767908 s and 3.388953–3.433851 s. These small-case results are not a
universal speed promise, a larger-system benchmark or a 2–7x mixed-precision
claim. The old double-solve Fast records in the sibling directory stay unchanged.

## Scientific and work gates

All first calls and measured endpoints pass independent `1e-7 Eh` energy and
`1e-6 Eh/bohr` maximum force-component gates. Across Fast samples the largest
errors are `3.98e-13 Eh` and `3.15e-10 Eh/bohr`. Basis/grid/model error and
SCF-root stability are outside these discrete-target comparisons.

Every mode has one complete SCF, one force evaluation and a native final-state
check. All repeats have 13 Fock builds for water and 14 for the tetramer, zero
mixed builds, and no skipped final Fock work. The source-reuse modes construct
52,975 and 9,471,311 canonical ERI values once per endpoint, recorded in the
execution-owned `coulomb_recurrence` census, then replay them for J/K. Those
counts are source values, not primitive recurrence FLOPs. The ordinary
no-reuse path has no complete ERI-evaluation counter; do not invent one or
multiply these cache counts into a claimed number of avoided primitive calls.

Public precision inventories are complete and accepted. Storage, compute,
accumulation and reduction records are FP64. No FP32 operator executes in
these samples; gradient kernels are unchanged. Fast reports one solve with
`status=strict_fp64`, `scope=native_fp64`, `target=null` and empty error lists.
It does **not** claim `observed_met` without a runtime observable comparison.
An explicit target still requests the separate, costly audit.

## Other validation and retained screening

- Four Slurm tests on the final binary pass for the new 24-AO admission,
  strict/AUTO refinement and chunk widths 1/2. They cover independent signed
  RKS/UKS J/K, full/partial residency, PBE0/B3LYP returned-state energies,
  warm/frozen/external seeds, capacity refusal and screening guards.
- The separate complete resident PBE0 force test passes, including its guard
  against production CPU derivative fallback and independent gradient check.
- Slurm job **7982**, compute-sanitizer memcheck with full leak checking, reports
  **0 errors and 0 bytes leaked** for two complete small-source E+F endpoints.
  `memcheck.json.xz` timings are instrumented and excluded from speed claims.
- CPU/controller/benchmark tests: 127 passed, 2 optional skips; additional
  source-policy/shared-owner host checks: 15 passed, 4 GPU-only skips.
  Documentation/API tests: 67 passed, 72 subtests; Sphinx warnings-as-errors
  succeeds. GPU checks above are separate from those skips.
- `*-prehybrid-guard.*.xz` retain the first source's passing tests, memcheck,
  runtime patch and five-repeat benchmark (job 7977; native source
  `0fbcb792...`, library `d7f4e970...`). Its Fast speedups were 1.19x/1.77x.
  The final change limits the smaller threshold to hybrid KS, preserving pure-J
  defaults. All timings and GPU gates above were repeated on that final source.
  The first guarded-source probe attempt skipped because ccache was absent from
  PATH; `gpu-tests-missing-ccache.log.xz` preserves it. The corrected job runs
  all four tests, with no skips; skipped probes are not counted as passing.
- `screen-initial.json.xz` (job 7975) retains every exploratory variant on the
  preceding `8e24cab8...` native source/library, not the final binary. AUTO cap-2
  remains slower for the tetramer; forced source reuse wins. Final-validation
  and spatial-force selectors alone do not explain the improvement.
- `trace-initial.json.xz` and `trace-ledgers.tar.xz` (job 7976) retain the
  instrumented baseline diagnosis. Summed completed J/K scopes are about
  397/2929 ms for water/tetramer versus 21/285 ms for Direct J/K forces.
  Nested scopes are not additive, and these profiles are not clean timings.

## Reproduction

Build the native library and PBE0 stationary AOT artifacts with the settings
above. Select the matching library and source tree, then run on an available
allowed Slurm node (not maintenance node n3), preserving assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 \
  python -m benchmarks.pbe0_fast --repeats 5 \
  --reference .artifacts/pbe0-precision/reference.json \
  --output .artifacts/pbe0-fast/native.json
```

Set `OMP_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=4`. Leave the resident-value
override unset; the driver scopes the no-reuse control to its own samples.
Generate a fresh independent reference with `benchmarks.pbe0_precision
--engine reference` or decompress the matching retained reference. Changing
geometry, basis, grid, tolerances or guess requires a new matched experiment.

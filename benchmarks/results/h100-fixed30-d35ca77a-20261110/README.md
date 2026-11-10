# H100 original fixed30 on upstream master d35ca77a4

This is a fresh run of the original fixed30 complete energy/force cohort on
exact upstream master `d35ca77a4c7142f8d6d3f0cbc1e8e96718d9af30` (tree
`d443ede0341151ce282deb3f16315b6ee689196c`). There are no source edits and
no explicit SCF, force-tile, solver or scientific environment overrides.
**180/180 fresh processes and 720/720 complete E/F endpoints pass the original
independent gates. Cold: 28/30 wins, complete-set geometric mean 1.839.
Warm: 27/30 wins, geometric mean 1.598.**

| Phase | All 30 wins | All 30 geomean | Direct 16 wins / geomean | DF 14 wins / geomean |
| --- | ---: | ---: | ---: | ---: |
| cold | 28/30 | 1.8394 | 14/16, 1.7365 | 14/14, 1.9643 |
| warm | 27/30 | 1.5981 | 14/16, 1.7502 | 13/14, 1.4404 |

A win requires the median of three same-device, process-paired
GPU4PySCF/GenerativeQC warm-median ratios to exceed 1.05, every pair ratio to
exceed 1, and every cold and warm vector to pass science. Cold covers
constructor, prepare, SCF, analytic forces and host export after CUDA context
initialization. Process startup, imports and separately prepared compilation
are excluded. Warm uses three same-owner calls seeded from the engine's own
frozen cold density. No oracle seeding is used.

## Identity

- Native source identity: `4c94e7388a633cab91f157605d30593930091b05b7f807540fad25a937b608e7`
- Core SHA-256: `a406958243dc17b6a6b3a1aac5df065b0264ff13aa3d953b82a5489723803521`
- Build: a fresh CMake configuration with the same options as the earlier fixed30
  builds. These are the six stationary force AOT profiles (B3LYP, M06-2X, PBE0,
  PBE, r2SCAN, WB97M-V), `portable_cuda`, SM90, Release, and verified sccache
  0.16.0 for every compiler invocation. The SM90 AOT package audit passed, and
  25 artifacts are hash-bound in the build receipt.
- Device: one physical NVIDIA H100 80GB HBM3 with MIG disabled, UUID
  `GPU-54fef602-4cd8-f4fd-c0e4-6910f03f6ead`, driver 595.58.03, recorded
  700 W limit. No locked clocks are claimed. The job had 20 CPUs and 200 GiB of
  memory, ran with `OMP/OPENBLAS/MKL_NUM_THREADS=1`, and was one serial qz job.
- Reference arm: the pinned GPU4PySCF 1.9.0 / CuPy 13.4.1 reference site and
  installation catalog from the earlier fixed30. Its mapped reference images
  match the catalog with SHA-256
  `90b182945183c66d76876c901ee30592850baab87cf0b37d9c74d460a929d413`.
- Protocol: `fixed30-protocol-d35-v1.json`, SHA-256
  `5811ff6840df51ac6934f24bb6aae331794c007881ef903bfb01bc78d9aa0c92`.
  It keeps the original pair protocol
  `b4211299e942f6bf75cd9972cfeace38281474360fcad6d77c4b480758af385b` unchanged:
  the cases, the two original WB97M-V DF exclusions, the process order, the
  numerical gates (1e-5 Eh/atom, 1e-4 and 3e-5 Eh/Bohr), the timers, the
  independent oracles and the compiled-cache snapshot. It differs from the
  db078 fixed30 protocol only in two ways. The measured source is
  d35ca77a4. The WB97M-V96 Direct default tiles are recorded rather than
  required. All three native WB97M-V96 Direct processes resolved SCF tile 256
  and force tile 1024 without an explicit request.
- Execution: the native build ran for 3879 s, and the 180-process campaign took
  4034 s in the same job. Nothing was retried, and nothing failed.

## Per-case results

The db078 columns come from the earlier fixed30 on
`db0788b2032ba99029d1e7a7a995380777d4981f`, which ran on a different physical
H100 (`benchmarks/results/h100-cold-force-fixed30-20261105`). They are
descriptive context only and are not paired before/after evidence for any
change.

| Case | db078 cold median | d35 cold median | d35 cold | db078 warm median | d35 warm median | d35 warm |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| hf-water-48-direct | 1.736 | 1.348 | win | 3.773 | 4.332 | win |
| hf-water-48-df | 3.204 | 3.418 | win | 1.551 | 1.547 | win |
| hf-water-96-direct | 0.625 | 0.700 | **no** | 1.071 | 1.174 | **no** |
| hf-water-96-df | 1.038 | 1.099 | win | 0.843 | 0.834 | **no** |
| pbe-water-48-direct | 2.320 | 2.371 | win | 2.124 | 2.347 | win |
| pbe-water-48-df | 2.569 | 2.546 | win | 1.679 | 1.677 | win |
| pbe-water-96-direct | 1.684 | 1.843 | win | 1.201 | 1.362 | win |
| pbe-water-96-df | 1.690 | 1.870 | win | 1.236 | 1.268 | win |
| pbe0-water-48-direct | 2.629 | 2.801 | win | 2.387 | 2.619 | win |
| pbe0-water-48-df | 2.519 | 2.430 | win | 1.837 | 1.749 | win |
| pbe0-water-96-direct | 1.428 | 1.773 | win | 1.161 | 1.319 | win |
| pbe0-water-96-df | 1.422 | 1.513 | win | 1.134 | 1.168 | win |
| b3lyp-water-48-direct | 2.533 | 2.893 | win | 2.331 | 2.567 | win |
| b3lyp-water-48-df | 2.404 | 2.425 | win | 1.764 | 1.731 | win |
| b3lyp-water-96-direct | 1.475 | 1.853 | win | 1.127 | 1.309 | win |
| b3lyp-water-96-df | 1.465 | 1.534 | win | 1.128 | 1.149 | win |
| r2scan-water-48-direct | 2.367 | 2.309 | win | 2.181 | 2.397 | win |
| r2scan-water-48-df | 2.336 | 2.386 | win | 1.839 | 1.857 | win |
| r2scan-water-96-direct | 1.650 | 1.927 | win | 1.407 | 1.618 | win |
| r2scan-water-96-df | 1.656 | 1.950 | win | 1.456 | 1.579 | win |
| r2scan-3c-water-48-direct | 2.267 | 2.546 | win | 1.245 | 1.623 | win |
| r2scan-3c-water-48-df | 2.119 | 2.354 | win | 1.201 | 1.340 | win |
| r2scan-3c-water-96-direct | 1.039 | 1.301 | win | 0.725 | 0.938 | **no** |
| r2scan-3c-water-96-df | 1.123 | 1.375 | win | 1.161 | 1.309 | win |
| m06-2x-water-48-direct | 2.367 | 2.433 | win | 2.439 | 2.655 | win |
| m06-2x-water-48-df | 2.163 | 2.128 | win | 1.972 | 2.002 | win |
| m06-2x-water-96-direct | 1.356 | 1.789 | win | 1.315 | 1.562 | win |
| m06-2x-water-96-df | 1.392 | 1.624 | win | 1.383 | 1.442 | win |
| wb97m-v-water-48-direct | 0.680 | 1.121 | win | 0.797 | 1.266 | win |
| wb97m-v-water-96-direct | 0.517 | 0.868 | **no** | 0.894 | 1.327 | win |

These are the remaining nonwins.

- Cold: HF96 Direct, with ratios 0.616, 0.726 and 0.700, and WB97M-V96 Direct,
  with ratios 0.867, 0.868 and 0.868.
- Warm: HF96 Direct (median 1.174, but one pair ratio is 0.994, below 1),
  HF96 DF (median 0.834) and r2SCAN-3c96 Direct (median 0.938).

Full ratios, per-pair native and reference medians, cold SCF work counters and
qualification status are in `summary.json`.

## Offline replay

Use a Git clone that contains d35ca77a4, with unoptimized Python 3.12+ and
NumPy 2.2.6. The replay needs no CUDA, compiler, solver or network.

```sh
uv run --offline --no-env-file --no-project --python 3.12 --with numpy==2.2.6 \
  python benchmarks/results/h100-fixed30-d35ca77a-20261110/replay.py
```

`replay.py` authenticates all nine `records/` parts and the joined XZ stream
against `cohort.json`. It also authenticates all 583 original members: plans,
raw endpoint records, qualifications, the campaign, the build and AOT receipts,
the protocols, the frozen inputs, the oracles and the producers. It restores
the task layout in a temporary directory and materializes the pinned
benchmark helpers from local Git objects of d35ca77a4. It then reruns the
independent qualification and timing reduction (`analyze-fixed30-d35-v1.py`)
and requires exact equality with both the sealed reduction and `summary.json`.

## Limitations

This is a measurement of the current master on one physical H100. It is not a
production-default promotion, and it does not attribute the change against db078
to any single PR. Comparisons with other sources or devices are descriptive.
Full stdout/stderr streams and the native build tree are retained outside Git.

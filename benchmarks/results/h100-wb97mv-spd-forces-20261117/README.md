# H100: WB97M-V Direct forces on the density-first s/p/d schedule

Same protocol as [h100-wb97mv-cold-20261117](../h100-wb97mv-cold-20261117/README.md):
WB97M-V/def2-SVP water clusters, native Standard defaults versus stock GPU4PySCF
1.9.0, fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-wb97mv-nsys-1117-v1`, cold = constructor + preparation + SCF + analytic
forces (full grid response) + host export, warm = median of three
frozen-own-density calls. Every process (36) passes the original independent gates
(energy 1e-5 Eh/atom, force max 1e-4, RMS 3e-5 Eh/Bohr). `results.json.xz` retains
all processes and the two Nsight Systems kernel summaries.

## Attribution

Nsight Systems 2025.1.3 on the policy-v3 image (one cold + one warm E/F each):

| GPU kernel family | GenerativeQC | GPU4PySCF |
| --- | ---: | ---: |
| full-range J/K force derivatives | 6.49 s (generic bounded queue) | ~0.6 s total J/K gradient kernels |
| long-range K force derivatives | 2.99 s | (included above) |
| VV10 pair sweeps (SCF + force) | 9.02 s | 29.43 s |
| resident J/K replay | 6.17 s | — |
| range-source construction | ~9.4 s | — |

The WB97M-V full-range force (J + exact global-hybrid K) ran the generic bounded
queue because `qualified_spatial_force_default` excluded every range-separated or
nonlocal owner, and the RSH derivative consumer did not forward the qualification
to its full-range traversal.

## Explicit schedules (`2953b8ae8`)

| water-96 Direct | Cold (s) | Warm (s) |
| --- | ---: | ---: |
| incumbent generic traversal | 36.748, 36.553 | 7.486, 7.473 |
| `relative-density` (DensityReuse) | 34.053, 33.903 | 4.988, 4.979 |
| `relative-reuse` | 33.913 | 4.984 |
| `spatial-reuse` | 35.301 | 6.187 |

Water-48: 13.979 → 13.047 s cold, 3.414 → 2.549 s warm.

## Default (`30688f848`), paired with GPU4PySCF

| Pair (order) | GenerativeQC cold | GPU4PySCF cold | GPU4PySCF/native | Warm (native / reference) |
| --- | ---: | ---: | ---: | --- |
| r0 (native first) | 33.847 | 39.563 | 1.169 | 4.982 / 11.999 |
| r1 (reference first) | 33.991 | 39.613 | 1.165 | 4.999 / 12.235 |
| r2 (native first) | 33.933 | 39.528 | 1.165 | 4.961 / 11.875 |

Median paired cold ratio 1.165, every pair > 1. Water-48 Direct: 13.087 vs 18.053 s
cold, 2.540 vs 4.908 s warm. Water-12/3 (below the 384-AO policy floor) and DF
owners are unchanged. `wb-v10` retains an earlier commit whose default never
reached the RSH consumer: it reproduces the policy-v3 baseline (36.36–36.47 s versus
GPU4PySCF 37.77–37.96 s in the same cohort).

Remote raw records: `.../h100-cold/20261117-wb97mv/wb-v8` (Nsight reports),
`wb-v9`, `wb-v10`, `wb-v11`.

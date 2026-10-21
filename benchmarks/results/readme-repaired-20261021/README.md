# Repaired default-grid README comparison (2026-10-21)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **192 populations** use native source `cbd4d8074de35105717c8aac7cd1f86babbe90ec` and the same driver/binary: eight methods, Direct/DF, 3/12/48/96 atoms, native Standard/Fast and one GPU4PySCF default profile. There are **192 complete populations and 768 complete energy-plus-force calls**. CBS and moved-geometry timings are unchanged.

## Observed comparisons

Times are complete cold and median-of-three warm endpoints on n1 RTX 5090. Ratios below require both engines to pass every applicable independent sample gate. Cold has one observation; warm variation is within one process, with populations on independently allocated cards. These are scoped latency observations, not a statistically interleaved or universal ranking.

| Mode | Phase | Standard wins / qualified pairs | Fast wins / qualified pairs |
| --- | --- | ---: | ---: |
| direct | cold | 18 / 31 | 18 / 31 |
| direct | warm | 25 / 31 | 25 / 31 |
| df | cold | 11 / 27 | 12 / 27 |
| df | warm | 18 / 27 | 18 / 27 |

A “win” means a lower measured latency, without implying significance for small differences. All slower valid points remain in the figures.

| 96 atoms | Phase | Standard / GPU4PySCF (s) | Fast / GPU4PySCF (s) |
| --- | --- | ---: | ---: |
| direct wb97m-v | cold | 326.516 / 378.663 | 291.151 / 378.663 |
| direct wb97m-v | warm | 32.703 / 122.178 | 30.848 / 122.178 |
| df m06-2x | cold | 65.773 / 31.055 | 64.648 / 31.055 |
| df m06-2x | warm | 12.341 / 15.265 | 12.163 / 15.265 |
| direct m06-2x | cold | 73.160 / 24.474 | 42.450 / 24.474 |
| direct m06-2x | warm | 10.313 / 8.530 | 9.694 / 8.530 |
| df r2scan-3c | cold | 1648.702 / 26.340 | 1633.010 / 26.340 |
| df r2scan-3c | warm | 173.968 / 12.870 | 172.075 / 12.870 |

## Accuracy and complete accounting

| Profile | Attempted | Complete | Qualified | Failed accuracy | Unverified |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 64 | 64 | 58 | 6 | 0 |
| fast | 64 | 64 | 58 | 6 | 0 |
| gpu4pyscf | 64 | 64 | 58 | 6 | 0 |

Common gates remain energy 1e-5 Eh/atom, maximum force 1e-4 Eh/Bohr and force RMS 3e-5 Eh/Bohr. Each cold and warm sample is checked against independently converged direct references; reference-grid stability and matching CPU/GPU spots use quarter gates. Default HF DF fitting error and small M06-2X grid errors are retained as failures where observed; no grid, auxiliary or tolerance is changed to remove them. The timing table and exact receipts retain every population, including any incomplete endpoints.

## Source and reproduction

Core SHA-256: `d3f883dc0ae0de87ea7759cbb2dfbfc977d77f5a55f8c88ea8ff60d32b5c4e85`. The source commit adds the tested shell selection on #2661 and master `0e691d4000414ecb5fbf563232819178511e1642`, which was current when frozen. All 5,852 archived source/asset inputs match the measured implementation; the later Agent Note does not change runtime source. The separate experimental DF stationarity prototype is excluded. GCC 13.3, CUDA 12.9.1, Release/sm120, FastCompile OFF and verified sccache 0.16.0 are retained in provenance. Each population records its exact packages, GPU visibility, work and loaded native core.

Forty unchanged independent oracle receipts and their audited reuse contracts remain available for offline replay. `provenance.json.xz` retains source/binary/library hashes, exact launchers, compiler-cache statistics and original raw-receipt hashes. `diagnostics.json.xz` separately retains the source-bound shell comparison and 41 GPU / 5 Compute Sanitizer endpoint tests (zero memory errors). Diagnostic forced-memory timings never replace a default population.

```bash
PYTHONPATH=python:. python benchmarks/results/readme-repaired-20261021/reproduce.py \
  --output .artifacts/readme-repaired-replay
```

This recomputes every acceptance gate and redraws both 2×4 figures plus timing and work tables. Full-precision records retain every E/F sample, actual control and solve census. The superseded README campaign has a verified published Git snapshot; its historical ranking and failures remain recoverable.

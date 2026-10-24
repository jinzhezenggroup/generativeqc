# Existing one-electron schedule in the fitted DFT force bridge

Measured candidate: `c08d702293ec433321e9a97e48c3aa39f6a4f03d`.
Baseline: `af19ae37a09b965c560408f818b68c8aeb9073cd`.
The two standalone DFT Hcore/Pulay bridges now honor the existing shared
one-electron schedule policy. Unset selects nucleus-cooperative mapping;
explicit mappings 0–3, resident D/W ownership and bounded host fallback remain.
No basis, quadrature, screening, convergence, precision or derivative formula
changes. Both arms explicitly use 32 DF charge partitions.

Five alternating fresh-process pairs on one Slurm RTX 5090 execute complete
water96 energy and analytic forces, followed by three same-owner warm calls.
Cold includes construction, preparation, SCF, forces and host export. Each
process contributes one cold observation and one median warm observation.

| Case / phase | Before median / MAD (s) | After median / MAD (s) | Gain | Required floor | Qualified gain |
| --- | ---: | ---: | ---: | ---: | --- |
| PBE DF cold | 20.615377 / 0.040208 | 20.108672 / 0.034356 | 2.458% | 2.000% | yes |
| PBE DF warm | 6.573549 / 0.027916 | 6.242300 / 0.039684 | 5.039% | 2.121% | yes |
| r2SCAN-3c DF cold | 44.213351 / 0.585833 | 42.971903 / 0.269374 | 2.808% | 3.904% | no |
| r2SCAN-3c DF warm | 12.437754 / 0.229373 | 11.672623 / 0.360549 | 6.152% | 9.866% | no |

The floor is the unchanged maximum of 2% and twice the sum of relative MADs.
All observations, including the first slower PBE baseline and the variable 3c
samples, are retained. These results do not establish that PBE/3c cold catches
GPU4PySCF. The earlier single-process pilot included apparent PBE/r2SCAN
regressions and remains a separate cohort; its observations are not pooled with
these controlled pairs. Intrusive profiles are also separate from clean timing.

All **128 complete calls qualify independently**: 80 paired, 28 pilot, four
profile and 16 cold/warm/moved/moved-warm calls. Static qualification uses the
existing audited dense-grid and CPU/GPU references. Changed-geometry calls use
retained GPU4 DF controls at the exact displaced geometry (atom one z +0.001
Bohr); this checks sampled schedule parity, not independent displaced-grid
convergence. Maximum moved force error is 1.274e-5 Eh/Bohr (Fast 3c), below the
unchanged gate. Standard/Fast compatibility covers PBE, PBE0, B3LYP, r2SCAN,
r2SCAN-3c and M06-2X DF, plus PBE Direct.

Validation includes 15 host checks, 45 independent one-electron/owner GPU checks,
complete native Fock composition with all four paired mappings and zero Compute
Sanitizer errors. The frozen sm120 package and all five stationary manifests
were built using verified sccache 0.16.0 after source-input authentication.
The receipt bundle contains all original numerical arrays, semantic work,
source/library identities, full build/cache/test logs and runner scripts. Its
source inventory is a lossless authenticated delta against the retained charge-
partition inventory. Both baseline and candidate identities remain unchanged.

```bash
PYTHONPATH=python:. python benchmarks/results/df-one-electron-schedule-20261023/reproduce.py \
  > .artifacts/df-one-electron-reproduced.json
```

The script authenticates every source hash, rechecks all 128 calls and recomputes
all four timing populations. Nine older DF endpoint receipts, including rejected
schedules, are separately losslessly compacted under `../issue377-379-df/` to
retain this evidence within the unchanged aggregate storage limit. No numerical
receipt is dropped and no external archive or cap change is used.

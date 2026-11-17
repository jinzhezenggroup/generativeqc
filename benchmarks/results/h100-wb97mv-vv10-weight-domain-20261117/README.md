# H100: molecular VV10 weight domain (policy v3)

Same Job, H100 and protocol as
[h100-wb97mv-cold-20261117](../h100-wb97mv-cold-20261117/README.md): WB97M-V/def2-SVP
water clusters, native Standard defaults versus stock GPU4PySCF 1.9.0, fresh
consecutive processes on one physical H100 80GB, cold = constructor + preparation +
SCF + analytic forces (full grid response) + host export, warm = median of three
frozen-own-density calls. `results.json.xz` retains every process; the scripts and
arm lists stay with the raw records in the remote directory. Every process passes the
original independent gates (energy 1e-5 Eh/atom, force max 1e-4, RMS 3e-5 Eh/Bohr)
against the tracked readme-refresh oracles.

## Selector experiment (`b8edeb6a2`, CUDA domain kernel only)

| water-96 Direct | Cold (s) | Warm median (s) |
| --- | ---: | ---: |
| rho >= 1e-10 (policy v2) | 39.489, 39.278 | 7.481, 7.455 |
| rho >= 1e-10 and abs(w) > 1e-14 | 36.596, 36.588 | 7.480, 7.442 |

Both arms do 13 Focks and four pair sweeps. Paired v2/v3 processes give identical
total energies in all four calls and force differences of at most 9.4e-13 Eh/Bohr;
maximum independent errors agree to three significant figures. GPU4PySCF's own
domain on this grid has 1,048,696 points with rho >= 1e-10 and 955,804 after the
weight screen (8.9% fewer points, about 17% fewer pairs).

## Complete policy-v3 implementation (`2953b8ae8`)

| Pair (order) | GenerativeQC cold | GPU4PySCF cold | GPU4PySCF/native | Warm (native / reference) |
| --- | ---: | ---: | ---: | --- |
| r0 (native first) | 36.503 | 40.167 | 1.100 | 7.482 / 11.972 |
| r1 (reference first) | 36.502 | 37.659 | 1.032 | 7.468 / 11.539 |
| r2 (native first) | 36.530 | 38.139 | 1.044 | 7.479 / 11.603 |

Every pair is faster; the median paired cold ratio is 1.044, so the endpoint wins
but stays just below the 1.05 median margin used by the #3004 acceptance rule.
Other sizes:

| Case | Cold / warm (s) | Before (`8fcfd5825`) |
| --- | --- | --- |
| Direct water-48 | 14.034 / 3.415 (GPU4PySCF 17.300 / 4.618) | 14.749 / 3.416 |
| Direct water-12 | 2.186 / 0.364 | 2.197 / 0.361 |
| Direct water-3 | 1.437 / 0.111 | 1.429 / 0.110 |
| DF water-96 | 21.642 / 5.941 | 23.900 / 5.926 |
| DF water-48 | 8.800 / 1.962 | 9.398 / 1.961 |
| DF water-12 | 2.410 / 0.313 | — |
| DF water-3 | 0.862 / 0.104 | — |

The saving is concentrated in the SCF pair sweeps; warm calls are unchanged.

Remote raw records: `.../h100-cold/20261117-wb97mv/wb-v4` and `wb-v5`. Two queue
attempts that failed before any scientific process ran (a missing delta archive and
a missing arm list) are retained under `attempt-06-missing-delta/`.

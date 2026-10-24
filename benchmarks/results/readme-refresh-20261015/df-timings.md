# Complete DF · cc-pVDZ-JKFIT endpoint timings

| Functional | Atoms (AOs) | GenerativeQC cold (s) | GPU4PySCF cold (s) | GenerativeQC warm (s) | GPU4PySCF warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 (24) | 0.439 | 5.993 | 0.012 | 0.192 |
| HF | 12 (96) | 0.749 | 5.873 | 0.052 | 0.238 |
| HF | 48 (384) | 6.459 | 9.207 | 0.667 | 1.578 [1.566–1.656]† |
| HF | 96 (768) | 33.649 | 54.456 | 3.268 | 9.578 [9.546–10.577]† |
| PBE | 3 (24) | 2.984 | 7.934 | 0.101 | 0.287 |
| PBE | 12 (96) | 4.403 | 11.359 | 0.420 | 0.490 |
| PBE | 48 (384) | 17.090 | 29.073 | 3.167 | 3.376 |
| PBE | 96 (768) | 55.530 | 57.769 | 9.118 | 9.881 [9.864–12.924]† |
| PBE0 | 3 (24) | 3.038 | 6.719 | 0.110 | 0.340 |
| PBE0 | 12 (96) | 4.572 | 8.558 | 0.459 | 0.562 |
| PBE0 | 48 (384) | 20.629 | 25.346 | 5.449 | 4.069 [4.056–4.489]† |
| PBE0 | 96 (768) | 68.463 | 81.637 | 12.318 | 15.344 [15.285–18.372]† |
| B3LYP | 3 (24) | 3.065 | 5.084 | 0.115 | 0.307 |
| B3LYP | 12 (96) | 4.579 | 6.536 | 0.495 | 0.700 |
| B3LYP | 48 (384) | 21.032 | 28.956 | 5.560 | 4.046 |
| B3LYP | 96 (768) | 68.497 | 91.989 | 12.684 | 21.570 [18.545–23.318]† |
| r²SCAN | 3 (24) | 3.506 | 7.222 | 0.170 | 0.300 |
| r²SCAN | 12 (96) | 5.912 | 9.076 | 0.626 | 0.557 |
| r²SCAN | 48 (384) | 26.829 | 32.990 | 4.416 | 4.714 |
| r²SCAN | 96 (768) | 77.058 | 64.152 | 11.849 | 12.671 [10.721–13.711]† |
| r²SCAN-3c | 3 (34) | 3.066 | 5.333 | 0.131 | 0.312 |
| r²SCAN-3c | 12 (136) | 6.713 | 8.161 | 0.715 | 0.656 |
| r²SCAN-3c | 48 (544) | 42.740 | 52.500 | 5.409 | 7.541 [7.521–8.288]† |
| r²SCAN-3c | 96 (1088) | 165.817 | 117.146 | 16.865 | 18.356 [18.337–24.735]† |
| M06-2X | 3 (24) | 3.622 | 4.750 | 0.165 | 0.311 |
| M06-2X | 12 (96) | 6.182 | 6.940 | 0.591 | 0.591 |
| M06-2X | 48 (384) | 30.991 | 29.255 | 6.225 | 4.239 |
| M06-2X | 96 (768) | 94.280 | 95.234 | 14.122 | 22.421 [21.964–28.877]† |
| ωB97M-V | 3 (24) | 7.721 | 18.880 | 0.377 | 1.665 |
| ωB97M-V | 12 (96) | 28.409 | 98.457 | 1.395 | 7.824 |
| ωB97M-V | 48 (384) | 330.504‡ | not run | 14.499‡ | not run |
| ωB97M-V | 96 (768) | 4717.471‡ | not run | timeout | not run |

Warm cells show the median of all five complete endpoints. † marks differing
SCF work and includes the full range; timings are never normalized by iterations.
‡ marks completed native calls without an independent accuracy check for that geometry.
A later timeout does not discard completed phases; incomplete warm repeat sets are not plotted.
r²SCAN-3c: native CPU gCP; reference CPU D4 + gCP.
ωB97M-V: DF/48,96: accuracy unchecked
96: DF 12 GiB; forces 4/4 GiB.

## Observed SCF work

Each cell lists iterations/builds. Builds are native Fock builds or reference
SCF J/K calls, including its initial potential. Multiple pairs retain all
observed warm branches. Counts do not imply equivalent work or stopping rules
between engines; complete endpoint time includes analytic forces as well.

| Functional | Atoms | GenerativeQC cold | GPU4PySCF cold | GenerativeQC warm | GPU4PySCF warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 | 15/— | 42/43 | 1/— | 1/2 |
| HF | 12 | 19/— | 49/50 | 1/— | 1/2 |
| HF | 48 | 18/— | 44/45 | 1/— | 1/2, 2/3 |
| HF | 96 | 18/— | 48/49 | 1/— | 1/2, 2/3 |
| PBE | 3 | 12/12 | 45/46 | 1/1 | 1/2 |
| PBE | 12 | 17/17 | 66/67 | 1/1 | 1/2 |
| PBE | 48 | 17/17 | 59/60 | 1/1 | 4/5 |
| PBE | 96 | 19/19 | 61/62 | 1/1 | 4/5, 6/7, 8/9 |
| PBE0 | 3 | 12/12 | 29/30 | 1/1 | 1/2 |
| PBE0 | 12 | 18/18 | 32/33 | 1/1 | 1/2 |
| PBE0 | 48 | 17/17 | 43/44 | 1/1 | 3/4, 4/5 |
| PBE0 | 96 | 17/17 | 42/43 | 1/1 | 1/2, 3/4 |
| B3LYP | 3 | 12/12 | 40/41 | 1/1 | 1/2 |
| B3LYP | 12 | 17/17 | 36/37 | 1/1 | 3/4 |
| B3LYP | 48 | 17/17 | 53/54 | 1/1 | 3/4 |
| B3LYP | 96 | 16/16 | 48/49 | 1/1 | 3/4, 4/5, 5/6, 6/7 |
| r²SCAN | 3 | 12/12 | 35/36 | 1/1 | 1/2 |
| r²SCAN | 12 | 17/17 | 40/41 | 1/1 | 1/2 |
| r²SCAN | 48 | 17/17 | 53/54 | 1/1 | 4/5 |
| r²SCAN | 96 | 17/17 | 52/53 | 1/1 | 1/2, 3/4, 4/5 |
| r²SCAN-3c | 3 | 13/13 | 38/39 | 1/1 | 1/2 |
| r²SCAN-3c | 12 | 18/18 | 42/43 | 1/1 | 1/2 |
| r²SCAN-3c | 48 | 19/19 | 59/60 | 1/1 | 4/5, 5/6 |
| r²SCAN-3c | 96 | 24/24 | 60/61 | 1/1 | 1/2, 5/6 |
| M06-2X | 3 | 13/13 | 35/36 | 1/1 | 1/2 |
| M06-2X | 12 | 19/19 | 39/40 | 1/1 | 1/2 |
| M06-2X | 48 | 18/18 | 41/42 | 1/1 | 1/2 |
| M06-2X | 96 | 18/18 | 43/44 | 1/1 | 3/4, 4/5, 7/8 |
| ωB97M-V | 3 | 23/23 | 32/33 | 1/1 | 1/2 |
| ωB97M-V | 12 | 36/36 | 45/46 | 1/1 | 1/2 |
| ωB97M-V | 48 | 36/36 | — | 1/1 | — |
| ωB97M-V | 96 | 37/37 | — | — | — |

The [measurement report](README.md) describes the protocol and unsuccessful
points; [df-summary.json.xz](df-summary.json.xz) retains all scalar observations,
residuals, errors and independent cold/moved reference forces.

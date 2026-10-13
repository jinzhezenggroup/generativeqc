# Complete DF · cc-pVDZ-JKFIT endpoint timings

| Functional | Atoms (AOs) | GenerativeQC cold (s) | GPU4PySCF cold (s) | GenerativeQC warm (s) | GPU4PySCF warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 (24) | 0.416 | 5.993 | 0.012 | 0.192 |
| HF | 12 (96) | 0.662 | 5.873 | 0.052 | 0.238 |
| HF | 48 (384) | 6.196 | 9.207 | 0.655 | 1.578 [1.566–1.656]† |
| HF | 96 (768) | 32.994 | 54.456 | 3.201 | 9.578 [9.546–10.577]† |
| PBE | 3 (24) | 3.053 | 7.934 | 0.118 | 0.287 |
| PBE | 12 (96) | 5.071 | 11.359 | 0.534 | 0.490 |
| PBE | 48 (384) | 16.576 | 29.073 | 3.114 | 3.376 |
| PBE | 96 (768) | 53.891 | 57.769 | 9.884 | 9.881 [9.864–12.924]† |
| PBE0 | 3 (24) | 2.985 | 6.719 | 0.129 | 0.340 |
| PBE0 | 12 (96) | 4.914 | 8.558 | 0.570 | 0.562 |
| PBE0 | 48 (384) | 22.850 | 25.346 | 5.288 | 4.069 [4.056–4.489]† |
| PBE0 | 96 (768) | 77.368 | 81.637 | 18.977 | 15.344 [15.285–18.372]† |
| B3LYP | 3 (24) | 2.731 | 5.084 | 0.118 | 0.307 |
| B3LYP | 12 (96) | 4.370 | 6.536 | 0.523 | 0.700 |
| B3LYP | 48 (384) | 20.794 | 28.956 | 5.587 | 4.046 |
| B3LYP | 96 (768) | 68.212 | 91.989 | 19.975 | 21.570 [18.545–23.318]† |
| r²SCAN | 3 (24) | 3.618 | 7.222 | 0.199 | 0.300 |
| r²SCAN | 12 (96) | 6.451 | 9.076 | 0.699 | 0.557 |
| r²SCAN | 48 (384) | 32.519 | 32.990 | 4.769 | 4.714 |
| r²SCAN | 96 (768) | 106.578 | 64.152 | 14.213 | 12.671 [10.721–13.711]† |
| r²SCAN-3c | 3 (34) | 2.784 | 5.333 | 0.141 | 0.312 |
| r²SCAN-3c | 12 (136) | 6.586 | 8.161 | 0.805 | 0.656 |
| r²SCAN-3c | 48 (544) | 50.312 | 52.500 | 6.164 | 7.541 [7.521–8.288]† |
| r²SCAN-3c | 96 (1088) | 226.247 | 117.146 | 23.080 | 18.356 [18.337–24.735]† |
| M06-2X | 3 (24) | 3.486 | 4.750 | 0.219 | 0.311 |
| M06-2X | 12 (96) | 6.467 | 6.940 | 1.031 | 0.591 |
| M06-2X | 48 (384) | 45.783 | 29.255 | 8.920 | 4.239 |
| M06-2X | 96 (768) | 150.082 | 95.234 | 36.357 | 22.421 [21.964–28.877]† |
| ωB97M-V | 3 (24) | 38.146 | 18.880 | 0.735 | 1.665 |
| ωB97M-V | 12 (96) | 51.540 | 98.457 | 5.789 | 7.824 |

Warm cells show the median of all five complete endpoints. † marks differing
SCF work and includes the full range; timings are never normalized by iterations.
‡ marks completed native calls without an independent accuracy check for that geometry.
A later timeout does not discard completed phases; incomplete warm repeat sets are not plotted.
r²SCAN-3c: native CPU gCP; reference CPU D4 + gCP.

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
| PBE | 3 | 16/16 | 45/46 | 1/1 | 1/2 |
| PBE | 12 | 24/24 | 66/67 | 1/1 | 1/2 |
| PBE | 48 | 17/17 | 59/60 | 1/1 | 4/5 |
| PBE | 96 | 18/18 | 61/62 | 1/1 | 4/5, 6/7, 8/9 |
| PBE0 | 3 | 15/15 | 29/30 | 1/1 | 1/2 |
| PBE0 | 12 | 23/23 | 32/33 | 1/1 | 1/2 |
| PBE0 | 48 | 23/23 | 43/44 | 1/1 | 3/4, 4/5 |
| PBE0 | 96 | 24/24 | 42/43 | 3/3 | 1/2, 3/4 |
| B3LYP | 3 | 12/12 | 40/41 | 1/1 | 1/2 |
| B3LYP | 12 | 17/17 | 36/37 | 1/1 | 3/4 |
| B3LYP | 48 | 17/17 | 53/54 | 1/1 | 3/4 |
| B3LYP | 96 | 16/16 | 48/49 | 3/3 | 3/4, 4/5, 5/6, 6/7 |
| r²SCAN | 3 | 15/15 | 35/36 | 1/1 | 1/2 |
| r²SCAN | 12 | 23/23 | 40/41 | 1/1 | 1/2 |
| r²SCAN | 48 | 17/17 | 53/54 | 1/1 | 4/5 |
| r²SCAN | 96 | 18/18 | 52/53 | 1/1 | 1/2, 3/4, 4/5 |
| r²SCAN-3c | 3 | 13/13 | 38/39 | 1/1 | 1/2 |
| r²SCAN-3c | 12 | 18/18 | 42/43 | 1/1 | 1/2 |
| r²SCAN-3c | 48 | 19/19 | 59/60 | 1/1 | 4/5, 5/6 |
| r²SCAN-3c | 96 | 25/25 | 60/61 | 1/1 | 1/2, 5/6 |
| M06-2X | 3 | 13/13 | 35/36 | 1/1 | 1/2 |
| M06-2X | 12 | 19/19 | 39/40 | 1/1 | 1/2 |
| M06-2X | 48 | 18/18 | 41/42 | 1/1 | 1/2 |
| M06-2X | 96 | 18/18 | 43/44 | 3/3 | 3/4, 4/5, 7/8 |
| ωB97M-V | 3 | 13/13 | 32/33 | 1/1 | 1/2 |
| ωB97M-V | 12 | 19/19 | 45/46 | 1/1 | 1/2 |

The [measurement report](README.md) describes the protocol and unsuccessful
points; [df-summary.json.xz](df-summary.json.xz) retains all scalar observations,
residuals, errors and independent cold/moved reference forces.

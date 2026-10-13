# Complete DF · cc-pVDZ-JKFIT endpoint timings

| Functional | Atoms (AOs) | GenerativeQC cold (s) | GPU4PySCF cold (s) | GenerativeQC warm (s) | GPU4PySCF warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 (24) | 0.464 | 5.993 | 0.012 | 0.192 |
| HF | 12 (96) | 0.733 | 5.873 | 0.054 | 0.238 |
| HF | 48 (384) | 6.509 | 9.207 | 0.673 | 1.578 [1.566–1.656]† |
| HF | 96 (768) | 165.501 | 54.456 | 59.318 | 9.578 [9.546–10.577]† |
| PBE | 3 (24) | 3.053 | 7.934 | 0.118 | 0.287 |
| PBE | 12 (96) | 5.071 | 11.359 | 0.534 | 0.490 |
| PBE | 48 (384) | 21.634 | 29.073 | 3.170 | 3.376 |
| PBE | 96 (768) | 1225.048 | 57.769 | timeout | 9.881 [9.864–12.924]† |
| PBE0 | 3 (24) | 2.985 | 6.719 | 0.129 | 0.340 |
| PBE0 | 12 (96) | 4.914 | 8.558 | 0.570 | 0.562 |
| PBE0 | 48 (384) | 22.850 | 25.346 | 5.288 | 4.069 [4.056–4.489]† |
| PBE0 | 96 (768) | 77.368 | 81.637 | 18.977 | 15.344 [15.285–18.372]† |
| r²SCAN | 3 (24) | 3.618 | 7.222 | 0.199 | 0.300 |
| r²SCAN | 12 (96) | 6.451 | 9.076 | 0.699 | 0.557 |
| r²SCAN | 48 (384) | 42.475 | 32.990 | 4.719 | 4.714 |
| r²SCAN | 96 (768) | 1221.951 | 64.152 | interrupted | 12.671 [10.721–13.711]† |

Warm cells show the median of all five complete endpoints. † marks differing
SCF work and includes the full range; timings are never normalized by iterations.
‡ marks completed native calls without an independent accuracy check for that geometry.
A later timeout does not discard completed phases; incomplete warm repeat sets are not plotted.

## Observed SCF work

Each cell lists iterations/builds. Builds are native Fock builds or reference
SCF J/K calls, including its initial potential. Multiple pairs retain all
observed warm branches. Counts do not imply equivalent work or stopping rules
between engines; complete endpoint time includes analytic forces as well.

| Functional | Atoms | GenerativeQC cold | GPU4PySCF cold | GenerativeQC warm | GPU4PySCF warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 | 17/— | 42/43 | 1/— | 1/2 |
| HF | 12 | 23/— | 49/50 | 1/— | 1/2 |
| HF | 48 | 21/— | 44/45 | 1/— | 1/2, 2/3 |
| HF | 96 | 24/— | 48/49 | 3/— | 1/2, 2/3 |
| PBE | 3 | 16/16 | 45/46 | 1/1 | 1/2 |
| PBE | 12 | 24/24 | 66/67 | 1/1 | 1/2 |
| PBE | 48 | 28/28 | 59/60 | 1/1 | 4/5 |
| PBE | 96 | 29/29 | 61/62 | — | 4/5, 6/7, 8/9 |
| PBE0 | 3 | 15/15 | 29/30 | 1/1 | 1/2 |
| PBE0 | 12 | 23/23 | 32/33 | 1/1 | 1/2 |
| PBE0 | 48 | 23/23 | 43/44 | 1/1 | 3/4, 4/5 |
| PBE0 | 96 | 24/24 | 42/43 | 3/3 | 1/2, 3/4 |
| r²SCAN | 3 | 15/15 | 35/36 | 1/1 | 1/2 |
| r²SCAN | 12 | 23/23 | 40/41 | 1/1 | 1/2 |
| r²SCAN | 48 | 25/25 | 53/54 | 1/1 | 4/5 |
| r²SCAN | 96 | 26/26 | 52/53 | — | 1/2, 3/4, 4/5 |

The [measurement report](README.md) describes the protocol and unsuccessful
points; [df-summary.json.xz](df-summary.json.xz) retains all scalar observations,
residuals, errors and independent cold/moved reference forces.

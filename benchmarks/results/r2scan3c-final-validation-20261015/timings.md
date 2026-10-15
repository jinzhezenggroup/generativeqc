# Complete Direct endpoint timings

| Functional | Atoms (AOs) | GenerativeQC cold (s) | GPU4PySCF cold (s) | GenerativeQC warm (s) | GPU4PySCF warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 (24) | 1.662 | 25.519 | 0.071 | 1.547 |
| HF | 12 (96) | 2.384 | 25.056 | 0.116 | 1.515 |
| HF | 48 (384) | 11.501 | 33.437 | 0.736 | 2.203 [2.188–2.879]† |
| HF | 96 (768) | 39.107 | 50.568 | 2.904 | 3.851 |
| PBE | 3 (24) | 3.147 | 6.191 | 0.130 | 0.493 |
| PBE | 12 (96) | 6.738 | 14.562 | 0.851 | 0.835 |
| PBE | 48 (384) | 21.865 | 41.748 | 4.166 | 4.961 [4.942–5.559]† |
| PBE | 96 (768) | 64.434 | 76.964 | 13.454 | 14.009 [11.962–15.165]† |
| PBE0 | 3 (24) | 3.966 | 19.067 | 0.159 | 1.450 |
| PBE0 | 12 (96) | 7.888 | 22.992 | 0.934 | 1.643 |
| PBE0 | 48 (384) | 24.878 | 50.648 | 4.604 | 6.149 [5.925–8.156]† |
| PBE0 | 96 (768) | 69.710 | 76.978 | 13.259 | 13.547 [13.404–15.359]† |
| B3LYP | 3 (24) | 4.027 | 22.189 | 0.154 | 1.287 |
| B3LYP | 12 (96) | 8.471 | 21.627 | 0.886 | 2.287 |
| B3LYP | 48 (384) | 25.785 | 61.729 | 4.755 | 5.927 |
| B3LYP | 96 (768) | 76.688 | 96.895 | 14.137 | 13.471 [13.397–16.973]† |
| r²SCAN | 3 (24) | 4.081 | 7.154 | 0.196 | 0.608 |
| r²SCAN | 12 (96) | 8.021 | 11.953 | 1.012 | 1.032 |
| r²SCAN | 48 (384) | 37.679 | 46.129 | 5.453 | 6.378 [6.359–7.116]† |
| r²SCAN | 96 (768) | 118.007 | 75.886 | 17.154 | 14.594 [11.992–15.876]† |
| r²SCAN-3c | 3 (34) | 1.426 | 20.882 | 0.187 | 0.867 |
| r²SCAN-3c | 12 (136) | 9.216 | 31.155 | 1.772 | 1.708 |
| r²SCAN-3c | 48 (544) | 83.937 | 93.431 | 21.951 | 11.364 |
| r²SCAN-3c | 96 (1088) | 456.842 | 168.718 | 88.312 | 31.250 [29.020–35.760]† |
| M06-2X | 3 (24) | 4.187 | 15.205 | 0.260 | 1.051 |
| M06-2X | 12 (96) | 7.919 | 25.024 | 1.430 | 1.775 |
| M06-2X | 48 (384) | 46.884 | 54.767 | 6.824 | 5.294 [4.994–7.289]† |
| M06-2X | 96 (768) | 150.704 | 97.687 | 20.014 | 13.406 |
| ωB97M-V | 3 (24) | 7.299 | 27.917 | 0.386 | 2.575 |
| ωB97M-V | 12 (96) | 30.861 | 109.918 | 1.641 | 8.837 |
| ωB97M-V | 48 (384) | 255.233 | 1216.489 | 11.444 | 105.410 |
| ωB97M-V | 96 (768) | 1336.273 | timeout | 59.897 | timeout |

Warm cells show the median of all five complete endpoints. † marks differing
SCF work and includes the full range; timings are never normalized by iterations.
‡ marks completed native calls without an independent accuracy check for that geometry.
A later timeout does not discard completed phases; incomplete warm repeat sets are not plotted.
r²SCAN-3c: native CPU gCP; reference CPU D4 + gCP.
ωB97M-V: 3/12: native cache primed.

## Observed SCF work

Each cell lists iterations/builds. Builds are native Fock builds or reference
SCF J/K calls, including its initial potential. Multiple pairs retain all
observed warm branches. Counts do not imply equivalent work or stopping rules
between engines; complete endpoint time includes analytic forces as well.

| Functional | Atoms | GenerativeQC cold | GPU4PySCF cold | GenerativeQC warm | GPU4PySCF warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 | 19/— | 42/43 | 1/— | 1/2 |
| HF | 12 | 27/— | 49/50 | 1/— | 1/2 |
| HF | 48 | 24/— | 44/45 | 1/— | 1/2, 2/3 |
| HF | 96 | 25/— | 44/45 | 1/— | 1/2 |
| PBE | 3 | 13/13 | 45/46 | 1/1 | 1/2 |
| PBE | 12 | 18/18 | 66/67 | 1/1 | 1/2 |
| PBE | 48 | 17/18 | 59/60 | 1/1 | 4/5, 5/6 |
| PBE | 96 | 18/19 | 61/62 | 1/1 | 4/5, 5/6, 6/7, 7/8 |
| PBE0 | 3 | 12/12 | 29/30 | 1/1 | 1/2 |
| PBE0 | 12 | 16/16 | 32/33 | 1/1 | 1/2 |
| PBE0 | 48 | 19/20 | 43/44 | 1/1 | 3/4, 4/5 |
| PBE0 | 96 | 17/18 | 37/38 | 1/1 | 3/4, 4/5 |
| B3LYP | 3 | 12/12 | 40/41 | 1/1 | 1/2 |
| B3LYP | 12 | 18/18 | 36/37 | 1/1 | 3/4 |
| B3LYP | 48 | 19/20 | 54/55 | 1/1 | 3/4 |
| B3LYP | 96 | 18/19 | 48/49 | 1/1 | 3/4, 5/6 |
| r²SCAN | 3 | 12/12 | 35/36 | 1/1 | 1/2 |
| r²SCAN | 12 | 18/18 | 40/41 | 1/1 | 1/2 |
| r²SCAN | 48 | 17/18 | 53/54 | 1/1 | 4/5, 5/6 |
| r²SCAN | 96 | 18/19 | 47/48 | 1/1 | 1/2, 3/4, 4/5 |
| r²SCAN-3c | 3 | 13/13 | 38/39 | 1/1 | 1/2 |
| r²SCAN-3c | 12 | 18/19 | 42/43 | 1/1 | 1/2 |
| r²SCAN-3c | 48 | 19/19 | 59/60 | 1/1 | 4/5 |
| r²SCAN-3c | 96 | 26/26 | 59/60 | 1/1 | 4/5, 5/6, 7/8 |
| M06-2X | 3 | 13/14 | 35/36 | 1/1 | 1/2 |
| M06-2X | 12 | 19/20 | 39/40 | 1/1 | 1/2 |
| M06-2X | 48 | 18/19 | 41/42 | 1/1 | 1/2, 2/3, 3/4 |
| M06-2X | 96 | 18/19 | 41/42 | 1/1 | 1/2 |
| ωB97M-V | 3 | 23/24 | 32/33 | 1/1 | 1/2 |
| ωB97M-V | 12 | 36/37 | 45/46 | 1/1 | 1/2 |
| ωB97M-V | 48 | 36/37 | 45/46 | 1/1 | 1/2 |
| ωB97M-V | 96 | 36/36 | — | 1/1 | — |

The [measurement report](README.md) describes the protocol and unsuccessful
points; [summary.json.xz](summary.json.xz) retains all scalar observations,
residuals, errors and independent cold/moved reference forces.

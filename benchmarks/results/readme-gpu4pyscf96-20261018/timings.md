# Complete Direct endpoint timings

| Functional | Atoms (AOs) | GenerativeQC cold (s) | GPU4PySCF cold (s) | GenerativeQC warm (s) | GPU4PySCF warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 (24) | 1.734 | 25.519 | 0.072 | 1.547 |
| HF | 12 (96) | 2.310 | 25.056 | 0.117 | 1.515 |
| HF | 48 (384) | 10.918 | 33.437 | 0.742 | 2.203 [2.188–2.879]† |
| HF | 96 (768) | 35.465 | 50.568 | 2.912 | 3.851 |
| PBE | 3 (24) | 1.441 | 6.191 | 0.101 | 0.493 |
| PBE | 12 (96) | 4.373 | 14.562 | 0.642 | 0.835 |
| PBE | 48 (384) | 21.379 | 41.748 | 4.020 | 4.961 [4.942–5.559]† |
| PBE | 96 (768) | 60.517 | 76.964 | 12.130 | 14.009 [11.962–15.165]† |
| PBE0 | 3 (24) | 1.825 | 19.067 | 0.138 | 1.450 |
| PBE0 | 12 (96) | 3.662 | 22.992 | 0.765 | 1.643 |
| PBE0 | 48 (384) | 23.808 | 50.648 | 4.623 | 6.149 [5.925–8.156]† |
| PBE0 | 96 (768) | 80.438 | 76.978 | 13.592 | 13.547 [13.404–15.359]† |
| B3LYP | 3 (24) | 1.885 | 22.189 | 0.147 | 1.287 |
| B3LYP | 12 (96) | 3.729 | 21.627 | 0.823 | 2.287 |
| B3LYP | 48 (384) | 24.747 | 61.729 | 4.822 | 5.927 |
| B3LYP | 96 (768) | 80.504 | 96.895 | 14.111 | 13.471 [13.397–16.973]† |
| r²SCAN | 3 (24) | 2.101 | 7.154 | 0.184 | 0.608 |
| r²SCAN | 12 (96) | 5.787 | 11.953 | 0.850 | 1.032 |
| r²SCAN | 48 (384) | 31.119 | 46.129 | 4.986 | 6.378 [6.359–7.116]† |
| r²SCAN | 96 (768) | 87.538 | 75.886 | 14.249 | 14.594 [11.992–15.876]† |
| r²SCAN-3c | 3 (34) | 1.610 | 5.964 | 0.182 | 0.801 |
| r²SCAN-3c | 12 (136) | 8.019 | 15.777 | 1.443 | 1.568 |
| r²SCAN-3c | 48 (544) | 61.976 | 79.081 | 11.376 | 11.195 |
| r²SCAN-3c | 96 (1088) | 254.164 | 169.868 | 42.717 | 26.588 [22.168–28.807]† |
| M06-2X | 3 (24) | 2.244 | 15.205 | 0.197 | 1.051 |
| M06-2X | 12 (96) | 5.037 | 25.024 | 0.907 | 1.775 |
| M06-2X | 48 (384) | 34.287 | 54.767 | 5.343 | 5.294 [4.994–7.289]† |
| M06-2X | 96 (768) | 109.647 | 97.687 | 15.310 | 13.406 |
| ωB97M-V | 3 (24) | 7.222 | 27.917 | 0.383 | 2.575 |
| ωB97M-V | 12 (96) | 29.978 | 109.918 | 1.659 | 8.837 |
| ωB97M-V | 48 (384) | 221.234 | 1216.489 | 10.997 | 105.410 |
| ωB97M-V | 96 (768) | 1190.725 | 4544.557 | 57.799 | 493.399 [399.659–587.833]† |

Warm cells show the median of all five complete endpoints. † marks differing
SCF work and includes the full range; timings are never normalized by iterations.
‡ marks completed native calls without an independent accuracy check for that geometry.
A later timeout does not discard completed phases; incomplete warm repeat sets are not plotted.
r²SCAN-3c: native CPU gCP; reference CPU D4 + gCP.
ωB97M-V: .

## Observed SCF work

Each cell lists iterations/builds. Builds are native Fock builds or reference
SCF J/K calls, including its initial potential. Multiple pairs retain all
observed warm branches. Counts do not imply equivalent work or stopping rules
between engines; complete endpoint time includes analytic forces as well.

| Functional | Atoms | GenerativeQC cold | GPU4PySCF cold | GenerativeQC warm | GPU4PySCF warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 3 | 16/— | 42/43 | 1/— | 1/2 |
| HF | 12 | 22/— | 49/50 | 1/— | 1/2 |
| HF | 48 | 21/— | 44/45 | 1/— | 1/2, 2/3 |
| HF | 96 | 21/— | 44/45 | 1/— | 1/2 |
| PBE | 3 | 12/12 | 45/46 | 1/1 | 1/2 |
| PBE | 12 | 17/17 | 66/67 | 1/1 | 1/2 |
| PBE | 48 | 17/18 | 59/60 | 1/1 | 4/5, 5/6 |
| PBE | 96 | 17/18 | 61/62 | 1/1 | 4/5, 5/6, 6/7, 7/8 |
| PBE0 | 3 | 12/13 | 29/30 | 1/1 | 1/2 |
| PBE0 | 12 | 18/19 | 32/33 | 1/1 | 1/2 |
| PBE0 | 48 | 17/18 | 43/44 | 1/1 | 3/4, 4/5 |
| PBE0 | 96 | 17/17 | 37/38 | 1/1 | 3/4, 4/5 |
| B3LYP | 3 | 12/13 | 40/41 | 1/1 | 1/2 |
| B3LYP | 12 | 17/18 | 36/37 | 1/1 | 3/4 |
| B3LYP | 48 | 17/18 | 54/55 | 1/1 | 3/4 |
| B3LYP | 96 | 16/16 | 48/49 | 1/1 | 3/4, 5/6 |
| r²SCAN | 3 | 12/12 | 35/36 | 1/1 | 1/2 |
| r²SCAN | 12 | 17/17 | 40/41 | 1/1 | 1/2 |
| r²SCAN | 48 | 17/18 | 53/54 | 1/1 | 4/5, 5/6 |
| r²SCAN | 96 | 17/17 | 47/48 | 1/1 | 1/2, 3/4, 4/5 |
| r²SCAN-3c | 3 | 13/13 | 38/39 | 1/1 | 1/2 |
| r²SCAN-3c | 12 | 18/19 | 42/43 | 1/1 | 1/2 |
| r²SCAN-3c | 48 | 19/19 | 59/60 | 1/1 | 4/5 |
| r²SCAN-3c | 96 | 24/24 | 66/67 | 1/1 | 1/2, 3/4, 4/5 |
| M06-2X | 3 | 13/14 | 35/36 | 1/1 | 1/2 |
| M06-2X | 12 | 19/20 | 39/40 | 1/1 | 1/2 |
| M06-2X | 48 | 18/19 | 41/42 | 1/1 | 1/2, 2/3, 3/4 |
| M06-2X | 96 | 18/18 | 41/42 | 1/1 | 1/2 |
| ωB97M-V | 3 | 23/24 | 32/33 | 1/1 | 1/2 |
| ωB97M-V | 12 | 36/37 | 45/46 | 1/1 | 1/2 |
| ωB97M-V | 48 | 36/37 | 45/46 | 1/1 | 1/2 |
| ωB97M-V | 96 | 36/36 | 45/46 | 1/1 | 1/2, 2/3, 3/4 |

Only GPU4PySCF ωB97M-V/96 is newly measured; see the [supplement](README.md). All native samples, historical native gates and other references retain their original publications.

## Updated Native Changed-Geometry Endpoints

All five moved-warm calls are retained. Counts are iterations/Fock builds.

| Atoms | Moved (s) | Moved-warm (s) | Moved work | Moved-warm work |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.502 | 0.182 | 9/9 | 1/1 |
| 12 | 6.128 | 1.447 | 13/13 | 1/1 |
| 48 | 44.668 | 11.426 | 13/13 | 1/1 |
| 96 | 155.577 | 42.754 | 13/13 | 1/1 |

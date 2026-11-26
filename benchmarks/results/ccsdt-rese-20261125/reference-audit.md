# Geometry reference audit

The comparison covered every pre-existing reference field in the 22-molecule
cohort, including geometries whose CUDA optimization reached its wall
time limit. We read the three original articles or their accepted manuscripts.
All 129 existing numbers agree with the specified source entries after
converting pm to Å. The water TZ angle is one additional, independently
published reference. No GPU geometry was regenerated.

## Source locations

| ID | Primary source and page | Table | Columns and rows |
| --- | --- | --- | --- |
| B-II | Bak et al., p. 6550 | II | `Emp.` and `CCSD(T)/cc-pCVQZ`, by molecule and bond. Values in pm divided by 100. |
| B-III | Bak et al., p. 6550 | III | `Emp.` and `CCSD(T)/cc-pCVQZ`, by molecule and angle. Values in degrees. |
| B-VII | Bak et al., p. 6555 | VII | Add the row's `Emp.` and signed `cc-pCVTZ − Emp.` entries, then divide by 100. Both operands are shown below in pm. |
| D-1 | Demaison et al., p. 13611 | 1 | trans-HONO, `r_e^SE` row with footnote d. Columns `r(O=N)`, `r(N–O)`, `r(O–H)`, `∠(ONO)` and `∠(HON)`. |
| D-4 | Demaison et al., p. 13613 | 4 | FNO, `r_e^SE` row. Columns `r(N–F)`, `r(N=O)` and `∠(FNO)`. |
| P-32 | Piccardo et al., accepted manuscript p. 32 | 4 | CH₂F₂ block, `r_e^SE / B3LYP/SNSD` column by parameter. |
| P-33 | Piccardo et al., accepted manuscript p. 33 | 4 | Ethenol block, same column. |
| P-34 | Piccardo et al., accepted manuscript p. 34 | 4 | Aziridine block, same column. |
| P-37 | Piccardo et al., accepted manuscript p. 37 | 4 | trans-Formic acid block, same column. |
| K-I | Kahn et al., p. 074106-5 | I | H₂O, conventional `ae-CCSD(T)` / `cc-pCVTZ` and `cc-pCVQZ` rows, `Angle (deg)` column. The dataset adds the TZ value. |

- Bak et al., DOI 10.1063/1.1357225. [Author-hosted article](https://trygvehelgaker.no/Publications/JChemPhys_114_06548_2001.pdf).
- Demaison et al., DOI 10.1021/jp064769v. [Institution-hosted article](https://theop11.chem.elte.hu/main_index_files/2006_Demaison_JPCA_110_13609_2006_WeakNX.pdf).
- Piccardo et al., DOI 10.1021/jp511432m. The authors' correction concerns fit-residual RMS units. [Accepted manuscript](https://cris.unibo.it/bitstream/11585/523965/7/Semi-Experimental.pdf), [authors' correction](https://doi.org/10.1021/acs.jpca.6b04648).
- Kahn et al., DOI 10.1063/1.3464837. [Institution-hosted article](https://escholarship.org/uc/item/4g45z2kz).

Bak Table II prints the NH₃ bond label as `R_NN`. The molecule,
footnote f and corresponding Table VII row `R_NH` identify the N–H
distance. For the NH₃ angle, Table III prints 107.25° under `Emp.`
and 106.7° under `Expt.` (footnote f). The QZ entry is 106.36°.
The C–H baseline for HCN and both acetylene bond baselines differ
between Tables II and VII. The table below uses Table II for
`re_se` and the separate Table VII operands for the TZ sum.

## Verified parameters

| Molecule | Parameter | Unit | `re_se` | `cc-pcvtz` | `cc-pcvqz` | Source |
| --- | --- | --- | ---: | ---: | ---: | --- |
| hf | H-F | Å | 0.9169 | 0.9167 | 0.9158 | B-II; B-VII (91.69 -0.02 pm) |
| h2o | O-H | Å | 0.9575 | 0.9584 | 0.9571 | B-II; B-VII (95.75 +0.09 pm) |
| h2o | H-O-H | ° | 104.51 | 103.6839 | 104.22 | B-III (SE, QZ); K-I (TZ) |
| co | C-O | Å | 1.1284 | 1.1326 | 1.1289 | B-II; B-VII (112.84 +0.42 pm) |
| n2 | N-N | Å | 1.0977 | 1.1006 | 1.0981 | B-II; B-VII (109.77 +0.29 pm) |
| f2 | F-F | Å | 1.4124 | 1.4133 | 1.4113 | B-II; B-VII (141.24 +0.09 pm) |
| hcn | H-C | Å | 1.0653 | 1.0668 | 1.0655 | B-II; B-VII (106.52 +0.16 pm) |
| hcn | C-N | Å | 1.1534 | 1.1569 | 1.1538 | B-II; B-VII (115.34 +0.35 pm) |
| hnc | H-N | Å | 0.9949 | 0.9957 | 0.9953 | B-II; B-VII (99.49 +0.08 pm) |
| hnc | N-C | Å | 1.1687 | 1.1723 | 1.1693 | B-II; B-VII (116.87 +0.36 pm) |
| ch4 | C-H | Å | 1.0859 | 1.0875 | 1.0864 | B-II; B-VII (108.59 +0.16 pm) |
| nh3 | N-H | Å | 1.0116 | 1.0126 | 1.0112 | B-II; B-VII (101.16 +0.10 pm) |
| nh3 | H-N-H | ° | 107.25 | — | 106.36 | B-III |
| hno | N-H | Å | 1.0517 | 1.0542 | 1.0524 | B-II; B-VII (105.17 +0.25 pm) |
| hno | N-O | Å | 1.2086 | 1.2118 | 1.2085 | B-II; B-VII (120.86 +0.32 pm) |
| hno | H-N-O | ° | 108.27 | — | 108.08 | B-III |
| hof | O-H | Å | 0.9678 | 0.9666 | 0.9657 | B-II; B-VII (96.78 -0.12 pm) |
| hof | O-F | Å | 1.4344 | 1.4353 | 1.4326 | B-II; B-VII (143.44 +0.09 pm) |
| hof | H-O-F | ° | 97.94 | — | 97.85 | B-III |
| co2 | C-O | Å | 1.1601 | 1.1637 | 1.1604 | B-II; B-VII (116.01 +0.36 pm) |
| c2h2 | C-H | Å | 1.0613 | 1.0633 | 1.0621 | B-II; B-VII (106.18 +0.15 pm) |
| c2h2 | C-C | Å | 1.2037 | 1.2067 | 1.2037 | B-II; B-VII (120.35 +0.32 pm) |
| ch2o | C-H | Å | 1.1007 | 1.1022 | 1.1008 | B-II; B-VII (110.07 +0.15 pm) |
| ch2o | C-O | Å | 1.2047 | 1.2071 | 1.2043 | B-II; B-VII (120.47 +0.24 pm) |
| ch2o | H-C-H | ° | 116.74 | — | 116.44 | B-III |
| trans-n2h2 | N-H | Å | 1.0286 | 1.0298 | 1.0284 | B-II; B-VII (102.86 +0.12 pm) |
| trans-n2h2 | N-N | Å | 1.2457 | 1.2506 | 1.2467 | B-II; B-VII (124.57 +0.49 pm) |
| trans-n2h2 | H-N-N | ° | 106.36 | — | 106.17 | B-III |
| c2h4 | C-H | Å | 1.0807 | 1.082 | 1.0809 | B-II; B-VII (108.07 +0.13 pm) |
| c2h4 | C-C | Å | 1.3307 | 1.3339 | 1.3312 | B-II; B-VII (133.07 +0.32 pm) |
| c2h4 | H-C-H | ° | 117.14 | — | 117.10 | B-III |
| fno | N-F | Å | 1.51681 | — | — | D-4 |
| fno | N-O | Å | 1.13176 | — | — | D-4 |
| fno | F-N-O | ° | 109.9361 | — | — | D-4 |
| trans-hono | N=O | Å | 1.1689 | — | — | D-1 |
| trans-hono | N-O | Å | 1.4250 | — | — | D-1 |
| trans-hono | O-H | Å | 0.9647 | — | — | D-1 |
| trans-hono | O-N-O | ° | 110.691 | — | — | D-1 |
| trans-hono | H-O-N | ° | 102.086 | — | — | D-1 |
| ch2f2 | C-F | Å | 1.3533 | — | — | P-32 |
| ch2f2 | C-H | Å | 1.0867 | — | — | P-32 |
| ch2f2 | F-C-F | ° | 108.29 | — | — | P-32 |
| ch2f2 | H-C-H | ° | 113.48 | — | — | P-32 |
| ch2f2 | H-C-F | ° | 108.74 | — | — | P-32 |
| syn-ethenol | O-H | Å | 0.9605 | — | — | P-33 |
| syn-ethenol | C2-O | Å | 1.3598 | — | — | P-33 |
| syn-ethenol | C2-H | Å | 1.0789 | — | — | P-33 |
| syn-ethenol | C1=C2 | Å | 1.3316 | — | — | P-33 |
| syn-ethenol | C1-Hcis | Å | 1.0812 | — | — | P-33 |
| syn-ethenol | C1-Htrans | Å | 1.0770 | — | — | P-33 |
| syn-ethenol | C2-O-H | ° | 108.70 | — | — | P-33 |
| syn-ethenol | C1=C2-H | ° | 122.58 | — | — | P-33 |
| syn-ethenol | C1=C2-O | ° | 126.26 | — | — | P-33 |
| syn-ethenol | C2=C1-Hcis | ° | 121.87 | — | — | P-33 |
| syn-ethenol | C2=C1-Htrans | ° | 119.58 | — | — | P-33 |
| trans-formic-acid | C-H | Å | 1.0918 | — | — | P-37 |
| trans-formic-acid | C=O | Å | 1.1973 | — | — | P-37 |
| trans-formic-acid | C-O | Å | 1.3417 | — | — | P-37 |
| trans-formic-acid | O-H | Å | 0.9656 | — | — | P-37 |
| trans-formic-acid | H-C=O | ° | 125.38 | — | — | P-37 |
| trans-formic-acid | O-C=O | ° | 124.78 | — | — | P-37 |
| trans-formic-acid | C-O-H | ° | 106.78 | — | — | P-37 |
| aziridine | C-N | Å | 1.4714 | — | — | P-34 |
| aziridine | C-C | Å | 1.4777 | — | — | P-34 |
| aziridine | N-H | Å | 1.0126 | — | — | P-34 |
| aziridine | C-Hcis | Å | 1.0805 | — | — | P-34 |
| aziridine | C-Htrans | Å | 1.0791 | — | — | P-34 |
| aziridine | C-N-H | ° | 109.16 | — | — | P-34 |
| aziridine | C-N-C | ° | 60.28 | — | — | P-34 |
| aziridine | N-C-C | ° | 59.86 | — | — | P-34 |
| aziridine | N-C-Hcis | ° | 118.19 | — | — | P-34 |
| aziridine | N-C-Htrans | ° | 114.38 | — | — | P-34 |
| aziridine | C-C-Hcis | ° | 117.81 | — | — | P-34 |
| aziridine | C-C-Htrans | ° | 119.49 | — | — | P-34 |

# Executed SCF work

Each entry is target iterations / Fock builds. GPU4PySCF reports effective-potential calls, a different operation count. Native preliminary Hückel Fock work is shown separately as `guess`. Active incremental solves also show full anchors / delta builds / post-SCF closure builds as A/D/C; periodic rebuilds are a subset of anchors, not extra work. All calls include one complete force evaluation. Full settings, residual histories and work metadata remain in the receipts.

| Mode | Method | Atoms | Engine | Cold | Warm calls |
| --- | --- | ---: | --- | --- | --- |
| df | HF | 3 | GPU4PySCF | 8/9 | 1/2; 1/2; 1/2 |
| df | HF | 3 | GenerativeQC | 9/10; guess 7 | 1/2; guess 0; 1/2; guess 0; 1/2; guess 0 |
| df | HF | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | HF | 12 | GenerativeQC | 10/11; guess 7 | 1/2; guess 0; 1/2; guess 0; 1/2; guess 0 |
| df | HF | 48 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | HF | 48 | GenerativeQC | 10/11; guess 7 | 1/2; guess 0; 1/2; guess 0; 1/2; guess 0 |
| df | HF | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | HF | 96 | GenerativeQC | 11/12; guess 7 | 1/2; guess 0; 1/2; guess 0; 1/2; guess 0 |
| df | PBE | 3 | GPU4PySCF | 8/9 | 1/2; 1/2; 1/2 |
| df | PBE | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | PBE | 12 | GPU4PySCF | 12/13 | 1/2; 1/2; 1/2 |
| df | PBE | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | PBE | 48 | GPU4PySCF | 11/12 | 1/2; 1/2; 1/2 |
| df | PBE | 48 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | PBE | 96 | GPU4PySCF | 11/12 | 1/2; 1/2; 1/2 |
| df | PBE | 96 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | PBE0 | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| df | PBE0 | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | PBE0 | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | PBE0 | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | PBE0 | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | PBE0 | 48 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | PBE0 | 96 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | PBE0 | 96 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | B3LYP | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| df | B3LYP | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | B3LYP | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | B3LYP | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | B3LYP | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | B3LYP | 48 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | B3LYP | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | B3LYP | 96 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| df | r²SCAN | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | r²SCAN | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN | 48 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | r²SCAN | 48 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | r²SCAN | 96 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN-3c | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| df | r²SCAN-3c | 3 | GenerativeQC | 8/8; guess 10 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN-3c | 12 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | r²SCAN-3c | 12 | GenerativeQC | 11/11; guess 10 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN-3c | 48 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | r²SCAN-3c | 48 | GenerativeQC | 10/10; guess 10 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | r²SCAN-3c | 96 | GPU4PySCF | 11/12 | 1/2; 1/2; 1/2 |
| df | r²SCAN-3c | 96 | GenerativeQC | 10/10; guess 10 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | M06-2X | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| df | M06-2X | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | M06-2X | 12 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | M06-2X | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | M06-2X | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | M06-2X | 48 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | M06-2X | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | M06-2X | 96 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | ωB97M-V | 3 | GPU4PySCF | 8/9 | 1/2; 1/2; 1/2 |
| df | ωB97M-V | 3 | GenerativeQC | 19/19; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | ωB97M-V | 12 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | ωB97M-V | 12 | GenerativeQC | 30/31; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | ωB97M-V | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| df | ωB97M-V | 48 | GenerativeQC | 28/29; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| df | ωB97M-V | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| df | ωB97M-V | 96 | GenerativeQC | 28/29; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | HF | 3 | GPU4PySCF | 8/9 | 1/2; 1/2; 1/2 |
| direct | HF | 3 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | HF | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | HF | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | HF | 48 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | HF | 48 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | HF | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | HF | 96 | GenerativeQC | 11/11; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE | 3 | GPU4PySCF | 8/9 | 1/2; 1/2; 1/2 |
| direct | PBE | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE | 12 | GPU4PySCF | 12/13 | 1/2; 1/2; 1/2 |
| direct | PBE | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE | 48 | GPU4PySCF | 11/12 | 1/2; 1/2; 1/2 |
| direct | PBE | 48 | GenerativeQC | 11/11; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE | 96 | GPU4PySCF | 11/12 | 1/2; 1/2; 1/2 |
| direct | PBE | 96 | GenerativeQC | 10/10; guess 7; A/D/C 3/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE0 | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| direct | PBE0 | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE0 | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | PBE0 | 12 | GenerativeQC | 10/10; guess 7; A/D/C 3/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE0 | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | PBE0 | 48 | GenerativeQC | 9/9; guess 7; A/D/C 2/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | PBE0 | 96 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | PBE0 | 96 | GenerativeQC | 9/9; guess 7; A/D/C 2/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | B3LYP | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| direct | B3LYP | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | B3LYP | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | B3LYP | 12 | GenerativeQC | 10/10; guess 7; A/D/C 3/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | B3LYP | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | B3LYP | 48 | GenerativeQC | 9/9; guess 7; A/D/C 2/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | B3LYP | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | B3LYP | 96 | GenerativeQC | 9/9; guess 7; A/D/C 2/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| direct | r²SCAN | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN | 12 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | r²SCAN | 12 | GenerativeQC | 10/10; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN | 48 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | r²SCAN | 48 | GenerativeQC | 9/9; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | r²SCAN | 96 | GenerativeQC | 9/9; guess 7; A/D/C 2/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| direct | r²SCAN-3c | 3 | GenerativeQC | 8/8; guess 10 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | r²SCAN-3c | 12 | GenerativeQC | 11/11; guess 10 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 11/12 | 1/2; 1/2; 1/2 |
| direct | r²SCAN-3c | 48 | GenerativeQC | 10/10; guess 10; A/D/C 2/7/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 11/12 | 1/2; 1/2; 1/2 |
| direct | r²SCAN-3c | 96 | GenerativeQC | 10/10; guess 10; A/D/C 3/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | M06-2X | 3 | GPU4PySCF | 7/8 | 1/2; 1/2; 1/2 |
| direct | M06-2X | 3 | GenerativeQC | 8/8; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | M06-2X | 12 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | M06-2X | 12 | GenerativeQC | 10/10; guess 7; A/D/C 2/7/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | M06-2X | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | M06-2X | 48 | GenerativeQC | 10/10; guess 7; A/D/C 3/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | M06-2X | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | M06-2X | 96 | GenerativeQC | 10/10; guess 7; A/D/C 3/6/1 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | ωB97M-V | 3 | GPU4PySCF | 8/9 | 1/2; 1/2; 1/2 |
| direct | ωB97M-V | 3 | GenerativeQC | 19/19; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | ωB97M-V | 12 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | ωB97M-V | 12 | GenerativeQC | 30/31; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | ωB97M-V | 48 | GPU4PySCF | 9/10 | 1/2; 1/2; 1/2 |
| direct | ωB97M-V | 48 | GenerativeQC | 28/29; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |
| direct | ωB97M-V | 96 | GPU4PySCF | 10/11 | 1/2; 1/2; 1/2 |
| direct | ωB97M-V | 96 | GenerativeQC | 28/29; guess 7 | 1/1; guess 0; 1/1; guess 0; 1/1; guess 0 |

# Executed SCF and precision work

Each entry is iterations / Fock builds / force evaluations / solves. GPU4PySCF reports effective-potential calls in the Fock column; these are not identical operation counts. Fast totals include all exposed candidate, audit and fallback solves. Warm entries retain every call in order, including partial populations. Missing entries have no completed endpoint; no work is inferred for a failed call.

| Mode | Method | Atoms | Engine | Cold | Warm calls |
| --- | --- | ---: | --- | --- | --- |
| df | HF | 3 | GenerativeQC Fast | 27/?/2/2 | 4/?/2/2; 4/?/2/2; 4/?/2/2 |
| df | HF | 3 | GPU4PySCF | 8/9/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | HF | 3 | GenerativeQC Standard | 12/?/1/1 | 2/?/1/1; 2/?/1/1; 2/?/1/1 |
| df | HF | 12 | GenerativeQC Fast | 34/?/2/2 | 4/?/2/2; 4/?/2/2; 4/?/2/2 |
| df | HF | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | HF | 12 | GenerativeQC Standard | 15/?/1/1 | 2/?/1/1; 2/?/1/1; 2/?/1/1 |
| df | HF | 48 | GenerativeQC Fast | 32/?/2/2 | 5/?/2/2; 5/?/2/2; 5/?/2/2 |
| df | HF | 48 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | HF | 48 | GenerativeQC Standard | 14/?/1/1 | 2/?/1/1; 2/?/1/1; 2/?/1/1 |
| df | HF | 96 | GenerativeQC Fast | 32/?/2/2 | 5/?/2/2; 5/?/2/2; 5/?/2/2 |
| df | HF | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | HF | 96 | GenerativeQC Standard | 14/?/1/1 | 2/?/1/1; 2/?/1/1; 2/?/1/1 |
| df | PBE | 3 | GenerativeQC Fast | 23/23/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | PBE | 3 | GPU4PySCF | 8/9/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | PBE | 12 | GenerativeQC Fast | 32/32/2/2 | 6/6/2/2; 6/6/2/2; 6/6/2/2 |
| df | PBE | 12 | GPU4PySCF | 12/13/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE | 12 | GenerativeQC Standard | 15/15/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | PBE | 48 | GenerativeQC Fast | 33/33/2/2 | 6/6/2/2; 6/6/2/2; 6/6/2/2 |
| df | PBE | 48 | GPU4PySCF | 11/12/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE | 48 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | PBE | 96 | GenerativeQC Fast | 33/33/2/2 | 6/6/2/2; 6/6/2/2; 6/6/2/2 |
| df | PBE | 96 | GPU4PySCF | 11/12/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE | 96 | GenerativeQC Standard | 15/15/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | PBE0 | 3 | GenerativeQC Fast | 23/23/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | PBE0 | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE0 | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | PBE0 | 12 | GenerativeQC Fast | 35/35/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | PBE0 | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE0 | 12 | GenerativeQC Standard | 17/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | PBE0 | 48 | GenerativeQC Fast | 32/32/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | PBE0 | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE0 | 48 | GenerativeQC Standard | 15/15/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | PBE0 | 96 | GenerativeQC Fast | 32/32/2/2 | 5/5/2/2; 5/5/2/2; 5/5/2/2 |
| df | PBE0 | 96 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | PBE0 | 96 | GenerativeQC Standard | 15/15/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | B3LYP | 3 | GenerativeQC Fast | 23/23/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | B3LYP | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | B3LYP | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | B3LYP | 12 | GenerativeQC Fast | 33/33/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | B3LYP | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | B3LYP | 12 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | B3LYP | 48 | GenerativeQC Fast | 33/33/2/2 | 5/5/2/2; 5/5/2/2; 5/5/2/2 |
| df | B3LYP | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | B3LYP | 48 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | B3LYP | 96 | GenerativeQC Fast | 32/32/2/2 | 5/5/2/2; 5/5/2/2; 5/5/2/2 |
| df | B3LYP | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | B3LYP | 96 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN | 3 | GenerativeQC Fast | 23/23/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | r²SCAN | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN | 12 | GenerativeQC Fast | 33/33/2/2 | 5/5/2/2; 5/5/2/2; 5/5/2/2 |
| df | r²SCAN | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN | 12 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN | 48 | GenerativeQC Fast | 33/33/2/2 | 5/5/2/2; 5/5/2/2; 5/5/2/2 |
| df | r²SCAN | 48 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN | 48 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN | 96 | GenerativeQC Fast | 31/31/2/2 | 7/7/2/2; 7/7/2/2; 7/7/2/2 |
| df | r²SCAN | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN | 96 | GenerativeQC Standard | 14/14/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN-3c | 3 | GenerativeQC Fast | 24/24/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | r²SCAN-3c | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN-3c | 12 | GenerativeQC Fast | 35/35/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | r²SCAN-3c | 12 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 17/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN-3c | 48 | GenerativeQC Fast | 35/35/2/2 | 6/6/2/2; 6/6/2/2; 6/6/2/2 |
| df | r²SCAN-3c | 48 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | r²SCAN-3c | 96 | GenerativeQC Fast | — | — |
| df | r²SCAN-3c | 96 | GPU4PySCF | 11/12/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | — | — |
| df | M06-2X | 3 | GenerativeQC Fast | 25/25/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | M06-2X | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | M06-2X | 3 | GenerativeQC Standard | 12/12/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | M06-2X | 12 | GenerativeQC Fast | 36/36/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | M06-2X | 12 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | M06-2X | 12 | GenerativeQC Standard | 17/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | M06-2X | 48 | GenerativeQC Fast | 34/34/2/2 | 5/5/2/2; 5/5/2/2; 5/5/2/2 |
| df | M06-2X | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | M06-2X | 48 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | M06-2X | 96 | GenerativeQC Fast | 34/34/2/2 | 6/6/2/2; 6/6/2/2; 6/6/2/2 |
| df | M06-2X | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | M06-2X | 96 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | ωB97M-V | 3 | GenerativeQC Fast | 44/44/2/2 | 4/4/2/2; 4/4/2/2; 4/4/2/2 |
| df | ωB97M-V | 3 | GPU4PySCF | 8/9/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | ωB97M-V | 3 | GenerativeQC Standard | 19/19/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | ωB97M-V | 12 | GenerativeQC Fast | 60/60/2/2 | 4/4/2/2; 5/5/2/2; 5/5/2/2 |
| df | ωB97M-V | 12 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | ωB97M-V | 12 | GenerativeQC Standard | 30/30/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | ωB97M-V | 48 | GenerativeQC Fast | 56/56/2/2 | 5/5/2/2; 5/5/2/2; 5/5/2/2 |
| df | ωB97M-V | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | ωB97M-V | 48 | GenerativeQC Standard | 28/28/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| df | ωB97M-V | 96 | GenerativeQC Fast | — | — |
| df | ωB97M-V | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| df | ωB97M-V | 96 | GenerativeQC Standard | — | — |
| direct | HF | 3 | GenerativeQC Fast | 12/13/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | HF | 3 | GPU4PySCF | 8/9/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | HF | 3 | GenerativeQC Standard | 12/?/1/1 | 1/?/1/1; 1/?/1/1; 1/?/1/1 |
| direct | HF | 12 | GenerativeQC Fast | 15/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | HF | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | HF | 12 | GenerativeQC Standard | 15/?/1/1 | 1/?/1/1; 1/?/1/1; 1/?/1/1 |
| direct | HF | 48 | GenerativeQC Fast | 35/37/2/2 | 4/6/2/2; 4/6/2/2; 4/6/2/2 |
| direct | HF | 48 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | HF | 48 | GenerativeQC Standard | 14/?/1/1 | 1/?/1/1; 1/?/1/1; 1/?/1/1 |
| direct | HF | 96 | GenerativeQC Fast | 35/37/2/2 | 4/6/2/2; 4/6/2/2; 4/6/2/2 |
| direct | HF | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | HF | 96 | GenerativeQC Standard | 14/?/1/1 | 1/?/1/1; 1/?/1/1; 1/?/1/1 |
| direct | PBE | 3 | GenerativeQC Fast | 8/8/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE | 3 | GPU4PySCF | 8/9/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE | 12 | GenerativeQC Fast | 10/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE | 12 | GPU4PySCF | 12/13/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE | 12 | GenerativeQC Standard | 15/15/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE | 48 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE | 48 | GPU4PySCF | 11/12/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE | 48 | GenerativeQC Standard | 16/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE | 96 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE | 96 | GPU4PySCF | 11/12/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE | 96 | GenerativeQC Standard | 15/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 3 | GenerativeQC Fast | 8/9/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE0 | 3 | GenerativeQC Standard | 11/12/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 12 | GenerativeQC Fast | 10/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE0 | 12 | GenerativeQC Standard | 17/18/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 48 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE0 | 48 | GenerativeQC Standard | 15/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 96 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | PBE0 | 96 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | PBE0 | 96 | GenerativeQC Standard | 15/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 3 | GenerativeQC Fast | 8/9/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | B3LYP | 3 | GenerativeQC Standard | 11/12/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 12 | GenerativeQC Fast | 10/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | B3LYP | 12 | GenerativeQC Standard | 16/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 48 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | B3LYP | 48 | GenerativeQC Standard | 16/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 96 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | B3LYP | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | B3LYP | 96 | GenerativeQC Standard | 16/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 3 | GenerativeQC Fast | 8/8/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 12 | GenerativeQC Fast | 10/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 12 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN | 12 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 48 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 48 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN | 48 | GenerativeQC Standard | 16/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 96 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN | 96 | GenerativeQC Standard | 14/14/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 3 | GenerativeQC Fast | 8/8/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 11/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 12 | GenerativeQC Fast | 10/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 17/18/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 48 | GenerativeQC Fast | 10/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 11/12/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 96 | GenerativeQC Fast | 15/15/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 11/12/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 22/22/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 3 | GenerativeQC Fast | 9/10/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 3 | GPU4PySCF | 7/8/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | M06-2X | 3 | GenerativeQC Standard | 12/13/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 12 | GenerativeQC Fast | 10/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 12 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | M06-2X | 12 | GenerativeQC Standard | 17/18/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 48 | GenerativeQC Fast | 10/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | M06-2X | 48 | GenerativeQC Standard | 16/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 96 | GenerativeQC Fast | 10/11/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | M06-2X | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | M06-2X | 96 | GenerativeQC Standard | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 3 | GenerativeQC Fast | 14/14/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 3 | GPU4PySCF | 8/9/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 19/19/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 12 | GenerativeQC Fast | 17/18/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 12 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 30/31/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 48 | GenerativeQC Fast | 16/17/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 48 | GPU4PySCF | 9/10/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 28/29/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 96 | GenerativeQC Fast | 16/16/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |
| direct | ωB97M-V | 96 | GPU4PySCF | 10/11/1/1 | 1/2/1/1; 1/2/1/1; 1/2/1/1 |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 28/28/1/1 | 1/1/1/1; 1/1/1/1; 1/1/1/1 |

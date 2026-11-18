# H100: cooperative CUDA D4 EEQ charges

r2SCAN-3c/def2-mTZVPP Direct water clusters with the README defaults, fresh
consecutive processes on one physical H100 80GB in qz Job `gqc-h100-opt-1118-qa`.
Cold = constructor + preparation + SCF + analytic forces + host export; warm =
median of three frozen-own-density calls. Base is master `a6d5a3c00` (native
identity `c5f05cc9…`). The candidate binary was built from this change before
clang-format re-wrapped three argument lists (identity `a43d2736…`); the
committed tree differs from it only in that whitespace. `results.json.xz`
retains every arm with all samples, forces and offline qualification verdicts,
the D4 replay comparison and the H100 D4 test log.

## Attribution

Nsight Systems 2025.1.3 on base, one cold + two warm calls: `eeq_prepare_kernel`
113 ms and `eeq_compose_kernel` 113 ms per call (one GPU thread per system), about
0.23 s of the 4.04 s warm endpoint. Both run after the SCF/force work.

## D4 batch replay (`D4CorrectionBatch`, CUDA, 12 replays each)

| Case | Base | Candidate |
| --- | ---: | ---: |
| water-96 | 230.6 ms | 26.0 ms |
| ragged 3/96/12/48 atoms | 230.6 ms | 25.9 ms |

EEQ charges are bitwise identical to base and replay-deterministic for every
system. Energies and gradients already differ between replays of the unchanged
base (atomic accumulation in fixed-charge D4): up to 1.0e-17 Eh and
2.4e-19 Eh/Bohr. Candidate-to-base differences stay within that spread
(≤ 1.04e-17 Eh, ≤ 2.4e-19 Eh/Bohr). Gradient-free replays publish identical
statuses and charges. All statuses succeed.

## Endpoint ABBA, water-96 r2SCAN-3c Direct

| Arm | Build | Cold (s) | Warm (s) |
| --- | --- | ---: | ---: |
| A0 | base | 9.931 | 4.036, 4.038, 4.037 |
| B1 | candidate | 9.679 | 3.835, 3.846, 3.836 |
| B2 | candidate | 9.739 | 3.821, 3.826, 3.839 |
| A3 | base | 9.913 | 4.029, 4.028, 4.027 |

Same iteration and Fock counts (cold 11/11, warm 1/1). `correction_energy`
(D4 + gCP) is bitwise identical in every sample; all arms pass the original
independent gates. The 2026-11-17 GPU4PySCF reference for this case is 3.899 s
warm ([readme-h100-20261117](../readme-h100-20261117/README.md)).

## H100 native tests

`generativeqc_d4_{production,ragged,reference,eeq}_tests` and their CUDA
variants plus `generativeqc_d4_schedule_cuda_tests`: 8/8 built tests pass,
including the 4100-system ragged isolation test.

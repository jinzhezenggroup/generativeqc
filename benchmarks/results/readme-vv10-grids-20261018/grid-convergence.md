# Grid differences — not FP64-qualified

GPU4PySCF vs 64×16×32, original/moved geometries. Baseline convergence is unproven.
Historical loose gates: 1e-5 Eh / 1e-4 Eh/bohr; FP64 thresholds: 1e-8 / 1e-7.
Neither column certifies a full FP64 endpoint or Fast total error budget.

| Method | Mode | Atoms | Grid | Points/atom | Max ΔE (Eh) | Max ΔF (Eh/bohr) | Loose gate | FP64 |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| wb97m-v | Direct | 3 | 24×8×16 | 3072 | 1.894e-04 | 1.076e-03 | fail | fail |
| wb97m-v | Direct | 3 | 32×12×24 | 9216 | 5.473e-06 | 1.802e-04 | fail | fail |
| wb97m-v | Direct | 3 | 32×16×32 | 16384 | 5.399e-06 | 1.635e-04 | fail | fail |
| wb97m-v | Direct | 3 | 40×16×32 | 20480 | 1.783e-06 | 6.010e-05 | pass | fail |
| wb97m-v | Direct | 3 | 48×12×24 | 13824 | 2.098e-06 | 3.479e-05 | pass | fail |
| wb97m-v | Direct | 3 | 48×16×32 | 24576 | 1.332e-06 | 2.969e-05 | pass | fail |
| wb97m-v | Direct | 3 | 64×12×24 | 18432 | 7.468e-07 | 1.847e-05 | pass | fail |
| wb97m-v | Direct | 3 | 64×16×32 | 32768 | 0.000e+00 | 0.000e+00 | baseline | baseline |
| wb97m-v | Direct | 12 | 24×8×16 | 3072 | — | — | unavailable | unavailable |
| wb97m-v | Direct | 12 | 32×12×24 | 9216 | 6.172e-05 | 2.171e-04 | fail | fail |
| wb97m-v | Direct | 12 | 32×16×32 | 16384 | 5.753e-05 | 2.456e-04 | fail | fail |
| wb97m-v | Direct | 12 | 40×16×32 | 20480 | 3.435e-06 | 1.149e-04 | fail | fail |
| wb97m-v | Direct | 12 | 48×12×24 | 13824 | 5.242e-06 | 4.251e-05 | pass | fail |
| wb97m-v | Direct | 12 | 48×16×32 | 24576 | 6.889e-06 | 3.206e-05 | pass | fail |
| wb97m-v | Direct | 12 | 64×12×24 | 18432 | 2.226e-06 | 2.743e-05 | pass | fail |
| wb97m-v | Direct | 12 | 64×16×32 | 32768 | 0.000e+00 | 0.000e+00 | baseline | baseline |

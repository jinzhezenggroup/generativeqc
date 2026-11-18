# H100: batched loads in the stationary geometry reduction

The KS force grid-response path calls `geometry_reduce` once per force tile (1050
tiles for r2SCAN-3c96). Each call sums `geometry_lanes` partial rows into `9*na`
columns with one thread per column. With 864 columns that is 14 blocks of
64 threads, and each thread issues one dependent global load per lane. On H100,
r2SCAN-3c96 spent 128 ms per warm call in it.

Each thread now loads 16 lanes into registers before adding them, in the original
lane order. The FP64 sum is the same left-to-right sum, with sixteen loads in
flight instead of one. The tail lanes keep the scalar loop.

## Measurements

Fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-opt-1118-qa`, in order base, batched, batched, base. Base is the parent
build snapshot (identity `0c5dcf62…`). Batched is the same tree plus only this
generator change (identity `0335e216…`). Both trees also contained the then-unmerged
#3115/#3116/#3118/#3119.

| Case | base warm (s) | batched warm (s) | base cold | batched cold |
| --- | --- | --- | --- | --- |
| Direct r2SCAN-3c 96 | 3.841, 3.849 | 3.783, 3.769 | 9.881, 9.636 | 9.680, 9.603 |
| Direct PBE0 96 | 2.119, 2.113 | 2.058, 2.058 | 7.256, 7.234 | 7.057, 7.002 |
| DF B3LYP 96 | 1.696, 1.702 | 1.645, 1.644 | 5.384, 5.394 | 5.247, 5.307 |

- Warm values are medians of three calls. Iteration and Fock counts are identical,
  and every arm qualifies.
- An nsys trace of the batched build shows `geometry_reduce` at 71 ms per
  r2SCAN-3c96 warm call (128 ms before).
- Cross-build force differences (r2SCAN 1.3e-11, PBE0 4.7e-13, B3LYP
  3.8e-13 Eh/Bohr) are within the same build's run-to-run spread (r2SCAN 1.2e-11,
  PBE0 4.6e-13, B3LYP 3.5e-13).
- `endpoint.json` holds every arm, sample and qualification error;
  `results.json.xz` adds all force vectors.

These measurements do not qualify any other GPU; the change has no architecture
switch.

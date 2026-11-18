# H100: README HF/DFT cohort on master `46942b9f0`, 2026-11-17

Source `46942b9f05de4ec39cdef4102e828a5b05fccf86` (master after #3112), native
source identity `fdb55c33f871e6407bcfadae4f8ee3f3f74ce5ac591e04f961174f6445ac6eef`.
Compared with the [2026-11-12 H100 cohort](../readme-h100-20261112/README.md)
(`a7241b4fd`), this source adds the ωB97M-V changes of #3111/#3112 and the
other master changes merged since #3072.

| Mode | Cold wins | Warm wins |
| --- | ---: | ---: |
| Direct | 32/32 | 31/32 |
| DF | 32/32 | 31/32 |

All 64 pairs / 128 processes / 512 E+force calls succeed, with no retries or
old timings. These are single-GPU point observations, not statistical rankings
or universal speedups. **H100 is not ahead in every case:**

| Case | Phase | GenerativeQC | GPU4PySCF | Native iter/Fock | Reference iter/V |
| --- | --- | ---: | ---: | --- | --- |
| DF HF96 | warm | 0.971 s | 0.758 s | 1/2 | 1/2 |
| Direct r2SCAN-3c96 | warm | 4.022 s | 3.899 s | 1/1 | 1/2 |

Both remain slow points from the 2026-11-12 cohort, with unchanged native
times (0.977 and 4.024 s then).

Direct ωB97M-V96 is now 34.180 versus 42.971 s cold (48.306 s on
`a7241b4fd`, 13 instead of 16 native Fock builds) and 4.966 versus 12.766 s
warm (2.57×); DF ωB97M-V96 cold is 22.044 versus 37.666 s.

Direct HF96 is no longer slow (cold 6.714 vs 7.636 s, warm 1.408 vs 1.727 s),
but the native times are unchanged from 2026-11-12 (6.741 / 1.397 s): the
reference was slower on this different physical H100 (cold 7.636 versus
5.986 s; it was 7.182 s in the 2026-11-11 cohort). Treat this pair as close,
not as a native improvement.

## Conditions

One 1×H100 80GB HBM3 (driver 595.58.03, SM90) qz Job allocation runs every
process consecutively on the same physical GPU, first-engine order balanced
32/32, math threads=1, `GENERATIVEQC_PROFILE=off`. A complete untimed prewarm
pass (one cold + one warm call per population, same order) fills CUDA/CuPy/JIT
caches first; its timings never enter the cohort. Cold includes constructor,
setup, SCF, forces and host export; warm is the median of three calls from the
engine's own frozen cold density; bars are min–max. The build is a
source-matched CUDA 12.9 Release from sccache 0.16.0.

The Standard arm uses public defaults with no grid, SCF, initial-guess or tile
controls. GPU4PySCF 1.9.0 / CuPy 13.4.1 / PySCF 2.14.0 uses stock SCF, grids
and default auxiliary tables. The preregistered order, oracles, reuse contract
and auxiliary inputs are byte-identical to the 2026-11-12 cohort.

## Accuracy

Accuracy is reported, not gated. Both engines are compared independently with
the unchanged `readme-refresh-20261026` dense Direct references (tighter SCF,
denser XC grids, no density fitting, full grid-response forces);
[errors.md](errors.md) lists both engines' largest energy/force errors for every
pair. The largest are 2.1e-5 Eh/atom (DF HF, density-fitting error) and
5.7e-5 Eh/Bohr (DF M06-2X), identical for both engines; call by call (cold with
cold, warm with warm), the two engines' errors differ by at most 1.6e-9 Eh/atom
and 3.5e-6 Eh/Bohr. All 1024 cross-engine full-vector comparisons pass.
`samples.json.xz` still records the historical acceptance-gate counts for
archive replay; they are not a publication claim.

Replay scalar errors, the error table and figures with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-h100-20261117 \
  --output .artifacts/readme-h100-replay \
  --error-table .artifacts/readme-h100-replay/errors.md
```

Full-vector, build and qualification receipts remain under the
`original_acquisition_root` recorded in `samples.json.xz`. Replay uses recorded
scalar errors, not fresh vector subtraction.

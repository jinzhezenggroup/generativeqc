# H100: README HF/DFT cohort on master `a7241b4fd`, 2026-11-12

Source `a7241b4fdaf586fef25da59913e4d267aca67461` (master after #3072), native
source identity `1f6947e66e5911285bda2f336cd3fa0a40431656a7bf1d128b35e0c32ac103c9`.
The RTX 5090 counterpart is the [#3072 cohort](../readme-rtx5090-composed-20261111/README.md)
(master `4024f95b8` plus the #3072 preparation optimizations, native identity
`9d305b13…`). This H100 source additionally contains #3069 (erf metric
schedule) and #3070 (KS density-confirmation guard), merged before #3072.

| Mode | Cold wins | Warm wins |
| --- | ---: | ---: |
| Direct | 30/32 | 30/32 |
| DF | 32/32 | 31/32 |

All 64 pairs / 128 processes / 512 E+force calls succeed, with no retries or
old timings. These are single-GPU point observations, not statistical rankings
or universal speedups. **H100 is not ahead in every case:**

| Case | Phase | GenerativeQC | GPU4PySCF | Native iter/Fock (guess) | Reference iter/V |
| --- | --- | ---: | ---: | --- | --- |
| Direct HF96 | cold | 6.741 s | 5.986 s | 11/11 (7) | 10/11 |
| Direct HF96 | warm | 1.397 s | 1.334 s | 1/1 | 1/2 |
| Direct ωB97M-V96 | cold | 48.306 s | 40.668 s | 16/16 (7) | 10/11 |
| Direct r2SCAN-3c96 | warm | 4.024 s | 3.751 s | 1/1 | 1/2 |
| DF HF96 | warm | 0.977 s | 0.752 s | 1/2 | 1/2 |

The native Direct 96-atom slow points are within 1% of the
[2026-11-11 H100 cohort](../readme-h100-20261111/README.md); single cold samples
of small systems vary by up to 41%. Direct HF96 is a new loss because the
reference was faster on this different physical H100 (cold 5.986 versus
7.182 s). Warm Direct ωB97M-V96 is 1.30×
and DF ωB97M-V96 1.99× faster; RTX 5090 Direct ωB97M-V96 warm is 9.42×.
The smaller H100 large-system Direct margins are not explained by this cohort.

## Conditions

One 1×H100 80GB HBM3 (driver 595.58.03, SM90) qz notebook allocation runs every
process consecutively on the same physical GPU, first-engine order balanced
32/32, math threads=1, `GENERATIVEQC_PROFILE=off`. A complete untimed prewarm
pass (one cold + one warm call per population, same order) fills CUDA/CuPy/JIT
caches first; its timings never enter the cohort. Cold includes constructor,
setup, SCF, forces and host export; warm is the median of three calls from the
engine's own frozen cold density; bars are min–max. The build is a
source-matched CUDA 12.9 Release from sccache 0.16.0.

The Standard arm uses public defaults with no grid, SCF, initial-guess or tile
controls. GPU4PySCF 1.9.0 / CuPy 13.4.1 / PySCF 2.14.0 uses stock SCF, grids
and default auxiliary tables (RTX 5090: CuPy 13.6.0).

## Accuracy

Accuracy is reported, not gated. Both engines are compared independently with
the unchanged `readme-refresh-20261026` dense Direct references (tighter SCF,
denser XC grids, no density fitting, full grid-response forces);
[errors.md](errors.md) lists both engines' largest energy/force errors for every
pair. As on RTX 5090, the largest are 2.1e-5 Eh/atom (DF HF, density-fitting
error) and 5.7e-5 Eh/Bohr (DF M06-2X), identical for both engines; call by call (cold with cold, warm with warm), the two engines' errors differ by at most 1.6e-9 Eh/atom and 3.5e-6 Eh/Bohr.
`samples.json.xz` still records the
historical acceptance-gate counts for archive replay; they are not a
publication claim.

Replay scalar errors, the error table and figures with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-h100-20261112 \
  --output .artifacts/readme-h100-replay \
  --error-table .artifacts/readme-h100-replay/errors.md
```

Full-vector, build and qualification receipts remain under the
`original_acquisition_root` recorded in `samples.json.xz`. Replay uses recorded
scalar errors, not fresh vector subtraction.

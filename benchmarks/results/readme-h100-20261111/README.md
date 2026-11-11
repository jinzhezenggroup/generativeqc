# H100: README HF/DFT cohort on master `5533417aa`, 2026-11-11

Source `5533417aac131c49fac0e02b9303a6c65f3dd774` (master after #3069), native
source identity `8c47adf9ee506c0048432f01f2e6d7cbb8c53f0eb6d13c6a0d936303ab751b43`.
Its RTX 5090 counterpart is the [#3067 cohort](../readme-rtx5090-composed-20261111/README.md)
(`a50c80f62`, native identity `f502f9e5…`). The only native delta is
`src/scf/cuda/df_source_setup.cpp` (#3069 erf metric schedule). The sealed
staging protocol named the earlier `5a1ae34ad` draft cohort as its counterpart;
`samples.json.xz` records the correction.

| Mode | Cold wins | Warm wins | Qualified cold | Qualified warm |
| --- | ---: | ---: | ---: | ---: |
| Direct | 31/32 | 31/32 | 30/31 | 30/31 |
| DF | 32/32 | 31/32 | 27/27 | 27/27 |

All 64 pairs / 128 processes / 512 E+force calls succeed, with no retries or
old timings. These are single-GPU point observations, not statistical rankings
or universal speedups. **H100 is not ahead in every case:**

| Case | Phase | GenerativeQC | GPU4PySCF | Native iter/Fock (guess) | Reference iter/V |
| --- | --- | ---: | ---: | --- | --- |
| Direct ωB97M-V96 | cold | 48.098 s | 42.049 s | 16/16 (7) | 10/11 |
| Direct r2SCAN-3c96 | warm | 4.008 s | 3.848 s | 1/1 | 1/2 |
| DF HF96 (fails gates) | warm | 0.956 s | 0.767 s | 1/2 | 1/2 |

Warm Direct ωB97M-V96 is 1.33× and DF ωB97M-V96 2.01× faster; the RTX 5090
values are 9.39× and 5.50×. The H100 Direct ωB97M-V and r2SCAN-3c large-system
margins are smaller than on RTX 5090 and are not explained by this cohort.

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
controls, as in the RTX 5090 cohort. GPU4PySCF 1.9.0 / CuPy 13.4.1 / PySCF
2.14.0 uses stock SCF, grids and default auxiliary tables (RTX 5090: CuPy
13.6.0).

## Accuracy

Both engines independently qualify 58/64 pairs against the unchanged
`readme-refresh-20261026` dense oracles: E/atom ≤1e-5 Eh, Fmax ≤1e-4 and
Frms ≤3e-5 Eh/Bohr. Failures are the same as on RTX 5090: all DF HF sizes and
Direct/DF M06-2X3. DF ωB97M-V qualifies at all four sizes; before #3069 the
native H100 erf metric failed every size. All 1024 cross-engine vector
comparisons pass; they do not replace the independent gates.

Replay scalar gates/figures with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-h100-20261111 \
  --output .artifacts/readme-h100-replay
```

Full-vector, build and qualification receipts remain under the
`original_acquisition_root` recorded in `samples.json.xz`. Replay uses recorded
scalar errors, not fresh vector subtraction. See the
[erf metric decision](../../../.agents/notes/implemented/compatibility/2026-11-11-sm90-erf-metric-schedule.md).

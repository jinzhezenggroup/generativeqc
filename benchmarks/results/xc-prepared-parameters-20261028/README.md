# KTBM sharing with prepared launch coefficients

This follow-up removes the [original shared worker's 3% point regression](../xc-parameter-sharing-20261027/README.md).
Sharing now defaults on when automatic CUDA AOT and CUDA are enabled. Explicit
`GENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS=OFF` retains literal specialization.
Automatic AOT admission and supported functionals do not change.

The compiler prepares coefficient/literal products after AD through its existing
Graph algebra, then supplies one immutable row by value in CUDA launch parameter
space. Both general and equal-spin workers retain physical derivative semantics.

| Measurement | Literal | Prepared shared worker |
| --- | ---: | ---: |
| KTBM point kernels | 104 | 4 |
| Selected package's 33 object bytes | 17,416,384 | 1,039,160 |
| Same configured complete library bytes | 720,203,192 | 704,027,504 |
| Code generation, seconds | 9.22867 | 11.73309 |
| Actual compilation, seconds | 22.29041 | 9.88550 |

Object bytes fall **94.03%**; configured library bytes fall **2.25%**. Combined
generation/compilation takes **31.4% less time** in this one controlled trial.
NVCC 12.9.1 targets SM120, with eight compiler jobs and verified sccache 0.16.0.
Both arms use `SCCACHE_RECACHE=1` for actual compilation; existing cache contents
are preserved/refreshed. Ordinary qualification uses normal cache reuse.

All 26 registrations were tested with one/two spins, 16,384/262,144 points and
ordinary/tile-127 layouts: **208 cases**, two randomized paired rounds each,
50 CUDA-event samples after 10 warm launches. The ratio of per-case median
times has median **0.989173** and maximum **0.997337**. Every case median is
faster than literal specialization. Both paths use 110/118 registers for
ordinary/batched execution, versus 116/122 for literal specialization and up to
140/148 for the original shared worker. Stack remains 96 bytes.

Complete SCF qualification uses the same seven cohorts and numerical controls as
the previous campaign: H4 RKS/UKS for KTBM_0, KTBM_12 and KTBM_GAP, plus water4
RKS KTBM_0. Five alternating fresh-process pairs produce **350 calls**, including
cold preparation/SCF, three frozen-seed warm calls and moved geometry. Iterations,
Fock counts and complete precision/work ledgers match. Maximum energy difference
is **2.274e-13 Eh**; phase median time ratios are **0.993306–1.003332**.
The 234-case independent Libxc/PySCF GPU qualification passes for all rows.

Promotion gates require aggregate and per-spin/layout point medians at most 1.0,
no point case above 1.01, and isolated endpoint phase medians within 2%, together
with existing numerical and work gates. A concurrent three-pair endpoint pilot
had noisy host cold/moved timing; its complete population is retained separately,
without pooling or choosing samples. Point and oracle jobs use separate Slurm
GPU allocations; the isolated endpoint job waits for those jobs to finish.

`receipts.json.xz` retains exact generated point sources, measured/publication
source reconstruction, binary identities, all samples and endpoint ledgers,
all 52 public-KS oracle receipts, ablation builds/timings, cache statistics and
drivers. Formatting before the complete build changed descriptor provenance;
the point benchmark and full-library worker bodies are byte-identical. The
final default-policy check is an opt-in migration comparison, not a required
source-fingerprint CI test.

```bash
python benchmarks/results/xc-prepared-parameters-20261028/reduce.py
```

Measurements qualify this KTBM package on Slurm node2 PRO6000 / SM120, not every
architecture, functional family or distribution. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-28-xc-prepared-parameters.md)
for ownership invariants and rejected approaches. Experiment drivers in the
receipt use retained local paths; adapt them when reconstructing the source.

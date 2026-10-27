# Optional KTBM CUDA parameter sharing

Historical opt-in implementation. The
[prepared-coefficient follow-up](../xc-prepared-parameters-20261028/README.md)
removes its measured point regression and supersedes the default policy below.

This is an **opt-in size/runtime tradeoff**, not a default performance promotion.
`GENERATIVEQC_SHARE_AUTOMATIC_XC_CUDA_PARAMETERS=OFF` retains literal constant
specialization. With automatic CUDA AOT enabled, setting it to `ON` shares the
audited KTBM family. Other registrations and singleton selections retain their
existing specialization. Functional coverage and scientific admission do not
change.

Baseline is `2152ae767` (full revision and measured/publication reconstruction
patches are in `receipts.json.xz`). Both packages select all 26 KTBM registrations
and retain 33 registry objects. NVCC 12.9.1 targets SM120 in Release mode;
compilation uses an AMD EPYC 9J14 with eight concurrent jobs. GPU qualification
uses Slurm PRO6000 allocations on node2.

| Measurement | Literal | Shared parameters |
| --- | ---: | ---: |
| KTBM point kernel entries | 104 | 4 |
| All CUDA entries in these objects, including incidental helpers | 208 | 8 |
| CUDA registry object bytes | 17,416,384 | 1,093,560 |
| Same configured complete library bytes | 720,203,192 | 704,080,680 |
| Code generation, seconds | 9.32949 | 11.63339 |
| Complete package compilation, seconds | 22.14684 | 10.16708 |

The selected package's objects shrink **93.72%**; the same configured library
shrinks **2.24%**. These are not full-catalog or distribution compression ratios.
One controlled compile trial reduces compiler time by **54.1%** and combined
generation/compilation time by **30.7%**, while generation itself takes longer.
The experiment uses verified sccache 0.16.0 with `SCCACHE_RECACHE=1` on both arms
to measure actual compiler work; it retains and refreshes the existing cache.
Ordinary validation keeps cache reuse enabled. Earlier partial/cache-hit CMake
builds are excluded from this comparison. This is one trial, not a timing
confidence interval or a multiarchitecture claim.

Parameter loads and reduced constant folding have a measurable cost. Across
208 paired point cases (all 26 rows, one/two spins, 16,384/262,144 points,
ordinary/batched layouts), the median candidate/literal ratio is **1.03089**
and the maximum is **1.04182**. Each arm retains 30 CUDA-event samples after
10 warm launches. General-worker registers rise from 116/122 to 122/128;
equal-spin workers use 140/148 registers. Stack usage stays 96 bytes, with no
local-memory or shared-memory increase in these resources. This is why sharing
is not enabled by default despite its substantial size reduction.

Seven complete endpoint cohorts use three alternating fresh-process pairs:
H4 RKS/UKS for KTBM_0, KTBM_12 and KTBM_GAP, plus water4 RKS for KTBM_0.
Each process includes cold configuration/preparation/SCF, three frozen-seed warm
recomputations and changed geometry. All 210 calls retain matching iteration,
Fock-build and full precision/work ledgers; maximum energy difference is
1.71e-13 Eh. The largest cold/warm/moved median ratio is **1.01507**. These
results meet the explicit 5% qualification bound, but do not establish zero
runtime overhead. H4 uses the existing oracle's strict tolerances and tile 37;
water4 uses its separately recorded tolerances and tile 1024. Failed or
incomplete large-grid pilots on the literal baseline do not enter these cohorts.

Independent validation passes 635 host tests (34 unavailable cases skipped),
15 CUDA-provider host contracts, 84 final option/parameter tests, 234 real-GPU
cases and two receipt-export checks. GPU tests cover every parameter row, ordinary/batched point layouts,
invalid input, explicit equal-spin fallback, AO/XC/Fock contraction and replay,
canonical CPU/CUDA science admission, and complete RKS/UKS cold/warm/displaced
SCF against independent PySCF/Libxc. The oracle receipts retain the actual shared
provider identity. The added opt-in switch generates all 34 source files
byte-identically to the qualified package; that one-off migration comparison is
not a permanent source-fingerprint test.

The compressed receipt retains source reconstruction, binary identities,
compiler/cache provenance, all point timings, complete endpoint outputs/work,
and independent oracle receipts. It records incomplete pilot retention and the
initial stale-mtime build that was discarded before qualification. Raw build
logs and binaries remain in ignored experiment storage.

```bash
python benchmarks/results/xc-parameter-sharing-20261027/reduce.py
```

`endpoint.py` accepts `--case h4|water`, `--name`, `--spins`, `--copies` and
`--output`; select matching source/library through `PYTHONPATH` and
`GENERATIVEQC_LIBRARY`, and always run it through finite `srun`.
`point-benchmark.cu`, `point-comparison.py` and `compile-trial.py` preserve the
experiment drivers. The latter two use the retained experiment directory
layout/compile commands; adapt those paths when reconstructing from the retained
base revision and patches. The controlled compile driver explicitly enables
sharing for the candidate. No real-GPU operation bypasses Slurm visibility.

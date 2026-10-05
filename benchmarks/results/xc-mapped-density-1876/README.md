# Actual local-AO XC density crossover

The compact GEMM candidate does not improve the production tile-256 domain:
48/96-atom complete fixed-density XC takes about 4.0%/2.8% longer. Generated
execution remains the incumbent. These are XC component endpoints, not PBE0
SCF/force speedups.

| atoms / AOs | tile | original generated / GEMM (s) | GEMM time change, original / moved |
| --- | --- | --- | --- |
| 48 / 384 | 128 | 1.6262 / 1.7563 | +8.00% / +7.20% |
| 48 / 384 | 256 | 1.0803 / 1.1234 | +3.99% / +3.94% |
| 48 / 384 | 512 | 0.7999 / 0.8121 | +1.53% / +1.62% |
| 96 / 768 | 128 | 3.4053 / 3.6348 | +6.74% / +8.91% |
| 96 / 768 | 256 | 2.3532 / 2.4191 | +2.80% / +2.82% |
| 96 / 768 | 512 | 1.8106 / 1.7957 | -0.82% / -0.87% |

The inputs reuse all four retained `pbe0-derivative-work-20261005/drivers/input-*.txt.gz`
geometry/basis files, spherical def2-SVP and the full v1 48×16×32 per-atom grid.
The positive diagnostic density is `density(nao, 1)` from the native XC test;
it is not a converged PBE0 density. Each comparison includes H2D density,
per-tile gather, complete XC evaluation, synchronization and E/V publication.
Basis/grid preparation, AO discovery and provider setup are reported separately.
Six samples per route retain the first as cold and use the median of the other
five as warm. Submission order alternates, including after geometry changes.

All 144 samples pass paired E/V/population checks. Maximum energy and potential
parity errors are `7.11e-15` and `2.22e-16`. This large-system comparator uses the
generated route; it is not an independent oracle. The indexed provider's
independent scalar, CPU E/V, capture and sanitizer gates are retained by #1961.
The same indexed gate passes again in this benchmark binary.

At tile 256, both geometries exactly reproduce the existing baseline's tile,
empty-map, maximum-active, active-sum and mapped/dense summand census. The
original 48/96-atom maps contain 342/536 AOs at most and preserve only
14.52%/5.44% of dense work. Logical gather traffic per evaluation is
2.37/7.09 GB (two matrix loads and one store per gathered element); these are
logical bytes, not measured DRAM traffic. The retained matrix cache remains
charged at global AO capacity, plus the separate 96 MiB provider allowance.

The generated route itself improves by about 23% for 96 atoms at tile 512.
This changes local domains and per-tile overhead; it is a candidate for separate
full-SCF/force qualification, not a default-promotion rule from this receipt.

`verify.py` reconstructs `summary.json` from the two compressed endpoint datasets,
checks every sample and work count, and matches both tile-256 geometries against
the previously retained composed baseline. `provenance.json` records the source,
binary and data hashes, actual compiler commands and qualification identity.
Build/Slurm scripts and ccache before/after statistics are included. All native
execution used finite Slurm `main`, `gpu:5090:1` on n1/node1; source hashes were
verified before execution. No CPU/reference integration enters measured CUDA.

```bash
python benchmarks/results/xc-mapped-density-1876/verify.py
# Inside a finite GPU Slurm job after building the native test:
gzip -dc benchmarks/results/pbe0-derivative-work-20261005/drivers/input-48-0.txt.gz > /tmp/xc-original.txt
gzip -dc benchmarks/results/pbe0-derivative-work-20261005/drivers/input-48-1.txt.gz > /tmp/xc-moved.txt
build/cuda-release-sm120/generativeqc_dft_cuda_tests --indexed-density-benchmark /tmp/xc-original.txt /tmp/xc-moved.txt
```

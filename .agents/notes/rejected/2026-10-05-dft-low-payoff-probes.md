# Decision: stop square-product tiling and exact-zero log truncation for DFT forces

Status: rejected
Date: 2026-10-05

## Problem and decision

Prioritize the large-system complete DFT energy/force endpoint. Two experiments
on base `04461baa3` did not justify changing production code. Their uncommitted
implementations remain in separate local checkouts; neither is part of this PR.

The retained 96-atom public-force profile attributes only 8.574354 ms to ten
ordinary spin square products, compared with a roughly 32 s endpoint. Replacing
the scalar products with a 16-by-16 shared-memory tile cannot materially close
that gap. CUDA event milliseconds were:

| Matrix size | Spins | Scalar | Tiled |
| --- | ---: | ---: | ---: |
| 48 | 1 | 0.0045888 | 0.00538773 |
| 384 | 1 | 0.067792 | 0.0737952 |
| 384 | 2 | 0.151556 | 0.123469 |
| 768 | 1 | 0.498403 | 0.483405 |
| 768 | 2 | 0.985845 | 0.957117 |

The independent long-double/mask/Graph harness passed 152 cases. The full KS
suite failed its J/XC operator census; no baseline comparison establishes the
cause. The attempted endpoint rejected a generic native build before execution.
These failures are not numerical or endpoint qualification. Preserve them in
`/data/jzzeng/qc-ks-prepared-products-1873/.artifacts/1873-ks-matrix/`.

The other experiment skipped `log(1)` and stopped an atom's log-product traversal
after its second exact zero. Zero, one-zero and multiple-zero adjoints remain
distinct, but the additional divergence did not pay off in the sampled domain:

| Atoms | Original phased Becke, ms | Zero-truncated, ms |
| --- | ---: | ---: |
| 48 | 11.0481596 | 11.1708961 |
| 96 | 28.1834402 | 28.3743687 |

These are isolated CUDA-event measurements over 64 sampled 256-point tiles,
16,384 points per case, synthetic signed seeds, and one n1 RTX 5090 finite Slurm
allocation per arm. Each arm internally uses cooperative/phased ABBAABBA; the
two arms themselves ran sequentially. They are not complete endpoint timings.
Both have zero maximum absolute gradient difference from the cooperative route.
91 host tests and 21 CUDA tests passed; 28 CUDA parameterizations skipped because
the probe specializes iteration 3. Evidence, emitted sources and cached compiler
receipts remain in
`/data/jzzeng/qc-becke-zero-logs-1894/.artifacts/1894-zero-logs/`.

## Revisit when

Square products need a different consumer profile in which they occupy a
material endpoint fraction. Exact-zero truncation needs a different execution
schedule or demonstrated point/atom coherence, not just fewer scalar operations.
Neither result rules out moving logarithms into the parallel pair-production
domain while retaining the original ordered atom reductions.

## References

- #1873, #1894, #1895.
- `benchmarks/results/pbe0-public-force-policy-20261005/README.md` and its
  `profile/statistics_cuda_gpu_kern_sum.csv`.
- Reproduce the Becke inputs using `benchmarks.becke_phased_probe --atoms 48 96
  --tiles 64 --tile-points 256`; compile with ccache, CUDA 12.9, `--fmad=false`,
  `--expt-relaxed-constexpr`, and `sm_120`, and execute through finite Slurm.

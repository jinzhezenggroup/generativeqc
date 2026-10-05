# Evidence: native XC rank-2k crossover at 384 and 768 AOs

Status: implemented qualification harness; production promotion deferred
Date: 2026-10-05

## Question and decision

#1953 found that rank-2k lost to generated potential assembly at 82/182 AOs.
Those measurements do not determine the crossover at the 384/768-AO extents
in #1876. Extend the same complete native fixed-density E/Vxc benchmark to
both exact sizes, with 128/256/512-point tiles and a rebuilt geometry. Retain
generated production selection while qualifying the large-domain candidate.

The harness requires the requested provider to be selected; optional-provider
fallback cannot masquerade as a library measurement. It records every timed
sample, full/tail reduction dimensions, semantic work, compact-panel bytes,
owner/provider preparation, arena/allowance and transfer/synchronization counts.
The existing `--potential-benchmark` keeps its 82/182-AO domain. The new
`--potential-large-benchmark` runs only by explicit request.

## Scope and oracle

Synthetic Cartesian s/f bases contain exactly 384 or 768 AOs on two centers.
Both geometries have 2,304 quadrature points; the second moves one center by
0.02 Bohr and rebuilds basis/grid/owner. Use restricted PBE coefficients with
exchange scale 0.75 and correlation scale 1.0, matching the semilocal part of
PBE0. This is a density-to-energy/potential endpoint, including actual AO,
density, point XC, panel packing, potential assembly, density H2D and E/Vxc
exports. It is not a full 48/96-atom PBE0 SCF/force benchmark.

An independent CPU integrator checks energy and every Vxc element at `2e-9`
absolute tolerance. All 24 shape/tile/geometry/provider endpoints passed,
including arena canaries. Library execution retains the canonical request and
prepared candidate from #1953; the test makes no new production vendor choice.

## Initial measured evidence

Slurm node1, main partition, RTX 5090, CUDA 12.9 / cuBLAS 120901. One cold
evaluation and five warm evaluations per endpoint; times below are warm medians
for the original geometry. Changed-geometry results preserve the same crossover.

| AOs | Tile points | Generated (ms) | Rank-2k (ms) | Rank-2k change |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 5.428 | 5.670 | +4.5% |
| 384 | 256 | 3.902 | 4.026 | +3.2% |
| 384 | 512 | 3.172 | 3.269 | +3.0% |
| 768 | 128 | 11.786 | 11.196 | -5.0% |
| 768 | 256 | 9.702 | 9.022 | -7.0% |
| 768 | 512 | 9.104 | 8.376 | -8.0% |

The independent maximum Vxc error is below `5.7e-14`. Six evaluations perform
2,043,740,160 scalar product summands at 384 AOs and 8,164,343,808 at 768 AOs,
identically for both providers and all tile sizes. Their product-call counts
are 108/54/30 for tiles 128/256/512. The last case includes a 256-point tail.
Both providers materialize 7,077,888 or 14,155,776 compact-panel bytes per
evaluation, respectively. Larger tiles change launch count and arena capacity;
they do not reduce semantic work.

Library preparation reserves a separate 96 MiB allowance, in addition to the
exact numeric arena (about 3.1–19.7 MiB across these cases). The first library
preparation takes 11.7 ms; later preparations take about 0.14–0.31 ms. Cold
latency includes CUDA lazy initialization (first generated/library calls were
31/90 ms), so it must not be replaced by the warm median in an amortization
decision. No full-SCF speedup is inferred from these fixed-density results.

Source/binary checksums, checked ccache compiler commands and before/after
statistics, device identity, scripts and complete outputs are under ignored
`.artifacts/1876-large-xc/` in `/data/jzzeng/qc-dft-xc-large-crossover-1876` on
the editing host and n1. The initial measurement is `endpoint-initial.log`;
the final run adds per-sample reporting and explicit provider-selection gates.
The final run again passes all 24 endpoints: 384-AO rank-2k is 2.94–4.67%
slower and 768-AO rank-2k is 4.99–8.00% faster across both geometries and
all tile sizes. Its executable SHA-256 is
`86da64a563d12863bdb99d1a110bc7c10b9a4ec9fe418d388aac6cad7d680bfd`.
The code baseline is #1953 head `c4018e09d`; only the qualification harness
changes, so this does not measure the later density-provider composition.

## Consequences and next gate

The old small-AO loss cannot be extrapolated to 768 AOs. Continue #1876 with
the density GEMM candidate on master's existing prepared density boundary,
preserving explicit density symmetrization, mixed arithmetic, strict refinement
and mapped/empty fallbacks. Qualify their composition against actual large
PBE0 cold/warm/geometry endpoints before changing production selection.

Do not infer a universal `nao >= 768` policy or promote the larger point tile
from these two-center fixtures. Local-AO sparsity, preparation amortization,
simultaneous provider resources and complete SCF/force work remain admission
inputs. Lower-priority millisecond-scale matrix migrations remain separate.

## References

- #1876 dense XC contractions; #1886 canonical prepared-provider boundary.
- [Original small-domain evidence](../architecture/2026-10-05-dft-potential-rank2k-lowering.md).
- `tests/native/dft_potential_lowering_cases.cuh`.

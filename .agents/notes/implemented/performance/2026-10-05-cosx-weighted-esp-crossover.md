# Decision: retain fused generated COSX ESP after endpoint crossover

Status: implemented
Date: 2026-10-05

## Problem

The weighted COSX ESP value region now has a shared prepared owner and two
complete recipes: fused generated execution and a pedantic batched matrix
product followed by in-place weight/finite publication. The projection/update
receipt in #1967 does not measure a library ESP recipe, so it cannot qualify
that recipe's promotion. A bare matrix-vector microbenchmark would also omit
its additional publication cost.

## Decision

Retain the production fused generated incumbent. Record the complete diagnostic
endpoint comparison at 384/768 AOs, tiles 64/256, original/moved geometry and
four independently admitted routes. Changing only ESP changes paired warm
medians by −0.209% to +0.0055% across generated and library projection/update.
Changing all three operations changes endpoint time by −0.242% to +0.034%.
This small measured difference on explicit coarse quadrature does not qualify
a production SCF/force promotion.

The common request and optional recipe remain useful architecture. The receipt
preserves every sample, rejected optional offer, scientific/semantic/precision
identity and six-site work count so future qualification can reuse the complete
contract without treating this limited domain as universal provider evidence.

## Rejected alternatives

- Promote from an isolated matrix product that excludes the weighting pass.
- Attribute endpoint time to a component without collecting a component split.
- Generalize this diagnostic density and 3×3×6 quadrature to converged production
  PBE0/COSX SCF or force behavior.
- Treat the first replay after CPU-subset qualification as process-cold CUDA or
  library initialization.

## Invariants

All three contractions retain their mathematical work and numeric storage.
Library ESP adds one publication pass per tile, with one scaled output per
point/AO. A single optional 96 MiB provider allowance serves all six full/tail
sites; generated reserves none. The fixed host reservation is 160 KiB.
An absent tail has zero calls and work. CPU/reference work stays outside every
timed build and remains absent from production execution.

The value-region finite publication policy is preserved. Derivative kernels
currently use stricter per-term finite guards through generated scalar helpers;
their migration needs its own semantic effects and acceptance gates.

## Evidence

`benchmarks/results/cosx-weighted-esp-1884/` retains 32 records / 192 endpoint
samples and reconstructs their summaries with `verify.py`. Maximum paired
raw/symmetric K error is `5.55e-16`, energy error `8.88e-16`; independent CPU
subsets at each actual AO dimension have K/energy errors below
`2.78e-17 / 8.67e-19`. The CPU subsets use three spread-out points and tile 2
to independently exercise full/tail arithmetic.

Slurm jobs 6000–6003 ran on n1/node1 with `main/gpu:5090:1` and finite 20-minute
limits. All successful allocations preserve assigned visibility and verify the
same 8,097 source files and binary hash. Compiler commands invoke ccache and
retain before/after shared-cache statistics. Exact source/binary/input/data
identities and build/qualification scripts accompany the tracked receipt;
complete manifests and raw logs remain in ignored local artifacts.

## Revisit when

A complete production quadrature/SCF/force qualification demonstrates a useful
gain with cold/warm/changed-geometry costs and independent numerical gates, or
a new complete fused candidate reduces publication or integral/data work.
These diagnostic timings alone do not authorize the default switch. #1884 and
#1886 remain open for derivative consumers, enclosing admission and the broader
provider boundary.

## References

#1884, #1886, #1967, #1968 and
[weighted ESP ownership decision](../architecture/2026-10-05-cosx-weighted-esp-provider.md).

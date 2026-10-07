# Decision: retain generated density contraction on the actual mapped XC domain

Status: implemented
Date: 2026-10-05

## Evidence

The [indexed provider](2026-10-05-indexed-xc-density-provider.md) preserves local
AO work and enables a meaningful comparison with the actual 384/768-AO water
baseline. Complete fixed-density XC on the full grid, including per-tile gather
and E/V publication, is slower with GEMM at tile 256: 3.94–3.99% for 48 atoms,
2.80–2.82% for 96 atoms. Tile 128 loses 6.74–8.91%; tile 512 loses 1.53–1.62%
for 48 atoms and wins only 0.82–0.87% for 96 atoms.

The retained receipt is
[`benchmarks/results/xc-mapped-density-1876/`](../../../../benchmarks/results/xc-mapped-density-1876/README.md).
It includes 24 endpoint records / 144 samples, cold and preparation costs,
changed geometry, exact semantic work and logical packing traffic. Both
geometries' tile-256 work census exactly matches the prior baseline. Matrix
storage remains bounded at global capacity; full dense work is never restored.

Maximum paired energy/potential errors are `7.11e-15` / `2.22e-16`. The large
comparison is against generated execution; the independent scalar/CPU,
changed-input capture and sanitizer qualification belongs to the indexed
provider gate, which is also rerun with this binary. The benchmark uses a
positive diagnostic density on real geometries, not a converged PBE0 state.

## Decision

Do not promote the indexed library candidate. Dense 768-AO GEMM profitability
cannot establish mapped profitability, even though both recipes lower the same
scientific request. Keep materialization identity and qualification independent,
and retain this negative endpoint evidence rather than using a dense shortcut.

A possible improvement is tile sizing: generated tile 512 itself is about 23%
faster than tile 256 for 96 atoms. That result trades fewer tiles for more local
AO summands, requires larger arena planning, and needs independent numerical
and complete SCF/force qualification. Do not attribute it to GEMM or silently
change the public resource/tiling policy here.

## Revisit when

A new recipe reduces map-factor traffic, batches work without restoring omitted
AOs, or demonstrably improves the complete resource-bounded SCF/force endpoint.
Capture-only timing cannot replace the current ordinary-stream exact-hybrid
consumer's costs. Keep the generated fallback and scientific precision/domain
unchanged while making these comparisons.

References: #1876, #1886, #1958, #1959, #1960, #1961.

## Follow-up: complete tile-control qualification

The [complete SCF-tile qualification](2026-10-06-pbe0-scf-xc-tile-qualification.md)
resolves the larger-tile endpoint question on a frozen RKS PBE0 cohort. It does
not reverse the indexed-library rejection here or promote a global tile default.

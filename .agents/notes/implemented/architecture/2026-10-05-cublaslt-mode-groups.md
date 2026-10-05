# Decision: prove cuBLASLt mode groups without rewriting semantic identity

Status: implemented
Date: 2026-10-05

## Problem and decision

Real source-response contractions contain multiple continuous free/reduction
axes. Restricting cuBLASLt to rank-2/3 would reject physical matrices already
represented by the canonical affine request. Do not create a flattened second
scientific operation or rely on matching products of extents.

The compiler and native provider independently derive ordered M/N/K and batch
groups from mode membership. Output mode order defines the free and batch
groups; the first operand defines reduction order. Every operand must prove
that lexicographic group coordinates have a single physical stride. Existing
row/column layout, leading-dimension and batch-overlap checks then apply to
those physical dimensions. Nonunit stride gaps and permuted group orders are
rejections; unit axes have no observable stride.

## Invariants and evidence

- Original modes, scientific identity and precision identity remain authoritative.
- No packing, output scatter, broadcast or new precision variant is introduced.
- Explicit provider reservations and prepare-only heuristic discovery remain.
- Python qualification compares grouped recipes to independent high-rank NumPy
  einsum across layout permutations, FP32/FP64, multiple batch axes and unit axes.
- Native qualification splits original matrix dimensions into semantic axes and
  checks real-device replay against independent affine addresses, including
  padding sentinels, beta updates, finite-error propagation and cached algorithms.

This extends provider eligibility only. Real-region integration and complete
endpoint/resource qualification remain separate #1888 work; no default promotion
or speedup follows from recognising a layout.

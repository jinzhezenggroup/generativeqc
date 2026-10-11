# Decision: immutable semantic HF/KS matrix-product routes

Status: implemented
Date: 2026-10-11

## Problem

The shared SCF adapter accepted an implementation boolean at every plain/spin
product. Moving GEMM alone would leave scientific execution free to switch an
already prepared route and would not establish canonical semantic ownership.

## Decision

Represent column-major square products, left transpose, physical/spin broadcast
and uniform scalar scaling in TensorIR. Emit ten canonical portfolios through
the existing lowering factory. The shared native selector retains the prepared
owner's incumbent; missing complete costs remain unknown, not synthetic timings.

The immutable binding borrows the existing stream/handle and records actual
device/runtime/provider versions. Shape/context compatibility is checked before
execution. SCF retains masked publication and precharged scratch. Vendor
submissions move into the shared tensor backend. HF/KS capacity accounting adds
a conservative 16 KiB preparation-host envelope for bounded selector temporaries.

## Rejected alternatives

- Rename the boolean: no canonical semantics or frozen compatibility results.
- Allocate a generic tensor plan per product: this duplicates arena/provider
  ownership and introduces replay-time work.
- Promote a new precision/crossover: structural or primitive evidence cannot
  qualify a complete scientific endpoint.

## Invariants

- Strict FP64 only; finite scalar scaling is an explicit SSA boundary.
- Iteration/capture performs no selection, allocation or provider discovery.
- Inactive/failed bytes remain untouched; invalid scratch never changes recipes.
- Runtime shape/context lifetime supplements symbolic AOT identities. This is
  not a new executable cache or resource owner.
- Optional exact-shape providers remain negative offers for the bounded domain.

## Evidence

Source-matched host/Slurm logs and cache snapshots are retained in ignored
`.artifacts/matrix-binding-20261011/`. NumPy independently checks asymmetric
views, scale and broadcasting. Host queue tests execute the actual selected-copy
body with changed masks/inputs and injected failures. The real-CUDA driver binds
each fixture before enqueue/capture. Forced primitive comparisons remain test
evidence, not complete HF/RKS/UKS energy/force or profitability acceptance.

## Consequences and revisit conditions

This advances #1873/#1886/#1890 and supplies a strict canonical consumer for
#1889. RHF profitability/resource/retry selector debt remains. Shell-span basis
transforms (#1878), rank-k consumers (#1877), orthogonalizer reconstruction
(#1883) and other routes require separate production/end-to-end gates. Revisit
family-wide incumbent retention when per-site executable recipes and complete
source-matched cost evidence exist.

## References

- `docs/developer/lowering_providers.md`
- `python/generativeqc_compiler/tensor/matrix_product.py`
- `src/tensor/cuda_matrix_product.hpp`
- `tools/qualify_ks_matrix_provider.sh`

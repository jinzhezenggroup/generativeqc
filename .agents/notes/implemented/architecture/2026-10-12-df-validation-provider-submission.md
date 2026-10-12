# Decision: share DF final-validation square-panel submission

Status: implemented
Date: 2026-10-12

## Problem

`df_final_validation.cpp` submitted its FP64 products directly to cuBLAS,
although the registered Tensor square-panel provider already supports the same
column-major square-padded buffers, transpose flags and occupied reduction rank.
This is one bounded ownership slice of #1890, not closure of that issue.

## Decision

Use `tensor::cuda::square_panel_product` with the existing borrowed plan handle.
Expose alpha as a host scalar with the backwards-compatible default 1.0. This
is necessary for the existing energy-weighted-density `1/spin_weight` product;
beta remains zero. All physical leading dimensions remain `nbf`, even when the
occupied reduction rank is smaller. Remove only the retired DF file's vendor
manifest entry; the Tensor provider's vendor inventory is unchanged.

## Rejected alternatives

- The compact row-major Tensor GEMM adapter derives leading dimensions from
  the logical reduction rank. It cannot express every square-padded transpose
  combination here without changing strides or repacking.
- Separate scale kernels or scratch would change work and lifetime/resource
  behavior for an ownership-only change.
- A new DF-local wrapper containing the vendor call would duplicate submission
  ownership rather than reuse the registered provider.

## Invariants

Preserve operand order, NN/NT/TN/TT flags, FP64, alpha/beta, reduction rank,
packed column-major storage and the explicit rank-zero asynchronous memset.
The plan still owns its handle, stream, modes, workspace and allocations.
The caller retains the original status-to-runtime_error conversion, including
ALLOC_FAILED; CUDA memset OOM remains bad_alloc. Existing ordinary and
exceptional Drain guards remain responsible for queued work and host staging.
No new transfer, allocation, admission, selector, reference fallback or equation
is introduced. Strict/reference/scalar and scientific acceptance gates remain.

## Evidence

`tests/python/test_df_final_validation_provider.py` includes the complete actual
production translation unit and shared header, with injected host vendor APIs.
It captures physical pointers, handle, transpose order, dimensions, alpha/beta
and leading dimensions, executes nonsymmetric products against an independent
NumPy long-double oracle, and checks errors and exceptional draining. Its 84
cases passed locally using GCC 16.2.0 through verified sccache 0.16.0. The
existing vendor-boundary suite passed 35 tests. Resource-ledger integration is
stubbed in this host probe; it does not establish actual CUDA allocation,
driver, device, endpoint or scientific acceptance. Real-device qualification
and latest-head independent review remain separate delivery gates.

## Consequences and revisit conditions

This change claims cleaner ownership only. The borrowed BLAS handle and full
canonical prepared LoweringRequest migration remain #1886/#1890/#933 debt.
Any later provider/layout/precision promotion needs its own independent science,
resource and complete-endpoint evidence. Revisit this primitive when a prepared
semantic binding can preserve the same operation and lifetime contract.

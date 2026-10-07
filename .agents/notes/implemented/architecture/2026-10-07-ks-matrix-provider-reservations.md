# Decision: keep KS matrix ownership and opaque reservations explicit

Status: implemented
Date: 2026-10-07

## Problem

The CUDA KS shared-matrix adapter adds a prepared optional cuBLAS handle per
eligible KS owner. Adding its 96 MiB allowance only to a diagnostic does not
admit it through the public resource plan. A successful generated fallback also
requires successful live provider teardown; hiding a failed destroy loses the
only retained handle and incorrectly admits continued execution.

## Decision

- Keep the 17-AO crossover and 96 MiB allowance in one native shape-only policy
  used by provider preparation and an additive private resource-inventory ABI
- Reserve each potentially live handle as `runtime_allowance` in the Python KS
  plan, separate from explicit numeric arenas and their allocation ledger
- Keep the existing three-output numeric inventory ABI unchanged; absent or
  failing additive inventory makes CUDA budget planning unavailable
- Retain the conservative reservation even if optional creation later falls
  back to generated execution; no concurrent item can reuse that reservation
- Check live destroy status before admitting fallback or discarding ownership;
  preparation failure leaves the handle owned for the caller's cleanup retry
- Admit only the shared allocation-measurement mutex header to the matrix
  layer; bucket/method, graph, and unrelated runtime ownership remain forbidden

## Rejected alternatives

Counting opaque provider bytes as numeric arena space would let the explicit
allocation ledger spend the provider reservation. Expanding the existing
three-output ABI would break older callers. A blanket runtime dependency
exception would erase independent ownership boundaries.

## Evidence and limits

The host-double regression executes the actual owner and private ABI, including
16/17-AO admission, allocation failure, capture rejection, oversized-provider
fallback, injected destruction failure, retained-handle retry, and device
restoration. Python tests use the compiled ABI through the production inventory
and planner, checking per-item budgets, exact/one-byte-short admission, unchanged
tiny/CPU routes, and exclusion of opaque bytes from the numeric ledger.

Host validation is not CUDA arithmetic, real allocation, graph-capture, or
endpoint-performance qualification. No speedup claim is made.

Reference: https://github.com/jinzhezenggroup/generativeqc/pull/2058

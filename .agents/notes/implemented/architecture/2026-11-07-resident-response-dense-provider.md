# Decision: prepared dense operations for resident response owners

Status: implemented
Date: 2026-11-07

## Problem and decision

The public RHF/UHF resident response adapter contained 29 direct GEMMs, repeated
vector operations and two independently spelled handle setup/cleanup sequences.
Those are execution-provider responsibilities, not response equations or public
method selection. Move them into `tensor::CudaDenseOperations`, retaining the
existing shared column-major Tensor GEMM/GEAM adapters rather than introducing a
second contraction implementation.

The host-safe declaration is consumed by the C++ API owner. Its CUDA translation
unit reuses the existing Tensor submission views and owns the supported Level-1
operations and optional-GEAM transpose-sum fallback. It allocates only the one
handle already admitted by each response owner, never another stream or numeric
buffer. No algorithm, precision, public C ABI, response vector lease or budget
changes.

## Lifetime and error invariants

- Create remains before the response owner's guarded setup; bind preserves the
  original stream-then-HOST-pointer-mode order and first failure. Math mode stays
  at its original provider default.
- The original response owner selects/restores the device, drains the stream,
  frees its arena and explicitly closes the handle at the same points. The
  wrapper intentionally does not add destructor-time CUDA work outside those
  lifetime boundaries. Close clears the private handle even on failure, matching
  the prior adapter's discarded destroy status and preventing a hidden retry.
- All execution statuses remain native statuses and cross the unchanged
  `resident_blas` allocation-versus-execution error mapping. GEMM retains every
  transpose, leading dimension, alpha/beta pointer, operand and output offset.
- `A + A^T` uses GEAM where declared; providers without it retain the original
  bounded copy followed by strided vector additions. There is no replay after
  failure. Successful submission counters include completed partial prefixes,
  including failure before publication of the complete sum.

## Evidence and remaining acceptance

`test_cuda_dense_operations.py` compiles the actual provider translation unit
with independent host recording providers in C++17/C++20, with and without GEAM.
It checks raw status/ABI forwarding, immutable stream/pointer mode, partial setup,
explicit/idempotent close, transpose-sum numerics, shape/overflow rejection and
failure-prefix work reporting. Together with neighboring resident-lifetime,
vendor-boundary and borrowed-view tests, 86 tests passed locally. CUDA 12.4.99
also compiled the complete new provider through verified sccache 0.16.0.

These are not real-GPU resident response or molecular E/F results. Source-matched
Slurm qualification is still required. This slice removes the API's direct-vendor
inventory entry but does not resolve the remaining CC/RHF selectors or SCF/GFN
provider audit tracked by #1890.

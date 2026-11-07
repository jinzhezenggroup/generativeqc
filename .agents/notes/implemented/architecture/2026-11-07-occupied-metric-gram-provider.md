# Decision: shared borrowed Tensor provider for the occupied metric Gram

Status: implemented
Date: 2026-11-07

## Problem

The occupied DF response emitter's remaining direct vendor calls were the
symmetric metric Gram's SYRK capability test, submission and GEMM fallback.
The [previous column-view migration](2026-10-21-occupied-df-response-column-views.md)
intentionally retained this branch as #1890 migration debt. Its vendor selection
belongs alongside the existing borrowed Tensor execution adapters, not in the
compiler-owned method equations.

## Decision

Add `ColumnGramView` and `submit_column_gram` to `src/tensor/cuda_runtime.cuh`.
The view expresses `C = alpha op(A) op(A)^T + beta C`, including physical input
and output leading dimensions and the triangle consumed by the caller. Compile-time
provider capability selects SYRK or the existing full-product GEMM adapter.
Dependent overload substitution, rather than C++20 `requires`, keeps the shared
header valid in its runtime-supported C++17 mode when SYRK is not declared.

The method still owns diagonal-first symmetric occupied-pair packing, diagonal
weight one, off-diagonal weight two, the packed stride, shape/overflow admission,
and its first-error return. Both reductions use `beta=1`. Native response owners
still own resources, the lower-triangle mirror and executed-work counters. The
provider reports triangular versus full-product work even on failure, preserving
the original report semantics without exposing a method-level provider selector.

Remove the emitter's retired vendor-inventory exception and classify the SYRK
references under the already-registered Tensor provider.

## Rejected alternatives

- Replacing SYRK with unconditional GEMM preserves the consumed triangle but
  widens executed work on providers that support triangular updates.
- Retrying a failed SYRK as GEMM risks reapplying an accumulated adjoint after a
  partial submission. Only compile-time absence selects the fallback.
- Unpacking or rescaling off-diagonal factors adds storage/data movement or
  changes floating-point operations without helping this ownership migration.
- Introducing a new provider handle, workspace or public selector would change
  the response's bounded resource/lifetime contract unnecessarily.

## Invariants and validation

- Borrow the admitted handle, host scalar pointers, stream and buffers unchanged;
  allocate nothing and do not change precision, pointer/math modes or workspace.
- Consume only the requested triangle until mirroring, including after the
  full-product fallback. Do not infer triangular work from mathematical symmetry.
- Host provider tests compile the production adapters in C++17/C++20 with each
  combination of optional SYRK and GEAM declarations. Recording providers verify
  argument/status forwarding and unchanged borrowed handle modes/stream.
- Numerical host tests compare the emitted weighted Gram with the independent
  full occupied contraction, and compare the shared primitive with an extended-
  precision Gram for both layouts/triangles, padded strides, signed coefficients,
  nonzero beta and beta-zero NaN overwrite. Existing finite-difference response
  checks remain in the targeted suite. Failure injection checks first/second
  product errors and the preserved partial adjoint without retry.
- Reproduction: `CCACHE=$(command -v sccache) PYTHONPATH=python python -m pytest -q
  tests/python/test_df_occupied_projection_lowering.py
  tests/python/test_df_occupied_response_provider.py
  tests/python/test_tensor_borrowed_cuda_views.py`.

Local validation on 2026-11-07:

- The targeted suites plus the neighboring charge/matrix providers, response
  projection/tile policies, Gram admission and compiler/vendor/provider boundary
  suites passed: **495 tests**.
- The complete generated DF-HF response header, included by a CUDA translation
  unit that calls the occupied metric Gram, compiled with CUDA 12.4.99 for both
  `-std=c++17` and `-std=c++20`, using `-arch=sm_80` for offline compilation.
  Compilation used verified sccache 0.16.0; no device code was executed.
- The vendor inventory checked 35 files with no errors; compiler structure
  checked 584 modules with no dependency errors. Ruff, changed-line clang-format
  and `git diff --check` passed.
- Ignored local evidence is retained under `build/issue1890/`, including the
  generated header, CUDA probe/objects, host test log, compiler versions and
  before/after compiler-cache statistics. Shared cache counters are not isolated
  per-task performance measurements.

Host evidence does not establish GPU numerics, full endpoint E/F qualification,
performance improvement or completion of all #1890 migrations. Real-device
qualification still requires an allowed, scheduler-allocated GPU; the available
local Slurm inventory was a drained `node3`, so it was not used.

## Revisit when

A canonical prepared Gram binding replaces borrowed submissions, or a provider
changes triangular capability/work reporting. Preserve the independent numerical
and lifetime/error gates through either transition.

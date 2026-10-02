# Decision: keep DF DIIS BLAS calls behind the DF SCF library provider

Status: implemented
Date: 2026-10-02

## Problem

Issue #762 requires external BLAS/LAPACK calls to stay behind shared runtime/compiler operation
boundaries rather than being embedded in scientific method code. The CUDA density-fitting DIIS
path owned the scientific residual sequence `FDS - SDF`, but `df_scf_diis.cpp` also submitted
`cublasDgemmStridedBatched` directly. That mixed equation ownership with the vendor-library
provider boundary.

## Decision

Keep the DIIS residual sequence, history/ring policy, normalization, dependent-history retirement,
and small solve in their existing owners. Add an explicit strided/accumulating GEMM operation to
`df_scf_library`, which already owns cuBLAS/cuSOLVER integration for DF SCF, and route the DIIS
matrix products through that provider. The existing `scf_gemm` entry point delegates to the same
operation for its ordinary unit-stride, alpha=1, beta=0 case.

## Rejected alternatives

- Moving the `FDS - SDF` composition into the library provider would hide scientific method
  structure inside runtime plumbing instead of fixing the ownership boundary.
- Generating DIIS history mutation or the augmented solve through TensorIR is outside this slice.
  TensorIR owns the pure Gram and extrapolation equations; stateful ring/solver policy remains
  native until its separate CUDA qualification is complete.
- Replacing cuBLAS with a handwritten GEMM has no correctness or endpoint-performance
  justification and would create another dense-linear-algebra implementation.

## Invariants

- The four residual products retain their existing order, matrix interpretation, strides, and
  alpha/beta coefficients.
- The provider borrows the existing plan stream and cuBLAS handle; no host synchronization or
  allocation is added.
- DIIS history mutation, normalization, dependent-history retry, and the augmented solve remain
  native runtime/solver policy.
- Scientific DF DIIS method code must not call cuBLAS directly.
- No numerical tolerance, capability gate, or fallback policy changes in this ownership refactor.

## Evidence

A structural regression test requires `df_scf_diis.cpp` to contain no cuBLAS call, requires the
DIIS path to consume `scf_gemm_strided`, and confirms the vendor batched GEMM remains in
`df_scf_library.cpp`. CUDA compilation and existing endpoint tests remain the integration gates.

## Consequences

The DF SCF library provider API now exposes explicit strides and accumulation coefficients so
scientific callers can express existing matrix-layout requirements without owning the vendor API.

## Revisit when

Revisit this boundary if the full CUDA DIIS residual/Gram/extrapolation path gains a qualified
dynamic/ragged compiler lowering that can replace the current native scientific consumer without
absorbing solver/runtime policy.

## References

- #762
- `src/scf/cuda/df_scf_diis.cpp`
- `src/scf/cuda/df_scf_library.hpp`
- `src/scf/cuda/df_scf_library.cpp`

Agent: ChatGPT
Model: GPT-5.6 Sol

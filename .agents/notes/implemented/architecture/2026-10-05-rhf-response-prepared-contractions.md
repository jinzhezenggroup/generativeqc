# Decision: prepare RHF frame contractions at the shared provider boundary

Status: implemented
Date: 2026-10-05

## Problem

Physical RHF frame response owned a BLAS handle and injected a GEMM callback into
generated TensorIR traversal. Its `matrix_blas` option exposed implementation
choice to scientific callers and duplicated the shared contraction provider's
resource, layout and finite-result responsibilities.

## Decision

Emit canonical typed descriptors for the existing five RHF programs: primal,
potential seed, weights, density direction and orbital action. The response owner
prepares one table per stage against one shared context, before any traversal.
Generated code borrows those tables. Non-contraction nodes reuse the original
scalar kernels and arena layout, preserving the scientific graph identity.

The complete numeric budget includes descriptor host storage and the shared
provider allowance. Dimension/budget rejection or unavailable optional context
storage selects the original complete scalar CUDA traversal. Preparation errors
other than the shared context's recognized optional-resource rejection propagate;
there is no retry after partially executed numerical work. Neither fallback nor
preparation changes the requested backend or precision.

## Rejected alternatives

- Moving the callback to another method-local helper would retain duplicated
  provider ownership and erase the canonical request at the execution boundary.
- Keeping a production BLAS/scalar boolean would preserve the implementation
  selector being retired by #1890. The validation-only probe instead imposes a
  real budget below the optional reservation, without increasing caller limits.
- Using the prepared traversal for the final orbital residual would remove the
  independent lowering check. It always temporarily unbinds the stage tables.

## Invariants

- Preserve exact, unscreened, self-adjoint J/K actions for signed response density.
- Keep sticky intermediate finite checks across AD and composed J/K actions.
- Keep the direct-plan stream alive until tables, context and all buffers die.
- Audit the final residual with scalar CUDA and restore bindings on both success
  and exception.
- Do not interpret prepared-call counts as complete endpoint work or a speedup.

## Evidence

`test_native_contraction_binding.py` validates semantic matrix projections for all
five RHF programs. `test_rhf_frame_response_codegen.py` compares generated CPU
maps to the independent interpreter and checks shared scalar kernel emission.
`test_rhf_frame_response_cuda.py` compares H2, water and LiH gradients to PySCF,
nonzero orbital response to central molecular energy differences, and prepared
versus scalar execution at exact and one-byte-below resource boundaries. Public
RCCSD energy/force gates exercise this owner through its production consumer.

Qualification commands and results are retained locally in `.artifacts/1890-rhf/`.

## Consequences and revisit conditions

This removes method-local vendor ownership while retaining an explicit scalar
fallback. The current shared table implements the admitted homogeneous FP64
schedule; admitting other precision or provider portfolios requires independent
qualification and complete endpoint evidence under #1886/#1889.

## References

- #1886, #1890 and the conventional RCCSD migration in #1868.
- [Current provider contract](../../../../docs/developer/lowering_providers.md).

# Decision: Describe SCF basis transforms as bounded tensor stages

Status: implemented
Date: 2026-10-08

## Problem

The CUDA SCF basis conversion uses column-major public/direct buffers and four
hand-written reduction kernels. A provider-neutral dense route needs the same
mathematics and physical addresses, including nonsymmetric inputs, before any
production selection policy can be qualified.

## Decision

TensorIR owns the two density and two Fock contraction stages. Its row-major
`transform[direct, public]` view is the transpose of the existing column-major
`C[public, direct]` allocation; the same interpretation applies to matrix
inputs, temporaries and outputs. The stages therefore represent
`C.T @ D_public @ C` and `C @ F_direct @ C.T` without a packing buffer or a
symmetry assumption. The graph also represents the final `hcore` addition.

The compiler emits canonical requests and bounded-region candidates for the
shared CUDA executor. This qualification only admits one packed, unmasked
matrix with no shell spans. Its generated guard rejects other domains and
address-range overflow. The existing production SCF path retains ownership of
activity, spin broadcast, shell spans, allocation and execution until the SCF
integration is separately qualified.

## Rejected alternatives

- Hard-code four cuBLAS calls in SCF: that would duplicate provider ownership
  and leave the mathematical request outside the compiler.
- Materialize transposes or rely on symmetric density/Fock matrices: neither
  follows from the storage contract, and both can conceal orientation errors.
- Switch production callers as part of descriptor qualification: active masks,
  spin broadcast, span selection and resource admission still need full endpoint
  evidence.

## Invariants

Preserve FP64 semantics, the exact public/direct orientation and every resolved
shape/stride in the native binding. A bounded provider may run only after the
packed-domain guard; the production path remains unchanged for other domains.
Do not infer production performance or complete SCF correctness from this
stage-only qualification.

## Evidence

The Python lowering tests compare asymmetric rectangular matrices with an
independent extended-precision oracle and check deterministic emission and
negative dimensions. A CUDA probe directly consumes the generated header and
shared `PreparedBoundedContraction`: on an H100 with CUDA 12.8 it reads back
all four stages and the `hcore` fold for rectangular cases and real d/f AO
expansions, under both cuBLAS and `generated.cuda`. The largest observed
per-stage absolute error in those cases was below `3e-15`. The probe also
checks unsupported domains, stride, alias and stage-identity rejection. This
evidence establishes stage behavior only; no complete energy/force or timing
gate is claimed.

## Revisit when

The SCF owner has source-matched active-mask, spin, shell-span, resource and
provider-selection contracts, plus complete energy/force and endpoint timing
evidence for a production route.

## References

- Issue #1878.
- `python/generativeqc_compiler/tensor/basis_transform.py`.
- `tests/python/test_basis_transform_lowering.py`.
- `tests/native/basis_transform_stage_probe.cu`.

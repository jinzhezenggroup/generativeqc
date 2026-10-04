# Decision: retain semantic identity through source-response row projections

Status: implemented
Date: 2026-10-05

## Problem

The shared DF MO source reverse program traverses raw source rows twice. Three
of its eight static contraction sites fix a leading axis, including one reduced
axis whose contributions accumulate into the coefficient adjoint. Describing
those sites only as GEMMs would discard their original TensorIR semantics.

## Decision

Project the canonical request through explicit leading fixed modes and operand
order. Preserve the source equation and parent semantic identity; derive the
projected shape, strides and semantic work from the remaining modes. Reject
interior slicing because this traversal supplies dense contiguous rows.

Emit row-major descriptors and an additional prepared callback traversal from
the same eight compiler steps. The original column-major callback stays usable
while native consumers migrate. Runtime provider selection is a separate change.

## Invariants

- No new scientific algebra, symmetry inference or precision admission.
- Every fixed reduction row contributes once with beta=1 after initialization.
- Two complete source reads, `3*n+5` calls and `6*n**3*q+2*n*n*q*q` summands.
- The existing two full scratch tensors and two row buffers suffice.
- Template identities describe representative compiler shapes; native validation
  binds the actual runtime extents, including dimensions of one.

## Evidence

`test_native_row_projection.py` verifies identity, operand order and rejection.
`test_df_cc_source_program.py` validates all eight descriptors over sixteen
shape pairs and executes both native traversals against the independently
contracted complete response expression, including nonsymmetric inputs.
The response numerical gate remains atol=3e-11 and rtol=3e-13.

## Revisit when

A consumer needs interior strided slices or noncontiguous row views. Such a
consumer must supply explicit physical strides and cannot relax this dense-row
contract by merely dropping axes.

References: #1886, #1887, #1890; `docs/developer/lowering_providers.md`.

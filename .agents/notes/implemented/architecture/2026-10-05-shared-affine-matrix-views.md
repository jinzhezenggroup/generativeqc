# Decision: share affine matrix layout proofs across execution providers

Status: implemented
Date: 2026-10-05

## Problem

cuBLASLt introduced a proof that existing semantic M/N/K/batch mode groups have
native matrix layouts. Adding CUTLASS/CuTe must not duplicate that address proof
or require an AOT provider to depend on a vendor-specific adapter.

## Decision

Move the pure compiler recipe and native matrix view proof into shared tensor
modules. The canonical contraction still owns scientific identity, precision,
original modes, strides and alias restrictions. The view proof resolves only
physical row/column layouts, contiguous groups and nonoverlapping batches.
Provider-specific legality and execution remain in their provider owners.

The existing cuBLASLt compiler adapter preserves its request/recipe/candidate
payloads and rejection wording. Native users move directly to the shared header;
there is no duplicate layout parser or forwarding native header.

## Invariants and evidence

Preserve logical mode order, unit-axis handling, unequal extents, padding,
transposed output, batch bounds, checked arithmetic and the rank-eight limit.
The existing compiler NumPy oracle and native FP32/FP64 cuBLASLt replay probe
exercise these properties. Recipe/candidate identity comparison across the move
must remain unchanged; compiler source provenance intentionally changes.

## Revisit when

A new provider needs a physically different view, such as broadcast or indexed
operands. Extend the shared proof only after independent address-oracle coverage;
keep unsupported layouts explicit rather than inferring from equal sizes.

## References

- #1886 and #1888.
- [Current provider contract](../../../../docs/developer/lowering_providers.md).

# Decision: bind physical rank-k output before the logical update

Status: implemented
Date: 2026-10-10

## Problem

The dense update equation accepted an unconstrained old-output matrix, whereas
native validation and publication read only its upper triangle and mirrored the
result. For coefficients `[1, 2]`, weight `1`, alpha/beta `1`, and old output
`[[10, 20], [30, 40]]`, the unbound dense equation gives `[[11, 22], [32, 44]]`
but native execution gives `[[11, 22], [22, 44]]`. Reserved scalar/output input
names could also collide with distinct source operands.

## Decision

The nonzero-beta update root takes a symmetric logical input named
`rank_k_bound_old_output`. A generated reader binds the raw physical buffer to
that input by reading `(min(p,q), max(p,q))` in the selected matrix order. All
native validation and publication sites use this reader; its source is part of
request semantic identity. Reference execution explicitly materializes the same
logical input with `symmetric_rank_k_bind_old_output`.

The separate overwrite root remains `alpha * Gram`. For `beta=+0/-0`, the reader
returns positive zero without inspecting the output buffer and native execution
selects the overwrite helper, preserving multiply signed-zero semantics. Source
inputs using reserved rank-k binding names fail closed before composition.

## Rejected alternatives

- A symmetry assertion on the raw buffer would reject currently admitted lower
  poison instead of defining which physical cells are authoritative.
- Keeping an unconstrained dense input would leave scientific identity different
  from the native operation.
- A new conditional TensorIR primitive is unnecessary because overwrite and
  update already have distinct admitted roots.

## Invariants

- Nonzero update reads only the physical upper triangle and mirrors publication.
- Zero beta does not read old output, including NaN or null test inputs.
- Binding source changes invalidate update request identity but not overwrite.
- Original source/precision admission and transactional all-output publication
  remain unchanged.

## Evidence and scope

Focused interpreter tests compare both roots with an independent
upper-authoritative oracle over both storage orders, signed beta, poisoned lower
cells, and zero-beta no-read. Collision and request-identity regressions fail
closed. qz Job `i1877-rankk-h100-1010z3` compiled exact commit
`3715893dc2f15789bf0b8cd52384a6558ace4c6e` and passed all 16 H100 cases for the
emitted reader and both providers. This remains qualification-only and is not a
production caller or complete method endpoint.

## Revisit when

A general physical-to-logical tensor binding abstraction can preserve the same
no-read, symmetry, identity and transactional contracts without broadening
admission.

## References

- #1877 and PR #2174
- `python/generativeqc_compiler/tensor/symmetric_rank_k.py`
- `src/tensor/cuda_symmetric_rank_k.cuh`

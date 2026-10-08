# Decision: static signed-axis semantics before Array API conformance

Status: implemented
Date: 2026-10-09

## Problem

The experimental public array namespace could trace generic transpose/reshape/broadcast but did not provide standard shape helpers, and its `permute_dims` rejected negative axes explicitly required by Python Array API 2025.12. Naively adopting NumPy conveniences on the public eager path alone would diverge from canonical TensorIR capture. General physical-chemistry index spaces cannot be treated as anonymous dimensions merely because extents match.

## Decision

- Normalize signed axes in one compiler-owned helper before constructing existing TensorIR `transpose`, `reshape`, `broadcast` or `gather` nodes.
- `expand_dims` supports 2025.12 tuple axes; positions are validated against the final rank and must be unique. `squeeze` requires explicit singleton axes. Both are admitted for anonymous/generic shapes, not scientific-domain reshaping.
- `moveaxis` is an axis permutation and can retain scientific domain identity; `permute_dims` now supports signed axes without altering the TensorIR rank/type contract.
- `broadcast_shapes` is a static tuple calculus; `broadcast_arrays` explicitly broadcasts each generic input independently and preserves each operand dtype. It does **not** imply implicit promotion when operands are later combined.
- Generic `flip` lowers reverse slices through existing immutable TensorIR gather/reshape. To avoid O(extent) unbounded static index tables, reverse axes larger than 65,536 fail explicitly during capture; eager NumPy remains a useful reference and does not have this limit.

## Rejected alternatives

1. Eager-only shape functions: normal Python array programs would still fail under capture.
2. TensorIR scientific-domain relabeling through shape-only generic operations: invalid physical index identity.
3. Adding a new TensorIR `flip` primitive just to increase interface coverage without AD/lowering/runtime qualification.
4. Unbounded static gather positions: graph/source metadata would scale silently with long reversed axes.

## Invariants

- Unchanged numerical element order and existing TensorIR AD, serialization and program identity rules.
- No new runtime-only IR opcodes or implicit device moves.
- Fail explicitly on duplicate/out-of-range axes, unsupported scientific shape remapping and excessive gather maps.
- Keep `__array_namespace__` and compliance claims disabled until broader dtype and operation conformance gates pass.

## Evidence

`tests/python/test_array_api_shapes.py` covers NumPy vs captured execution for signed/multiple axes, float32/64, scalar and empty shapes, broadcast dtype retention, TensorIR JVP, explicit QC index preservation and invalid controls.

Review qualification: `flip` constructs gathers only for reversed axes. Routing
through generic slice indexing would also enumerate the untouched axes before
building slice nodes, defeating the compiler work bound for shapes such as
`(10**12, 2)` reversed only on the last axis. A symbolic regression verifies this
case without allocating an array, checks that only two gather positions are
retained, and also covers an empty axis tuple. The focused creation, shape,
experimental facade, and frontend suites pass together (421 tests).

## References

- #2111, #633, #2061
- https://data-apis.org/array-api/2025.12/API_specification/manipulation_functions.html

Agent: ChatGPT
Model: GPT-6

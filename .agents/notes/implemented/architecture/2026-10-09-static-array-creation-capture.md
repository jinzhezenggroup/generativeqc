# Decision: context-local static creation in the experimental Array API

Status: implemented
Date: 2026-10-09

## Problem

The public Array API preview had eager NumPy array constructors only through `asarray`. Calling a shape-only constructor inside a compiled Python function previously returned an ordinary NumPy value, not a traced TensorIR node; passing that value into symbolic expressions could not build a canonical TensorIR graph. Constructing dense literal arrays for uniform values would also make source/serialization size scale with the output element count.

## Decision

- Keep `generativeqc_compiler.array_api` the symbolic owner and TensorIR the only mathematical IR.
- Wrap one execution of `trace(function, inputs)` in a `ContextVar[bool]` capture scope. Reset the exact token in a `finally` block on successful and failing traces. This is thread/task-local and nested-safe; it is not a global mode switch.
- Public `zeros/ones/full` distinguish eager NumPy reference behavior from compilation-time capture through that scope. Their `*_like` forms instead distinguish symbolic from eager inputs directly.
- Encode each uniform symbolic array as **one** finite exact scalar `constant` and an existing `broadcast` into anonymous generic dimensions, avoiding O(product(shape)) compile-time payload materialization.
- Keep type and device boundaries explicit. Only float32/float64 and host reference execution are qualified. `device` must be omitted/`None`. An integer or boolean `full` fill without an explicit supported floating dtype is rejected, not silently cast in contradiction of the Array API's default dtype rule.
- `*_like` on explicit scientific AO/occ/vir/aux/spin tensors fails closed instead of converting scientific index spaces into anonymous generic shapes.

## Rejected alternatives

1. Always return NumPy arrays from shape-only constructors, including inside traces: these cannot be combined safely with `VibeArray` operations and are not captured as TensorIR nodes.
2. Materialize one `constant` literal per output element: creates O(N) compiler graph payloads and serialization costs for uniform arrays.
3. Interpret input shape as sufficient to reconstruct all scientific TensorIR metadata: silently erases scientifically meaningful axis and representation identity.
4. Advertise full Array API conformance: missing dtype families, special values, constructor coverage, device protocols and standard inspection still prohibit any such claim.

## Invariants

- No Python tracing or eager NumPy execution in native prepared SCF/CC hot paths.
- No global tracing state leaks across exceptions or concurrent tasks.
- No implicit device transfer, default integer type emulation by fake float arrays, scientific-domain relabeling or backend-specific graph semantics.
- Constant payload remains scalar-size independent of requested static shape.

## Evidence

`tests/python/test_array_api_creation.py` checks reference/eager parity across float32/float64 and zero/scalar/matrix shapes; captured TensorIR topology, large-shape one-scalar payload, JVP, scientific fail-closed cases, device/dtype diagnostics, and exception reset.

## Consequences

This is a **bounded** array creation addition, not a full array creation protocol. Dynamic symbolic shapes, integer/bool/complex dtype defaults, `empty`, external devices and full standard conformance remain open under issue #2111.

## References

- #2111 (Array API completion tracker)
- #633 and #2061 (foundation and public preview)
- https://data-apis.org/array-api/2025.12/API_specification/creation_functions.html

Agent: ChatGPT
Model: GPT-6

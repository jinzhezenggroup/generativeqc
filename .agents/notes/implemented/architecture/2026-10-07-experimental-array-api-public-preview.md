# Decision: expose the bounded Array frontend as an experimental public API

Status: implemented
Date: 2026-10-07

## Problem

The compiler-owned Array frontend is already used by production-adjacent TensorIR
construction and has qualified DLPack interoperability, but external users would
have to import `generativeqc_compiler.array_api` directly. That path exposes an
internal ownership package and gives no public stability or capability boundary.
At the same time, the implemented subset is intentionally smaller than the
Python Array API standard and must not advertise conformance.

## Decision

Expose a curated facade at `generativeqc.experimental.array_api`.

- Keep `generativeqc_compiler.array_api` as the sole implementation and semantic
  owner.
- Make the ordinary public path shape/dtype based: `@array_api.compile`
  specializes from concrete runtime arguments, so users do not need TensorIR
  `IndexSpace`, `Index`, or `TensorSpec` declarations.
- Re-export the canonical symbolic operations, tracing helpers and DLPack
  boundary; retain the explicit TensorIR constructors only as an advanced
  scientific-annotation path.
- Mark the facade with its own experimental `API_VERSION` and capability fields.
- Keep `VibeArray.__array_namespace__` absent and continue reporting no Array API
  version.
- Return the canonical TensorIR `Program` from both explicit `trace` and the
  inferred `CompiledFunction.lower(...)` path; do not introduce a public wrapper
  IR or alternate program identity.
- Generic public arrays use anonymous shape-based axis identities and standard
  shape broadcasting/indexing/matmul semantics. Explicitly scientific arrays
  retain strict QC domain identity and fail closed when shape-only operations
  would erase that metadata.
- Inferred runtime inputs are non-differentiable by default. The public compile
  boundary accepts an explicit tuple of differentiable parameter names and maps
  only those inputs to differentiable TensorIR parameters.
- Keep experimental package import lazy.

## Rejected alternatives

### Expose `generativeqc_compiler.array_api` as the user contract

Rejected because it would turn compiler package layout and ownership into public
compatibility surface and make future compiler reorganization unnecessarily
breaking.

### Add `generativeqc.array_api` as a stable top-level API now

Rejected because the operation, dtype, broadcasting, device, and error-semantics
coverage is intentionally incomplete and has not passed a versioned Array API
conformance matrix.

### Implement `__array_namespace__` for discoverability

Rejected because the protocol is a standards-discovery signal. Returning a
bounded non-conformant namespace would misrepresent compatibility to generic
array consumers.

## Invariants

- Public capture must lower to exactly the same TensorIR program identity as the
  canonical compiler frontend.
- Generic public arrays may use ordinary shape broadcasting. Scientifically
  annotated arrays must not gain compatibility merely because extents match.
- Unsupported dtype promotion, dynamic shapes, Python control flow, and other
  undeclared behavior must fail closed.
- Equal extents must not erase AO/occupied/virtual/auxiliary/spin/batch domain
  identity.
- Public exposure must not move Python tracing into native SCF/CC steady-state
  execution.
- DLPack interop must retain explicit same-device and zero-copy checks and must
  not imply arbitrary external autograd support.
- No Array API conformance claim or discovery hook is allowed without a declared
  standard version and conformance evidence.

## Evidence

- Public-facade tests compare the public trace logical hash with the internal
  compiler frontend for the same symbolic equation.
- Inferred-call tests cover `(C * occupation) @ C.T`, generic broadcasting,
  exact scalars, reshape/indexing/newaxis, empty slices and batched matmul without
  any TensorIR type declarations.
- A declared differentiable inferred input is exercised through TensorIR JVP;
  undeclared inferred inputs remain non-differentiable.
- Import tests verify `generativeqc.experimental` does not eagerly activate the
  Array frontend.
- Capability tests require the public report to remain explicitly experimental,
  with `array_api_version=None` and `array_namespace_protocol=False`.
- Existing issue-#633 TensorIR identity, AD, fail-closed domain, and DLPack tests
  remain the implementation-level regression suite.

## Consequences

Ordinary users can now write NumPy-like symbolic expressions without importing
an internal compiler package or declaring TensorIR metadata. Advanced
quantum-chemistry code can opt back into explicit AO/occupied/virtual/auxiliary
spaces when those semantics matter. The facade still adds a public maintenance
obligation: generic array behavior, strict scientific behavior, the capability
report and user documentation must evolve together.

## Revisit when

Reconsider promotion to a stable `generativeqc.array_api` surface and
`__array_namespace__` only after a declared Python Array API version has an
explicit conformance matrix covering operations, dtype promotion, broadcasting,
device semantics, errors, and interoperability. Promotion should also preserve
the same TensorIR scientific-domain guarantees.

## References

- #633
- `docs/user/experimental_array_api.md`
- `docs/developer/array_api_frontend.md`
- `.agents/notes/implemented/architecture/2026-09-20-array-api-tensorir-frontend.md`

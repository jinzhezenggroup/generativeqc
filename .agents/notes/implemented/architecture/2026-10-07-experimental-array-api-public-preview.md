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
- Re-export the canonical symbolic operations, tracing helpers, DLPack boundary,
  and the minimal TensorIR type constructors needed to define symbolic inputs.
- Mark the facade with its own experimental `API_VERSION` and capability fields.
- Keep `VibeArray.__array_namespace__` absent and continue reporting no Array API
  version.
- Return the canonical TensorIR `Program` from `trace`; do not introduce a public
  wrapper IR or alternate program identity.
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
- Unsupported broadcasting, dtype promotion, dynamic shapes, Python control
  flow, and other undeclared behavior must fail closed.
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
- Import tests verify `generativeqc.experimental` does not eagerly activate the
  Array frontend.
- Capability tests require the public report to remain explicitly experimental,
  with `array_api_version=None` and `array_namespace_protocol=False`.
- Existing issue-#633 TensorIR identity, AD, fail-closed domain, and DLPack tests
  remain the implementation-level regression suite.

## Consequences

Advanced users can now write symbolic tensor expressions without importing an
internal compiler package, while the project can still evolve the preview
surface before committing to standards compatibility. The facade adds a small
public maintenance obligation: changes to the compiler subset must keep the
public capability report and user documentation synchronized.

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

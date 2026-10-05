# Decision: bind COSX derivative scalar checks to shared prepared sites

Status: implemented
Date: 2026-10-05

## Problem

COSX derivatives still owned native dense projection loops after the value path
moved to shared lowering. Their arithmetic contract is stricter: the existing
scalar TensorIR helper checks every accumulator/input and every published scalar
update. A final matrix audit cannot preserve failure before later cancellation
or a zero ESP weight. Replacing these loops with the value-path recipe would
silently change that contract.

## Decision

Retain both scientific authorities: the existing dense TensorIR projection or
weighted ESP graph, and its existing scalar update/publication programs. A
bounded common recognizer binds their original hashes, strict FP64 and explicit
exact-order/finite effects. It recognizes only the current scalar accumulator
plus binary product and optional scalar weight publication.

Extend the existing prepared contraction executor to invoke those generated
helpers at each increasing reduction index. Hashes are validated before enqueue;
ordinary execution cannot bypass a checked binding. The shared runtime owns the
matrix loops and zero/sticky-error publication. Generated descriptors bind the
scalar helpers; native consumers supply borrowed addresses and fixed slots.
Unsupported optional provider recipes reject
the complete request, including under qualification hooks.

Molecular value projection prepares two full/tail sites, reserving 96 KiB of host
storage. Explicit-point derivatives prepare six sites for value projection,
three spatial jet slices and weighted value ESP, reserving 160 KiB. Both use one
existing stream/context and no provider device allowance or new numeric cache.
Jet slices retain their parent TensorIR identity; the compiler traverses each
axis without a repeated density allocation or packing.

Optional execution diagnostics are copied only after the complete synchronized
response succeeds. They preserve immutable generated identities after owner
destruction, actual calls and scalar summands, and weighted publication counts.
The molecular pure resource estimate uses the portable shared reservation API.

## Rejected alternatives

- Reconstruct scalar arithmetic in a new handwritten checked kernel.
- Replace intermediate checks with only a final output audit.
- Select a library despite lacking a complete exact-order checked recipe.
- Expand three density copies or pack spatial jets to obtain a batched GEMM.
- Treat the remaining composed response or molecular pullback as migrated just
  because this binary projection now has a shared owner.

## Invariants and qualification

Preserve original scalar-helper rounding, intermediate failures, zero-on-invalid
publication, sticky errors, borrowed layouts and full/tail work. Direct native
gates compare every prepared projection/jet/ESP shape to independent long-double
loops and to the retained incumbent CUDA helper traversal for exact rounding.
Partial overflow followed by cancellation, NaN with zero weight, attempted
ordinary execution, mismatched helper identity and optional qualification are
explicit rejection/failure cases.

Existing complete CPU analytic derivative, changed-geometry, density convention,
spherical s/d/f, translational and lifetime gates qualify the real consumers.
Shared provider regressions and full memcheck/initcheck protect ordinary request
execution. Source/binary manifests, finite Slurm assignments, compiler commands
and ccache statistics belong in ignored `.artifacts/1884-cosx-checked/`.

## Consequences and revisit

Generated execution remains the only supported checked recipe. This is an
ownership migration without a performance promotion. Composed ESP response,
symmetric projection, bidirectional potential, pullback contractions and
enclosing Fock resource admission still require subsequent slices. #1884 and
#1886 remain open.

## References

#1884, #1886, #1968 and
[weighted value ESP decision](2026-10-05-cosx-weighted-esp-provider.md).

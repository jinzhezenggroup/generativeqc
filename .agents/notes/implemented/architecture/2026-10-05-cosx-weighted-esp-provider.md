# Decision: prepare weighted COSX ESP as a complete contraction region

Status: implemented
Date: 2026-10-05

## Problem

COSX projection and exchange update already consume shared prepared sites. The
remaining value-path ESP application combines a point-batched matrix-vector
product, one weight per point and checked publication in a single native kernel.
Migrating only its bare matrix product would lose that complete contract and
could hide the cost of an additional weighting pass or duplicate provider owners.

## Decision

Keep the existing two-node TensorIR graph: a binary point-batched contraction
followed by a leading-batch weight broadcast. A bounded common region recognizer
retains the original program/root identity, requires exclusive intermediate
liveness and homogeneous arithmetic, and projects all three external inputs into
the canonical request. The nonfinite publication effect is explicit. It cannot
change weight axes, reorder outputs, donate a seed or silently cast an input.

Extend the existing prepared matrix table with an optional validated batch-scale
view. Generated and library recipes compete for that same complete request.
The generated recipe retains the increasing-column FMA chain and a single fused
weighted/finite publication kernel. The library recipe uses a strided-batched
matrix product and one combined in-place weight/finite publication pass. The
unweighted intermediate is not visible to another consumer and requires no extra
numeric allocation. An invalid publication becomes zero and sets the sticky
error before the next contraction can consume it; host success remains forbidden.

The six COSX sites (projection/update full/tail and ESP full/tail) share one
context on the existing grid stream. Their bounded host reservation is 160 KiB;
all selected library sites share the existing optional 96 MiB device allowance.
Missing resources or optional preparation rejection retain generated execution.
Replay changes only addresses/slots and performs no allocation or selection.
Scaled elements and split publication passes are counted separately from scalar
summands. Scale/output aliasing and missing/undeclared scale arguments are errors.

## Boundaries

The weighted recipe is currently implemented by generated CUDA and the existing
pedantic matrix provider. Other optional matrix/affine providers explicitly
reject this publication contract until they have a qualified complete recipe;
they must not execute an unweighted contraction and claim the weighted identity.
No method-local provider selector is introduced. Qualification bits remain test
hooks, and production keeps the fused generated incumbent.

ESP integral generation and final symmetrization remain with their current
scientific owners. Derivative consumers and enclosing Fock resource admission
still require independent slices. This change does not qualify production
PBE0/COSX performance or close #1884/#1886.

## Evidence and revisit conditions

The compiler tests compare asymmetric ESP and signed/zero weights against
independent long-double loops, and verify CPU/CUDA canonical identity and liveness
rejection. Native gates additionally compare generated output to an independent
host FMA chain, cover full/tail shapes, alias/layout/finite/sticky-error behavior,
and compare every independently admitted value route to the discrete CPU COSX
oracle under both density conventions. Shared provider regressions ensure the
optional scale table does not change existing unweighted execution.

Promote a library recipe only after retained complete cold/warm/changed-geometry
endpoints include its publication cost and preserve the actual scientific work.
The previous projection/update endpoint receipt supplies no such ESP promotion.

## References

#1884, #1886, #1966 and #1967. Source/binary manifests, compiler/cache records,
finite Slurm assignments and qualification logs are retained under the ignored
`.artifacts/1884-cosx-esp/` directory.

## Composition with the projection/update endpoint receipts

The endpoint harness introduced by #1967 also consumes the diagnostic site table.
Its four-site schema described projection/update only, so extending the native
arithmetic gates alone would silently omit weighted ESP work from new receipts.
Current endpoint output therefore uses a separate six-site schema, including the
ESP full/tail batch dimensions, scalar summands, weighted elements and split
publication passes, and independently exercises all three qualification bits.
The native gate preserves exact generated-catalog identities and rejections for
all six sites, including optional offers that are not admitted.

Historical four-site data, summaries, provenance and source attribution remain
unchanged. The verifier distinguishes historical and current schemas instead of
relabeling old measurements as weighted-ESP qualification. Host regressions use
the actual generated descriptors and endpoint harness serialization to cover
both contracts. These compatibility checks provide no new device timing,
sanitizer result or production endpoint promotion.

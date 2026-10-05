# Decision: preserve coupled COSX ESP publication in shared lowering

Status: implemented
Date: 2026-10-05

## Problem

The molecular COSX derivative applies each point's ESP matrix to its density
projection and the transpose to its symmetric projection. The original scalar
program advances both reductions together; either update failure zeroes both
outputs. Splitting this into independent matvecs changes failure semantics even
when finite results agree. Assuming ESP symmetry would also erase an observable
read direction rather than represent the original equation.

## Decision

Keep the two existing TensorIR einsums and original scalar program as scientific
authority. A bounded common recognizer proves both square batched contractions,
one borrowed matrix, unit coefficients, strict FP64 and matching scalar roles.
The prepared descriptors carry forward/transpose roles and the same complete
scientific/scalar identities. Common execution checks both bindings before one
enqueue and preserves the original simultaneous increasing-k traversal.

Neither member can execute independently. Optional providers that cannot honor
the complete checked region remain explicit negative candidates. Two existing
output allocations receive the joint publication; there is no packing, copied
transpose or new device storage. Molecular preparation has eight full/tail
sites and a 192 KiB host reservation, consumed by enclosing Fock admission.

## Rejected alternatives

- Independent library matvecs with only final-output audits.
- Assuming a symmetric ESP matrix to omit the transposed view.
- Retaining the dense loop in a method-owned wrapper around a shared API.
- Materializing a transpose or shared intermediate buffer.

## Invariants and evidence

Asymmetric matrices independently exercise both read directions. Long-double
oracles check mathematics, and a retained incumbent CUDA traversal checks every
finite output bit. Full/tail, forced qualification, per-contraction work,
partial-execution rejection, wrong helper, overlapping output and one-sided
failure gates protect the complete region. A failure in either update must
zero both outputs while preserving an enclosing sticky error.

The common variant lookup also preserves context-retained module quarantine
from the optional CUTLASS owner. Both ordinary and joint execution reject a
quarantined table before shape/helper/pointer checks, including after release.
The optional native CUTLASS failure regression exercises both joint-entry gates;
its placeholder scalar helper must never enqueue.

Source/binary manifests, ccache commands/statistics and finite Slurm qualification
receipts live in ignored `.artifacts/1884-cosx-bidirectional/`. Complete Fock,
production-default HF/PBE0 SCF and independent molecular derivative regressions
remain acceptance gates. This is an ownership migration, not a speedup claim.

## Consequences and revisit

Composed ESP response, molecular pullbacks and point reductions still require
separate migrations. A faster paired provider must prove the joint scalar/error
contract before replacing the retained ordered generated implementation.

## References

#1884, #1886, #1973 and
[symmetric projection](2026-10-05-cosx-symmetric-checked-projection.md).

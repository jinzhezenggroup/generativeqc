# Decision: bind molecular-SCF promotion to observed J/K value work

Status: implemented
Date: 2026-10-12

## Problem

Automatic pure semilocal KS resolution requests Coulomb without exchange, but a
requested plan does not prove which providers actually executed. Version-1
molecular-SCF receipts required numerical and lifecycle evidence without retaining
the newly available native value-provider census. That evidence could therefore
remain numerically passing while unnecessary exchange or an unobserved provider
invalidated the method-work claim.

## Decision

Version-2 results bind the additive native J/K census into each lifecycle row and
its existing content hash. Every passing CPU pure semilocal row must have complete
observed counts, a positive Coulomb count, and no exchange applications. Counts are
uint64 values, never iteration-derived estimates. Unknown work remains unavailable,
not zero. Failed rows retain observed incorrect or unavailable work without
promoting it. The native qualification runner transports its actual diagnostic,
including final physical closure work, rather than reconstructing counts in Python.

Public-admission producer validation requires the current result/qualification
schemas. Historical version-1 receipts are not upgraded by inventing counts; they
need a new execution. This does not extend endpoint coverage to CUDA, forces,
response, or particle-changing empty-spin continuation. Complete precision,
integral preparation, derivative, and other operator work remain separate claims.

Qualification compiles its runner through the existing verified cache/process
owner and links separately. Cache selection is execution provenance, not science
identity; cache hits or endpoint speedups are not assumed.

## Rejected alternatives

- Plan inspection or iteration counts: neither observes actual dispatch or extra
  final closure evaluations.
- Missing counts treated as zero: unsupported providers would appear J-only.
- Keeping version-1 promotion: old receipts cannot prove the new work gate.
- Weakening independent energy/residual gates: work correctness supplements, not
  replaces, those gates.

## Evidence and invariants

Behavioral tests cover actual-count transport, unknown counts, extra K, overflow,
boolean/float/string counts, rehashed invalid work, retained failures, and legacy
receipt/stage rejection. The native-linked independent PySCF/Libxc qualification
and exact source/library provenance belong to the owned ignored qualification
directory, not fabricated fixture counts. No scientific formula, occupation,
threshold, convergence budget, or provider-selection policy changes.

## References

#1118, #1121, #1124; `docs/developer/libxc_bulk_capabilities.md`;
`src/scf/coulomb_exchange_work.hpp`.

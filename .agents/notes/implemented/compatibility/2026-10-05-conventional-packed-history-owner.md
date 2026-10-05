# Conventional preparation with packed DIIS histories

Status: implemented
Date: 2026-10-05

## Decision

Conventional prepared contractions and restricted packed DIIS share the existing
CUDA solver owner. Preparation does not change the two-allocation numeric
contract: the main arena and history payload must both fit before the traversal
can execute. For conventional preparation, a resource refusal of the second
allocation releases the partial arena and optional provider, then tries the
complete scalar pair once. Provider
release and partial-owner cleanup remain serialized with allocation measurements.
Arithmetic, driver, drain, and free errors still propagate.

Admission includes both numeric allocations, the active provider allowance, and
its host descriptors. High-water diagnostics retain an earlier main arena and
provider even when a later history refusal selects the scalar traversal. Packed
history rejection after execution remains its separate full-history-or-Jacobi
policy, with unchanged physical gates and independent replay. A successful
packed-to-full replacement includes the still-live conventional provider in its
new coexistence peak, rather than retaining only the earlier packed peak.

The native solver record retains the conventional 34-column prefix and appends
the eight packed-history diagnostics. Its Python consumer uses the same order.
Neither feature's counters are replaced or interpreted as evidence for a new
complete-endpoint speedup.

## Validation

The owner fixture extracts the production constructor, cleanup and replacement
code. It crosses conventional/DF ownership with full/packed histories and injects
history-allocation failures. The conventional cross-case checks bounded scalar
retry, complete cleanup and preservation of the earlier provider-inclusive peak.
The separate lifetime fixture checks the combined refusal while another owner
holds the allocation-measurement lock. Native probe tests consume the complete
record and retain the scientific oracle checks.

Real-device and complete-endpoint qualification remain separate from these
host-side integration checks. The scalar and prepared traversals continue to use
the original scientific programs, with pair coordinates applied only to history
storage.

## References

- PR #1868: shared conventional prepared contractions
- PR #1927: compiler-owned restricted packed DIIS histories

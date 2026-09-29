# Decision: Separate precision-work query from aggregate provenance

Status: implemented
Date: 2026-09-27

## Problem

DFT-MP-v1 acceptance needs the ordered Fock/audit timeline and an executed
operator dtype/arithmetic census. The existing fixed
`vibeqc_precision_provenance` descriptor contains only aggregate counters, so an
adapter cannot reconstruct the missing order or operator identities honestly.
At the time this contract was introduced, CUDA-KS, generated-Fock, Python result
facade, and `scf::PrecisionProvenance` files were owned by active independent
work. A representation change could not take over those execution owners.

## Decision

Expose a separate version-1, two-call C query for single calculations and
original-index batch items. Its fixed summary and row descriptors carry ordered
events, returned-state identity, operator identity, four dtype roles, arithmetic
mode, and counts. The legacy aggregate descriptor and its layout are unchanged.
The returned-state identity is a process-local execution token and remains
separate from the portable scientific state label used for independent-oracle
root matching.

The internal `scf::PrecisionWork` sidecar travels through the method-neutral
result and API handles without being embedded in `scf::PrecisionProvenance`.
Its default is versioned but incomplete with an empty inventory. Until an
execution owner supplies every event and operator for the returned attempt,
public queries therefore expose `complete = false` and cannot satisfy the
DFT-MP-v1 validator.

## Rejected alternatives

- Expanding the fixed aggregate ABI would silently change existing caller
  expectations and still would not define a variable-length query protocol.
- Reconstructing events from aggregate counts would invent ordering, phase,
  state identity, and operator arithmetic. The validator continues to reject
  that approach.
- Editing active CUDA-KS, generated-Fock, `Calculator`/`PreparedBatch`, or
  `scf::PrecisionProvenance` owners would create a parallel implementation and
  unsafe merge pressure.

## Invariants

- Aggregate precision ABI size, offsets, counters, and availability behavior
  remain compatible.
- A missing producer, partial attempt, unknown enum, or old library stays
  uncertified; adapters do not repair it.
- Failed/rejected calls clear the new sidecar before validation so a query
  cannot expose a previous run or neighboring batch item.
- `complete` and `operator_inventory_complete` are execution-owner assertions,
  not conclusions inferred by the API layer.
- Detailed counts must agree with the aggregate counters before DFT-MP-v1 can
  accept a row.

## Evidence

The native precision-policy target checks descriptor validation, atomic
copy-out, exact aggregate layout, and the default incomplete record. Python
tests cover optional old-library binding, unknown enum preservation, original
batch index routing, failed replay invalidation, and DFT-MP-v1 negative controls.

On exact upstream base `fe68c41acb30d65d4fa5a8d408d3d79f2719b1cc`, a CPU-only
Release build compiled both C API translation units and the precision-policy
target; its focused CTest passed 1/1. The source-matched Python selection passed
47 tests with four environment-optional skips, and exported-symbol inspection
found both single and batch precision-work query functions. Local host-only
decoder/validator tests passed 49/49, and all changed-file pre-commit hooks
passed.

## Consequences

This slice establishes a compilable/queryable representation but deliberately
does not claim complete execution instrumentation or DFT-MP-v1 acceptance. The
active CUDA execution owners must populate the sidecar, and the active Python
facade owner must merge the decoded detail into `Result.precision`, before a
production row can become complete.

## Revisit when

Revise the detail version only when a required operator/event cannot be
represented compatibly. Remove the incomplete fallback only after every public
execution domain that advertises completeness has an execution-owned producer;
never replace it with aggregate reconstruction.

## References

- [Count mixed Coulomb work at the executing provider](2026-09-29-executed-mixed-coulomb-census.md)
- #1303
- #1189
- #1190
- #1191

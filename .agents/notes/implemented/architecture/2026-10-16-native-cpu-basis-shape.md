# Decision: native compact basis shape for CPU resource requests

Status: implemented
Date: 2026-10-16

## Problem

CPU HF/KS resource phases already query native allocation inventories, but their
public/Cartesian AO and primitive dimensions were still computed by Python.
General contractions must count expanded rows and retained zero coefficients,
not just element shell templates. A separate planner formula can drift from the
normalized scientific system's representation rule.

## Decision

Use an allocation-free private native metadata query on compact expanded shell
rows. Public AO counts share `molecule::shell_ao_count` with normalized systems;
Cartesian counts share `molecule::cartesian_count`. Bound every aggregate by
both host `size_t` and signed-int64 resource metadata limits. Publish all five
counts only after validating every row, without coefficients, coordinates,
normalization, provider selection or scientific array allocation.

Python marshals the rows and retains mathematical provenance, basis expansion,
capability checks and ECP metadata. CPU requests discover the additive symbol
on their already-selected library. Missing symbols and library-free requests
retain portable arithmetic. A discovered refusal never retries Python sizing.
CUDA resource requests deliberately retain their previous shape path; this
increment is not CUDA endpoint qualification.

## Rejected alternatives

- Constructing a normalized native system just to count dimensions would turn
  a metadata-only estimate into potentially expensive preparation/allocation.
- Depending on the unmerged catalog/template consolidation would unnecessarily
  couple this independent count migration to expansion ownership.
- Treating compact counts as complete allocation or lease authority would hide
  unresolved provider-retained and simultaneously-live resource scope.

## Invariants and evidence

`tests/python/test_native_basis_shape.py` exercises independent PySCF dimensions,
Cartesian/spherical and general-contraction rows, distinct DF auxiliary shapes,
actual HF/KS planner routing, no implicit library loading, native refusal,
transactional late failure, exact signed limits and neighbor recovery.
Existing scientific system tests retain normalization and backend admission.
Full endpoint measurements belong to source-matched PR evidence, not a claim
that a scalar metadata query improves scientific throughput.

## Revisit when

Native catalog/template expansion can also supply the planner's complete
mathematical provenance, or CUDA shape cutover has its own complete endpoint
acceptance. Full provider/lease/receipt authority remains separate R2 work.

## References

- #2227 R2/R6; #933; #934.
- `docs/maintainer/resource_planning.md`.

The separately qualified CUDA consumer follows in
[native CUDA basis counts](2026-10-29-native-cuda-basis-shape.md); this note's
original CPU qualification scope remains unchanged.

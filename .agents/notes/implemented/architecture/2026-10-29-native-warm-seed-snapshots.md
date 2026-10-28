# Decision: retain precision audit seed arrays in native snapshots

Status: implemented
Date: 2026-10-29

## Problem

The prepared capacity query bounds numeric seed export before it can materialize
resident CUDA densities. Precision audit suspension nevertheless queried each
seed twice, allocated Python arrays, and reconstructed per-item descriptors for
restore. This left array lifetime and copying in a production Python loop even
though the existing prepared owner already provides export and transactional
physical restoration.

## Decision

Add an immutable native host snapshot around the existing method-owned
`HfWarmState` representation, which already serves HF and KS. Capture checks the
complete all-slot bound before exporting any seed, copies source states once,
and publishes the handle and actual numeric receipt only after all copies succeed.
The snapshot owns no context, device storage or scientific solver. It can survive
source batch/context destruction and can be restored repeatedly into a matching
new prepared owner. Existing physical validation remains authoritative.
Restore binds the target's retained resource ledger before validation, so the
installed SDK cannot bypass eigensolver/provider scratch admission. The Python
descriptor and snapshot paths share resource diagnostic/error presentation;
native-managed observation reads the retained owner's receipt rather than
installing a competing scope.

Python retains model/topology identity checks, resource reservations, metadata,
update selection and independent candidate/auditor history slots. The snapshot
is explicitly not a model-admission or cross-model projection token. A public
SDK caller must preserve the same ordered model, basis, charge and spin contract
as for the existing descriptor restore API. Shape/physical validation does not
prove model identity. This boundary is deliberately narrower than moving the
whole precision controller to C++.

Native array views are read-only and valid only while the snapshot lives. The
Python diagnostic property copies them, so saved diagnostic arrays do not borrow
native storage or keep a source context alive. Production capture/restore does
not create those diagnostic arrays. Typed binding/library ownership is shared
with the existing bounded metadata transport. Only a missing capture symbol
uses the old descriptor loop; incomplete protocols or native failures propagate.

## Invariants

- No candidate-to-auditor or auditor-to-candidate seed transfer.
- Preserve source coordinates, density spin convention and frozen update policy.
- Every target still executes its own SCF after restoration; no cached result.
- A deficient all-slot limit fails before any source materialization.
- Capture failures preserve output handles/receipts and release partial copies.
- Native restore keeps its existing all-or-nothing physical validation.
- Closing a snapshot requires no source context or device; borrowed reads and
  destruction are externally serialized.
- Numeric bytes exclude descriptors, allocator overhead and validation scratch;
  Python retains existing scratch/resource reservations. This is not RSS or
  opaque-provider lifetime accounting.

## Rejected alternatives

Keeping NumPy arrays as the owner retains the production reconstruction loop.
Moving resident GPU objects into the snapshot would prevent the audit controller
from releasing candidate device residency before admitting its independent
auditor. Reimplementing density validation or treating matching dimensions as
model identity would create a second scientific authority.

## Validation and revisit conditions

Compiled fault-injected owners exercise pre-export refusal, partial-copy failure,
output transactionality, source teardown, concurrent immutable views, repeated
restore and failed restoration. Actual HF/KS replay and the existing independent
precision-audit suites exercise frozen/moved histories and candidate/auditor
isolation. Complete CPU/CUDA audited endpoints must retain scientific/work gates
and the existing 1.05 performance threshold before review-ready promotion.

Revisit the boundary when the native precision controller owns immutable model
identity, history selection and host reservations together. Resident seed storage
requires an explicit device-capacity contract and a bounded host fallback.

## References

- #2227 R2/R4/R6; R3 remains excluded.
- [Capacity decision](2026-10-29-native-warm-seed-capacity.md).

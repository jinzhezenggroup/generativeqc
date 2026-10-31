# Decision: separate Libxc scalar provenance from integral tuning

Status: implemented
Date: 2026-10-31

## Problem

Libxc capability subjects hashed every integral compiler file. Editing the
production tuning publication validator consequently changed all 221 Libxc
subjects although shell recurrences and schedules are not XC scalar inputs.

## Decision

Capability source schema v2 retains pinned upstream/catalog semantics and the
conservative common/XC source inventories. Its integral inputs are the package
initializer, Graph algebra (`expr.py`), scalar C emission (`scalar_c.py`), and
CUDA scalar emission (`cuda.py`). Current XC scalar imports use these owners;
Graph's shared implementation dependencies remain in the common inventory.
This is a scientific dependency inventory, not the eager integral package's
complete transitive import closure. New mathematical dependencies must extend
this inventory. Native endpoint whole-build identity checks remain unchanged.

The migration acknowledges exactly 221 identity changes against the preceding
master snapshot summary. Qualified stages are unchanged; all native campaign
bytes and measured identities remain untouched. Already-stale native receipts
remain stale, with zero current-source native endpoint admissions. Old identity
receipts are not aliased to the new subjects and no execution is fabricated.

After decoupling, tuning publication validates every supplied measured IR,
including legacy rows without independent Fock recurrences. It compares the
selected consumer through the same specialization rules as production, rejects
a mismatch before row mutation, and preserves implicit schedule-only legacy
trials. Joint force/Fock requests are compared by their executed consumer.

## Rejected alternatives

- Refreshing 221 identities for each unrelated tuning edit leaves the coupling.
- Dropping source/upstream integrity checks or accepting old subject identities
  would silently reuse qualification for changed scientific inputs.
- Relabelling historical native campaigns would invent current-source evidence.
- Hashing only imported module names would miss actual Graph/emitter changes.

## Evidence

Behavioral tests cover scalar-input invalidation, tuning-independent capability
identity, recurrence mismatch rejection before mutation, retained consumer
semantics after manifest reload, and legacy implicit publication. Existing
independent Libxc fixtures, historical receipt corruption tests, and exact
transition checks remain active. No native/GPU numerical rerun is claimed.

## Revisit when

XC emission starts using another integral mathematical owner, or native endpoint
qualification is redesigned with independently attested component identities.

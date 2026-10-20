# Decision: admit MD-J against its owner's capacity

Status: implemented
Date: 2026-10-19

## Problem

The 96-atom r2SCAN-3c basis needs 124,792,120 bytes for incumbent MD and
27,213,984 additional bytes for the reference hierarchy. The direct plan could
afford that 152,006,104-byte allocation, but `MdJHost::prepare` clamped its
caller-supplied remaining capacity to 128 MiB. The reference selector therefore
fell back despite an adequate owner budget.

The 128-MiB value originated as an optional reservation in private KS's
shape-only provider estimate. It is not a mathematical, CUDA, ABI or indexing
limit. The public-ledger guard, retained-owner charges, actual allocator budget
and optional rollback protect other allocations. Treating the reservation as a
second hard cap prevented use of capacity already assigned to this plan.

## Decision and invariants

Remove the additional clamp. Descriptors, transforms, Hermite buffers, hierarchy
and work counters must fit the remaining capacity supplied by the direct owner.
Rename the unchanged shape-estimate allowance to `kMdJDefaultReservation`.

The private provider query still adds its existing 128-MiB reservation; dropping
that reservation would reduce ordinary KS capacity and could lose established
MD routes. Public shape estimates and their candidate identities do not change.
Public ledger-bound and resident-source-only plans remain ineligible for MD.
Do not substitute live GPU free memory for an owner budget.

If base MD fits but the hierarchy does not, retain incumbent MD. If base MD
does not fit, retain the normal provider. Allocation/preparation failure still
fences and rolls back the optional group; unrelated CUDA/numerical errors
propagate. No scientific screening, integral math, density transform, precision
policy or runtime kernel changes in this capacity repair.

## Validation boundary

Real def2-mTZVPP 48/96-atom spherical/Cartesian inventories exercise full
reference admission, exact capacity, one-byte-short hierarchy fallback and
one-byte-short base refusal. The 1088-AO case must admit the hierarchy with a
budget larger than the former limit. Existing fault-injection, ledger/fleet,
force/rebuild reservation and opt-out tests remain in place.

A real CUDA 96-atom/1088-AO test constructs a standalone 1-GiB owner, proves
reference admission and actual work, compares nonsymmetric tail-density J with
the incumbent exact-mask route and verifies zero-density replay. This explicit
standalone budget is not a production default or public-ledger bypass.
Complete r2SCAN-3c energy/force measurements and independent references qualify
the newly admitted production route. Counters/profilers stay outside clean
timing populations.

## Rejected alternatives

- Replace 128 MiB by another fixed constant: reproduces the hidden ceiling at
  a different system size and ignores capacity already supplied by the owner.
- Compress hierarchy solely to satisfy the cap: may improve memory efficiency,
  but is unnecessary for correct admission and adds a separate layout/screening
  validation burden.
- Admit MD under a public ledger using currently free bytes: steals reservations
  from later owners; a real optional planner candidate is required.
- Delete the private reservation with the clamp: confuses capacity requested by
  a provider estimate with the capacity actually passed to an owner.

## Evidence and follow-up

The local campaign is retained in `.artifacts/pr2609/budget/`, including exact
source/binary receipts, clean endpoints, work counts, large-basis sanitizers,
pinned upstream MD compilation and GPU4PySCF comparisons. The previous
[quartet-lane decision](2026-10-19-md-j-quartet-lane-scheduling.md) records the
pre-repair hierarchy-cap refusal; this decision supersedes that capacity policy.
The [original resource correction](2026-10-08-md-j-default-cold.md#resource-admission-correction)
still governs public-ledger ownership and private reservation accounting.

Admission is not a universal performance promise. Broader bases and methods
must retain endpoint qualification; the reference engine remains opt-in.


## Measured admission, separate from later algorithm changes

The source-matched n4 capacity campaign admitted144.96MiB and measured
reference r2cold236.820267s versus incumbent244.214933s; GPU4PySCF cold was
153.635193s. Admission alone therefore did not repair the performance gap.
All54 campaign E/F endpoints passed. Later primitive/scheduling/spatial repairs
are documented separately in the [primitive decision](2026-10-19-primitive-complete-md-j.md)
and [spatial decision](2026-10-19-xc-spatial-grid-order.md); do not attribute their
gains to removal of the cap or rewrite this campaign's source provenance.

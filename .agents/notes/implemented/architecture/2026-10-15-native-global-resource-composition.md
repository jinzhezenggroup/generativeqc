# Decision: share global capacity composition in the native runtime

Status: implemented
Date: 2026-10-15

## Problem

Per-provider native sizing did not make the native runtime authoritative over
the complete declared plan. Production Python still selected coupled provider
alternatives and composed their overlapping lifetimes. The installed CLI had
another cap/headroom and candidate loop. This left a central #2227 R2 ownership
gap even when every individual capacity query agreed.

## Decision

The existing native runtime gains one generic resource composition owner under
`src/runtime/resource_planning.*`. Both the CLI and the private additive
`generativeqc_resource_compose_v1` bridge call it. The bridge accepts immutable,
disjoint, flat metadata records and publishes a complete result only after
descriptor validation and finite enumeration succeed. It constructs no
scientific owner or backend state.

The event sweep uses inclusive lifetimes and retires old buffers before adding
new ones at a shared endpoint. Sparse phase identifiers, including int64's
maximum, do not create a dense timeline. Only touched atomic spaces and their
aggregate host/device counters change. Persistent/output/cache/library/runtime
buffers also contribute to the independently swept resident inventory.

The native owner minimizes `(sum of cap excesses, sum of relative costs,
lexical choices)`. The two objective sums are wider than int64/uint64, while
each byte capacity and concurrent peak remains signed-int64 bounded. Exact
IEEE-754 headroom arithmetic moves from the CLI into the shared engine;
reserves and present zero caps preserve their existing meanings. All supported
combinations are checked before publication: an early feasible choice does not
hide a later simultaneous-live overflow. No product-sized result list is kept.

The runtime adapter, not compiler code, marshals ResourceRequest metadata.
Calculators and budgeted prepared batches pass their already-loaded library.
Explicit supplied selections get a fixed-candidate admission query rather than
being reoptimized. New-library rejection never invokes portable selection.
The optional CPU energy-to-force workspace upgrade uses the same fixed-selection
admission; its revised descriptor is only published after successful scientific
execution. A rejected upgrade does not alter live owners or the energy plan.
Old/no-library dry runs retain the portable planner and load no provider.

## Rejected alternatives

- Loading the native library from `generativeqc_compiler` would break standalone
  source generation and invert the compiler/runtime dependency boundary.
- Rewriting the public portable data model would unnecessarily change plan
  serialization and compromise independent derived-byte validation. It remains
  a pointer-free descriptor, not a native lease or allocator handle.
- Floating-point total costs or wrapped uint64 objectives can reorder legal
  candidates. Two integer limbs preserve exact ordering without compiler-specific
  integer extensions.
- Stopping at the first feasible alternative would change overflow rejection
  and cost/name tie semantics. Search stays finite and evaluates all admitted
  combinations before publishing one result.

## Compatibility and boundaries

Feasible CLI preference ordering is unchanged. For an entirely infeasible CLI
DF candidate set, the diagnostic selection now uses the same least-excess
ordering as the composable planner rather than arbitrarily retaining its first
candidate. No scientific execution is admitted in either case.

The engine owns declared-capacity selection/admission, not scientific method
eligibility, actual allocation reservations, ResourceSession retries, prepared
lease transitions or complete host/device receipts. Python still constructs
some provider inventories and alternatives, and ResourcePlan retains portable
derived-byte validation. These remaining R2/R6 boundaries must not be erased
from the tracker. R1/R3/R4/R5 also retain their independent acceptance gates.

## Evidence and work bounds

Native contracts cover exact/subnormal headroom, zero/unlimited limits, sparse
and adjacent phases, simultaneous host/pinned/device constraints, objective sums
beyond uint64, deterministic ties, finite-search refusal, later overflow,
malformed/aliased schemas and transactional publication. Python compares the
native result and full diagnostics with the independent portable model across
deterministic ragged cases, then exercises production entry points, supplied
selection preservation and complete CPU exact/below-budget replay.

The sweep costs `O(B log B + S)` for selected buffers/spaces. Enumeration costs
that amount per admitted combination, with additional linear descriptor checks;
metadata scratch is linear in the declared inputs, not the product or phase IDs.
This is a work bound, not an endpoint performance or physical RSS claim. CPU
scientific/public resource and Python-free CLI tests remain separate gates.

## Revisit when

The native owner can bind declared plans to complete allocation journals and
prepared leases, or production session fallback policy is moved from the
portable executor. Keep independent portable validation/oracles available even
after native production authority expands.

## References

- #2227 R2/R6; #933 / #934
- `docs/maintainer/resource_planning.md`
- `src/runtime/bounded_workspace.hpp`
- `python/generativeqc_compiler/common/resources.py`

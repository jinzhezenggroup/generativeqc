# Decision: protect complete D4 fleets before admitting optional XC storage

Status: implemented; complete endpoint qualification in progress
Date: 2026-10-29

## Problem

The selected-inventory path in #2872 disables optional AO maps and point batches.
Its CUDA budget campaign preserved E/F but launched 221,476 kernels instead of
172,628 and took up to 3.41922 times the baseline. The earlier library hashing
and scalar planning fixes do not repair this native schedule change. #2875
correctly excludes named corrections from an electronic-only capacity promise.

## Decision

Extend the existing numeric-cap promise with every retained D4 owner using the
same native layout queried by preparation. Replicated tables, execution scratch
and later force workspaces coexist with the entire electronic fleet. Rebuilding
an item destroys its old correction owner first; unused promise credit remains
reserved for its replacement. D3/gCP retain the previous bounded path.

A separate optional candidate uses allocator-owned AO-map bounds, the spatial
copy size and the existing native point-panel policy ceiling. It also covers
the diagnostic history-square reserve. Share that point-panel policy with
execution rather than copying a Python sizing formula. This is conservative
headroom, not claimed actual live storage or a universal resource planner.

Only the validated complete KS+D4 candidate converts its own selected capacities
into a numeric cap. Provider allowances enter this cap because the native
promise converts them to actual leases. Other global requests cannot lend their
bytes. Generic selected inventories retain their original authority, and both
original cache/minimum candidates remain available at tighter budgets. Missing
optional protocol support retains those candidates. Native refusal is final.

## Invariants and rejected shortcuts

Do not remove the selected-inventory XC guard, treat current free memory as a
future-owner bound, borrow the entire user's global budget, or infer a complete
native contract from the number of inventory rows. Dense mandatory credit stays
reserved even while optional maps are live. Failed numerical/performance samples
remain failures; neither lower peak memory nor unchanged SCF iteration counts
qualifies extra GPU work.

## Evidence

The existing public native capacity probe covers scalar/fleet correction replay,
shared resource ownership, moved geometry, below-capacity refusal and teardown.
Python tests additionally require an actual complete promise on the preferred
D4 path and selected-inventory admission at the mandatory resource floor.
Full numerical and performance receipts belong in
`benchmarks/results/d4-capacity-20261029/`; earlier failures retain their source
and binary identities. Performance promotion requires both original eight-group
endpoint campaigns and actual work/storage profiles.

## References

- #2227 R2/R6; #2872; #2875.
- [Original promise rationale](2026-10-26-native-xc-capacity-promises.md).
- [Failed selected-inventory diagnosis](../../rejected/2026-10-26-prepared-admission-xc-reservations.md).

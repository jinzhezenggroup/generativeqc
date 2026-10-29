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

The first complete run at `9d0df5629` (Slurm 14903) passes the default campaign
at 1.01182439 but fails the budget campaign at 3.47645361. The s/p-only promise
does not cover the original matrix's d-shell water/def2-SVP member, so whole
fleets retain selected-inventory fallback. Preserve every sample and both
profiles in `sp-only-cuda-qualification.json.xz`, recoverable through the
verified `sp-only-history.json` Git snapshot manifest. The srun exit is 1; the completed
scheduler record expired before retrieval and accounting storage is disabled.
The running record and failure output remain evidence, without an invented
completed scheduler receipt.

Individual Direct D4 promises now admit the existing qualified s/p/d force
context using the same allocator-owned bounds. Uncorrected and DF promises keep
their original angular domain. The larger executable regression uses four
def2-SVP waters, so a basis-name assumption cannot silently miss this domain
again. Its initial 2 GiB host cap correctly refused the declared 3.244 GB host
setup bound; this larger test uses 4 GiB host and 2 GiB device capacity. The
original timing matrix keeps both of its 2 GiB caps and its unchanged 1.05 gate.
This extension still needs the full qualification matrix.

The d-shell diagnostic (Slurm 14907) still launches 172,970 kernels against
172,628: public admission suppresses the MD-J source even after XC admission is
repaired. Complete individual Direct D4 promises now carry narrow authority to
try that optional source. Their mandatory sizing explicitly excludes MD-J,
independent of any surrounding promise. During optional construction, a scope
withholds all mandatory credit; MD-J and subsequent optional component reuse
share at most 128 MiB of unpromised capacity. Every sizing check uses the smaller
of the local owner budget, that ceiling and current unpromised capacity.
The native additive query reserves this ceiling and conservative host staging
and signature metadata. Ordered Schwarz borrows existing device scratch and
uses this same admitted host envelope. Missing capacity retains generated J;
uncorrected, DF and other public resource owners do not inherit this authority.
This repair requires fresh numerical, lifetime and complete-endpoint evidence.

## References

- #2227 R2/R6; #2872; #2875.
- [Original promise rationale](2026-10-26-native-xc-capacity-promises.md).
- [Failed selected-inventory diagnosis](../../rejected/2026-10-26-prepared-admission-xc-reservations.md).

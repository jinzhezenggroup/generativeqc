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
The larger regression now uses 5 GiB host capacity: its 3.244 GB mandatory host
setup plus two conservative optional MD staging envelopes exceeds 4 GiB.
The original small-fixture performance matrix retains its 2 GiB host/device
caps; this larger regression still asserts actual optional admission and bounded
combined peak, while the separate exact/below-floor test covers refusal.

## References

- #2227 R2/R6; #2872; #2875.
- [Original promise rationale](2026-10-26-native-xc-capacity-promises.md).
- [Failed selected-inventory diagnosis](../../rejected/2026-10-26-prepared-admission-xc-reservations.md).

## Retained request validation

Complete MD-J admission at `b2e5de7d1` restores exact GPU work/storage, but its
budget timing still fails at 1.2613104. The retained campaign and host profile in
`benchmarks/results/d4-capacity-20261029/md-complete-cuda-qualification.json.xz`
show that replay reconstructs 56 complete requests for already prepared owners.

A complete native CUDA D4 batch now retains its validated immutable electronic
request. Execution still verifies the scientific model and initial-guess policy,
then compares the remaining planning inputs by value: immutable topology,
charge/spin, effective KS profile, backend/device, library, supported properties
and the entire environment. Any difference falls back to full reconstruction
and equality checking against the prepared plan. Unrelated environment changes
can therefore pass validation; changed resource schedules cannot borrow stale
capacity. Comparing all environment entries also covers controls consumed by
native inventory queries without maintaining a second Python control registry.
External plans retain their full native validation before preparation. CPU
output-aware force rebudgeting and all other owners retain their existing path.
The cache holds one request per prepared owner and does not replace native
capacity enforcement or correction ownership.

This deliberately rejects a process-wide identity-only request cache: it could
miss model mutation, batch-local profile selection or environment scheduling.
The optimization's complete-endpoint timing acceptance still requires the same
8-group gate, including scalar preparation and moved-geometry replay.

The first retained-request diagnostic at production `49705a84e` (test fixes at
`82c37dbf4`, Slurm 14913) still fails the 1.05 gate: 448 complete calls in two
groups yield 1.1008970. Its full population is retained as a diagnostic, not a
replacement for eight-group qualification. A subsequent host profile shows
72 replay-input snapshots spend 0.122 seconds under the profiler, and 72 plan
serializations spend 0.075 seconds. When effective options are the calculator's
own options, the already-required model signature proves their value; only a
separately selected profile needs an additional payload snapshot. Complete D4
ledgers also retain one immutable plan serialization, returning a deep copy for
each diagnostic. Plan replacement invalidates that cache by value. Native
observations and capacity verification remain fresh, and public diagnostic
mutation cannot change the private snapshot or any later observation.

Detached diagnostic reuse at `8f1c2b3f6` reduces the next two-group maximum to
1.0733022 (Slurm 14915), but preparation/scalar cases still miss the gate.
Preparation was rediscovering the same full native graph for optional headroom,
correction sizing and the actual prepared owner. CUDA preparation now carries
one local selection result through those consumers. This result never crosses
a constructor or loaded-library lifetime; public dry runs and CPU callers retain
their original discovery boundary. Native catalog refusal and preparation
failures remain authoritative, without a second discovery/compatibility retry.

The catalog-carry two-group diagnostic (`5e42386b2`, Slurm 14916) still fails
at 1.0792407. Warmed profiling shows comparable actual native preparation
(0.6178/0.6134 seconds baseline/candidate), but 72 diagnostic-template copies
consume 0.1614 seconds under the profiler. General-purpose `deepcopy` is not a
suitable implementation for this canonical data-only tree. Copy dict/list/tuple
containers directly, sharing only immutable JSON scalars; retain tuple types and
the existing mutation, changed-plan and authoritative-capacity tests. JSON
round-tripping was rejected because it changes tuple containers to lists.

Full eight-group qualification at `1e214273b` (Slurm 14918) now passes every batch
phase, preserving exact GPU work/storage and 150 GPU test gates. Three scalar
cases still exceed 1.05, with maximum 1.0586252. CUDA metadata was resolving an
already resolved immutable KS option again during each scalar preparation.
`resolve_ks_options` now records its exact selector as non-scientific provenance
on the frozen result. CUDA planning may reuse that same validated value; the
field is excluded from comparison and `to_payload()`, cannot be set in construction,
and resets on `dataclasses.replace`. Changed options, different selectors and
profile-derived replacements retain full resolution. Shape, precision, native
provider/capacity and execute-time model/provenance checks are unchanged. CPU
planning keeps its previous resolution path.

Selector-bound option reuse at `a677a78c0` passes the eight-group scalar-only
diagnostic (1.0475318), but full qualification still fails at 1.0544498 in Slurm
14920. All 180 GPU tests and both exact work/storage profiles pass. Warmed host
profiling also identifies 24 redundant model hashes while preparing per-item
basis metadata. The complete CUDA D4 constructor already takes a full scientific
model snapshot immediately before that metadata loop. Carry this signature
through that one synchronous operation, while preserving public metadata
validation order and all later execute-time mutation checks. The independent
public metadata query must produce the same complete identity for each item.
Do not retain this signature across caller-visible mutation points or extend
the optimization to unrelated prepared owners without qualification.

Full qualification of this constructor-local snapshot (`af71794db`, Slurm
14922) passes both eight-group populations: default 1.0172885, budget 1.0333197.
Each contains 1,792 complete calls; both profiles exactly preserve baseline GPU
work/storage, and all 180 GPU tests and independent native/SDK references pass.
The preceding scalar diagnostic is retained separately within the receipt. New
upstream provider-lifetime integration is a different source/binary qualification;
the successful pre-integration receipt must not substitute for that validation.

Integration at `2b1441760` preserves the newer provider lifetime and passes fresh
qualification in Slurm 14923: default 1.0167437 and budget 1.0330590 across the
same two 1,792-call eight-group populations. Both actual GPU work/storage profiles
still match baseline, with 194 GPU/Python checks, independent native/installed
references, memcheck and all 27,475 input checks passing. These source-bound
receipts are retained as `integrated-cuda-*`; earlier raw measurements remain
recoverable through verified Git-ancestry manifests without raising storage caps.

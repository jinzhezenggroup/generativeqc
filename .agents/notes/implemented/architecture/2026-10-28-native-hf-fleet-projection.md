# Decision: prepared HF fleets own projection transactions

Status: implemented
Date: 2026-10-28

## Problem

The shared CPU system-projection owner already prepared overlaps and projected
all spin channels. Python still exported each retained source density, decided
source eligibility, created transient system owners, staged proposed seeds and
requested their installation. SDK consumers could not use the same transaction.
This is a remaining R4/R6 ownership gap in #2227, separate from #2841's scalar
plan admission policy.

## Decision

Add an optional prepared-HF projection facet backed by the existing `FleetPlan`.
The fleet borrows its retained source density, reuses `system_projection`, stages
every accepted candidate, and uses the existing physical warm-state validator
before a no-throw commit. The additive C ABI locks both owner contexts, handles a
shared context once, rejects running/closed owners and publishes detached POD
diagnostics only after the transaction succeeds.

Native eligibility is distinct from last-good seed storage. Replays revoke it
before input validation; only retaining a successful result grants it again.
Restore and clear revoke it too. An imported proposal followed by a successful
solve with frozen updates still contains the proposal, not the new solution.
Granting eligibility from the solve's status alone would incorrectly promote that
old density. This is why the grant belongs to `retain_warm_state`'s successful
caller, rather than a final scan over per-item statuses.

The frontend transports explicit policy/geometry and formats provenance. It does
not export a density or prepare a metric. Native report publication performs no
allocation after commit. Python report allocation can still fail; since targets
must start without seeds, clearing the target restores the pre-call state and
preserves the source. Cleanup failures retain the original exception.

## Boundaries and rejected alternatives

- Extending the system adapter with another Python eligibility loop would retain
  the production ownership gap and source-density copies.
- Reusing Python status/projection tags as native authority lets stale or changed
  frontend metadata misclassify a seed. Native eligibility is the authority.
- New projection equations or another physical validator would duplicate existing
  scientific owners. All metric, spin and physical checks stay shared.
- The native capacity covers simultaneously-live numeric matrix work and retained
  proposals, including final validation. It does not pretend to bound whole-process
  RSS, prepared arenas, copied topology metadata or opaque provider allocations.
- Unsupported CUDA/ECP/above-g metadata, older protocols and custom policies retain
  explicit compatibility. Entire-fleet metadata preflight precedes science.
  An admitted numerical/resource failure never retries another implementation.

## Invariants and evidence

Projection requires identical ordered nuclei/geometry, electron counts and spin.
An explicitly moved request must equal the geometry of the retained source
solution. Projection proposes a seed; target SCF and final verification remain
mandatory. Strict failures publish nothing; non-strict rejected neighbors remain
cold without discarding accepted neighbors.

`test_native_fleet_projection.py` exercises native-only routing, source immutability,
concurrent readers, exact/below numeric limits, malformed coordinate spans,
failed/stale/imported/frozen eligibility, transaction rollback and report failures.
Moved water/OH Direct/DF E/F comparisons use independent PySCF. The public-header
SDK test uses one shared context and covers RHF plus UHF with an empty beta spin.
The endpoint evidence records separate baseline/candidate libraries and complete
source/target execution rather than attributing a projection microbenchmark to
whole-application performance.

## Revisit when

Extend admission only with independent backend/model qualification. A future
complete resource ledger may replace the current matrix bound, but must preserve
honest provider/lifetime accounting and exact/below-limit transactional behavior.
General geometry extrapolation or frozen-seed provenance needs an explicit new
contract; neither should be inferred from the current projection API.

## References

- [Current public contract](../../../../docs/user/progressive_hf.md)
- [Runtime ownership tracker](https://github.com/jinzhezenggroup/generativeqc/issues/2227)

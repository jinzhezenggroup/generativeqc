# Decision: bounded exact-content reuse of stationary snapshot grids

Status: implemented; complete large-device qualification in progress
Date: 2026-10-02

## Problem

Clean-master PBE0/def2-SVP warm energy-plus-force endpoints spend approximately
0.94 s (12 atoms) and 1.90 s (24 atoms) rebuilding an `ExplicitGrid` and its
canonical JSON identity. The final native density and snapshot lease really do
change on replay, but the molecular grid commonly does not. Reusing an old
snapshot would be scientifically and lifetime-incorrect; repeatedly serializing
unchanged grid input is unnecessary.

## Decision

The public CUDA force batch lazily owns one `SnapshotGridCache`. Native snapshot
decode still checks its live lease, exact basis/model, density and occupations.
Only an implicitly constructed immutable grid can be reused, after comparing
every current point, weight and owner with the retained grid and requiring the
same native owner. Floating-point comparisons are bitwise, including signed
zero, to preserve the existing public canonical grid identity exactly.

Explicit caller-supplied grids keep their existing fail-closed content proof.
No density, force, SCF snapshot, device pointer or native lease is cached. A miss
releases the cache's old reference before constructing the replacement. Ragged
batches and changed geometries therefore retain at most one grid, not one grid
per item or geometry. Batch closure drops the retained reference.

The hard retention cap is 128 MiB, including owned numeric buffers, Python
owner storage and conservative object overhead. Oversized grids use ordinary
uncached construction. CUDA KS resource plans charge the cap as persistent
storage independently of the existing force staging reservation. CPU snapshots
remain uncached by default. Endpoint work publishes reuse, actual retained
bytes, budget and the number of source points checked.

## Rejected alternatives

- Reusing by geometry label, pointer or solve epoch alone cannot establish exact
  input equality and can accidentally authorize a stale native state.
- Changing the public grid serialization/hash would invalidate scientific
  identities solely to optimize an internal consumer.
- An unbounded per-geometry or per-batch-item cache silently changes the prepared
  resource contract and is especially risky for moving ragged batches.

## Evidence

The helper tests check bitwise changes, signed zero, owner identity, ragged shape,
immutable owned buffers, capacity fallback and invalid replacement. Real native
CPU handoff tests explicitly enable the same cache and verify that warm grid
reuse still rejects stale leases, false labels and supplied mismatching grids;
changed-geometry decode replaces the entry and zero-budget decode has the same
public identity. The focused host suite passed 62 tests with 11 GPU skips.

Local profiles and exact-source diagnostic journals are under
`.artifacts/pbe0-large-20261002/`. Baseline native library bytes are retained
separately from Python changes so cached-grid measurements do not imply that an
unrebuilt library contains newer native source edits.

## Revisit when

Reconsider the single-entry policy if a bounded multi-system working set is
measured to matter. A native exact grid digest could remove the remaining
linear comparison, but only with a versioned source proof and unchanged native
lease validation. Do not infer a force-result cache from grid reuse.

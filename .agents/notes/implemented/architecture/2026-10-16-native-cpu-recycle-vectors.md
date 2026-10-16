# Decision: retain and replace eligible CPU recycle vectors in the native runtime

Status: implemented
Date: 2026-10-16

## Problem

The public CPU recycle space retained dimension-sized vectors and performed
projection/replacement arithmetic in Python even when physical actions were
native. Moving only the outer RHS loop would leave those vector lifetimes and
scientific runtime operations outside the shared owner. Simply borrowing native
views into Python would also make retained tracebacks/diagnostics unsafe after
replacement or destruction.

## Decision

Add one native CPU numeric recycle owner, an additive public C bridge and a thin
Python adapter. Preserve the existing policy: old directions first, solution,
then Arnoldi columns; two-pass MGS; cutoff `1e-12*max(1, original_norm)`; no update
for failed solves. Reuse the existing scaled norm and CPU linear algebra. These
are vector operations, not duplicated physical equations. The existing RHF frame
recycle owner projects using operator images and has different semantics; it is
not substituted for this orthonormal-direction policy.

Creation is metadata-only. A shape query supplies additional capacities before
whole-solve admission acquires the handle or any numerical storage. Native
tracked vectors survive between RHS solves, with no normal Python vector
publication. Metadata validates signed/repeated strides and declared extents
before access, without stacking every candidate into a temporary panel. Capacity
does not imply work bounded independently of the offered basis: rejected or zero
directions may require visiting all offered columns. Production basis counts
remain controlled by the existing solver, and endpoint evidence retains actual
semantic work.

Public generation checks protect native vector state; physical method/reference
leases remain with the method adapter. Replacement and changed retained limits
commit together only after all visited candidates succeed. A failed attempt can
raise allocation telemetry but cannot publish partial vectors or a new generation.
Diagnostic column-range copies detach before return and survive destruction.

## Compatibility and rejected alternatives

- Keep explicit compatibility for missing protocols, zero dimensions, explicit
  seeds/transport, foreign data and non-CPU/resident engines. A selected library's
  load error or native error is never permission for a second numerical attempt.
- Do not mirror every native retained vector in a Python list between solves.
- Do not first copy the entire old native space when a foreign update needs a
  smaller destination. Stream old vectors through one reusable column buffer;
  preserve the old native owner until the compatibility replacement succeeds.
  Otherwise shrinking capacity can exceed the newly admitted temporary bound,
  and a failed candidate can erase the original space before rollback.
- Native update capacity preserves the existing replacement-plus-three-vector
  reservation. An already native space entering a foreign compatibility update
  reserves two additional vector equivalents for the streamed column and legacy
  working/coefficient/publication overlap. Old retained payload is charged
  separately. Ordinary native CPU admission is not inflated for an unused
  compatibility transition. Projection compatibility scales its reusable buffer
  in place to preserve the two-vector additional reservation.
- Do not claim that solver numeric payload telemetry is full physical endpoint
  allocation authority. Object/allocator/Python bookkeeping, detached diagnostics
  and physical provider storage have separate contracts.

## Evidence and remaining scope

Public-header-only SDK tests exercise independent directions/projectors, exact and
below-budget admission, changed limits, stale generations, signed strides,
overflow, failed-candidate rollback, zero capacity and detached publication.
Python tests use independent QR/dense oracles, scaled C/F/strided/reversed views,
failure injection, no normal diagnostic publication, bounded compatibility,
transport and whole-solve preflight. Complete CPU response endpoint comparisons
use preserved matching master Python/library artifacts, moved coordinates,
actual residual/convergence/work receipts and construction-through-cleanup time.

This slice does not migrate outer GMRES/RHS/block/recycled schedules, value-only
rank SVD, method-specific operators/RHS, seven-source molecular HVP or resident
GPU engines. R1–R6 in #2227 remain intact. Revisit these boundaries only with
independent numerics, lifetime/resource gates and complete endpoint evidence.

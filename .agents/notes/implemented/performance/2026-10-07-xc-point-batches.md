# Decision: batch independent XC point domains without merging AO maps

Status: implemented (qualification-only; default remains one tile)
Date: 2026-10-07

## Problem

#2073 records an eight-CTA physical-PBE point launch repeated for each of
9,216 256-point SCF tiles. CUDA Graph replay changes host submission overhead,
not the independent device work supply. Increasing individual tile size can
merge AO support and silently increase selected point-by-AO-squared work.

## Decision

Retain several independent compact AO panels, produce each tile's features
with the existing density provider, then submit their combined point domain.
The same canonical FP64 point consumer handles tile-local channel-major slots.
The tail is compact; a point maps directly to its slot without a device descriptor
or gather kernel. Reuse the original density/potential scratch because those
panels do not need to survive the point submission. Scatter Vxc and reduce totals
serially in the historical tile order.

The scientific compiler owns admission and point-domain mapping. Native code
owns optional allocation, lifetime, validated resource binding and stream
orchestration. Admission scans actual AO counts and checks products against a
finite byte cap before multiplying; it does not key on molecule/device identity.
The independent experimental controls request tiles and an additional allowance.
Actual retained bytes participate in the shared numeric resource ledger and the
ordinary KS XC resource diagnostic. An allocation failure retains the original
maps and one-tile executor, not a dense/CPU substitute.

## Rejected alternatives

- Merge selected AO maps: changes the admitted scientific work domain and can
  restore dense quadratic work merely to enlarge launches.
- Re-evaluate AO panels after batching features: conceals extra AO/jet work.
- Concurrent potential scatter with atomics: races or changes deterministic
  accumulation semantics without a separate numerical qualification.
- Retain every density-product panel: unnecessary data residency; scratch is
  dead after each feature contraction and can be reused.
- Promote on launch counts alone: complete cold/warm/moved E+F profitability
  and the independent source-specialization/composed ablations remain required.

## Invariants

- Original tiles, FP64 arithmetic, basis/grid/screening and AO maps are unchanged.
- AO, density, contractions, scalar reductions and matrix scatter retain their
  original work counts; only independent point submissions are batched.
- Responses and mixed arithmetic retain the one-tile executor. An admitted
  owner cannot subsequently discover another map or rebind mixed density.
- No allocation, host descriptor packing, point staging or additional explicit
  synchronization occurs inside evaluation/capture. Owner replacement on moved
  geometry recalculates residency; density changes do not reuse stale features.
- Optional residency is additional to the original full-capacity scratch, not
  described as replacing it. Tiny domains and insufficient resources fall back.

## Evidence

CPU generation/host-selector tests execute the emitted C++ planner and cover
ragged counts, high occupancy, empty tiles, compact tails, overflow-sized tile
requests, response/mixed/tiny rejection and exact byte bounds.
Native `--point-batches` exercises independent CPU E/V and exact same-tile
baseline equality, both spin layouts, spherical/cartesian bases, scaled PBE,
captured density changes, geometry-specific maps, resource rejection and teardown.
Complete endpoint evidence is retained separately; this note does not claim
default profitability or qualify #2072's source-specialization arm.

## Consequences and revisit conditions

More independent CTAs require AO residency. Contractions remain ordered, so this
does not address their launch count or promise potential-kernel acceleration.
Consider grouped provider contractions only with a separate ordered reduction
plan and independent complete E/V and moving-grid E/F gates. Promote batching
only after clean interleaved fresh-process/cold/warm/moved endpoint populations,
retained source/binary/generated identities, explicit work/memory counts, and
separate scheduling/source-specialization/composed comparisons establish a guard.

## References

- #2073; source-specialization sibling #2072; evidence owner #1965.
- `docs/developer/xc_native_cuda.md`
- `docs/maintainer/performance_engineering.md`

# Decision: automatic through-f dispatch with geometry-bound screened rows

Status: implemented; endpoint performance qualification in progress
Date: 2026-10-01

## Problem

The [previous canonical source](2026-10-01-canonical-public-ao-jk.md)
improved the complete three-atom OMol25-level endpoint but required an opt-in
and still traversed every pair-of-pairs before screening. At 1856 public AOs,
the dense canonical schedule has 1,484,875,413,456 candidates per radial pass.
Memory boundedness alone does not remove that work. Generic RSH derivatives
also repeated dynamic recurrence-order dispatch inside the force consumer.

The user requests best default performance, not additional performance knobs,
and suggests reusing successful HF modules where possible.

## Decision

Generated SPD value/force owners retain their existing selection, including
master's force-capable generated-exchange fix. Through-f plans automatically
select the canonical source when its existing budget admits the pair inventory
and six spatial-matrix scratch equivalents. No environment setting controls
this policy; test/profiler work counters are borrowed device observers and are
null in production.

Reuse HF's scientific compiler-owned contracted ERI evaluator, sparse public-AO
expansion, eight-permutation scatter, exact native Schwarz bounds, and stream
resource ownership. Do not widen the generated SPD class masks: missing f
source coverage does not become valid simply because its scheduling exists.

Derivative-capable canonical plans additionally retain the packed shell AO
offsets and shell-pair indices when budgeted. HF's prepared shell-warp
one-electron gradient kernel then borrows these arrays, native resident D/W,
and the already charged Direct derivative scratch. This eliminates the
standalone metadata pack/allocation/upload in the unchanged-geometry force
path. If these small optional arrays do not fit, keep the standalone bounded
one-electron bridge. Consume/download hcore/Pulay before RSH resets the reused
scratch; all operations use the owning stream and drain before return.

For geometry-only work admission, CUB sorts native FP64 Schwarz keys descending
inside each item/angular bucket. Each bra row binary-searches the admitted ket
prefix, with a sorted-order triangle for equal buckets. CUB inclusive scans
build row-work offsets. Value and RSH derivative kernels binary-search those
row offsets and enumerate only admitted quartets. Compare `bra * ket < cutoff`
directly; division by bra changes rounding at equality and is not equivalent.
Zero screening still admits zero-bound pairs, matching the original predicate.

The force consumers share one final-density RSH algebra helper, preserving
Full = Short + Long, spin weights, unique-atom contraction and translational
reconstruction. Only angular-homogeneous consumers instantiate a fixed
compiler recurrence order; the dense generic consumer uses the original
dynamic evaluator. Derivative outputs remain source-major J/SR-K/LR-K.

Optional screening storage is O(P), P = N(N+1)/2 per item: two FP64 key arrays,
two int32 ID arrays, seven uint64 row pages, segmented-sort offsets, and an
explicit workspace sized to the larger of CUB sort and scan requirements.
Charge all retained arrays and workspace before allocation. Geometry changes
rebuild the plan; unchanged SCF iterations reuse the sorted rows. There is no
quartet tensor, CPU integral/oracle preparation, or density-dependent pruning.

## Fallbacks and invariants

- If CUB's int-sized inventory or optional screening capacity does not fit,
  retain dense canonical traversal and its exact screening predicate.
- If pair/matrix storage does not fit, or mixed-J is requested, retain the
  existing generic bounded source.
- Allocation, driver and numerical failures remain errors, not silent success.
- Preserve the requested radial operators, FP64 storage/accumulation, AO order,
  output masks, item offsets, and stream-ordered lifetimes.
- Do not remove the current 1024-AO stationary-force cap without independent
  resource and numerical validation; 96-atom def2-TZVPD currently exceeds it.

## Evidence and acceptance

Native qualification includes independent CPU full/SR/LR matrices through f,
both representations/spins, nonsymmetric densities, two geometries, output
masks, minimum-budget generic and dense-canonical admission, screened work
counts, exact-product/equality/empty-row probes, and displaced CPU-value
derivative gates. Complete cold/warm/moved RKS and UKS endpoints must retain
host force return and independent energy/force gates. Historical preview
timings are never relabeled as this build. Passing one small endpoint does not
establish the requested performance through 100 atoms.

## Revisit when

Profile complete larger endpoints before extending source coverage. A shell
Cartesian f consumer using HF's generated source contracts may reduce public-AO
expansion amplification, but requires explicit complete source inventory and
independent full/SR/LR value and force gates, not a broader capability mask.

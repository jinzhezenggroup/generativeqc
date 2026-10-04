# Decision: retain RCCSD DIIS histories and Gram entries by physical ring row

Status: implemented; complete endpoint qualification pending
Date: 2026-10-05

## Problem

Issue #1900 identifies avoidable work in the native CUDA RCCSD owner. At full
history, it shifts both complete amplitude and residual histories; every solve
then rebuilds the Gram matrix. Current master already computes only one triangle,
so the correct baseline work is `h*(h+1)*N/2` scalar dot summands, not `h*h*N`.
Here `N=o*v+o*o*v*v`. These costs must not be mislabeled as CC contraction FLOPs.

## Decision

Keep histories in fixed device slots with method-neutral host ring metadata.
Store the Gram matrix with a fixed physical stride equal to history capacity.
Every insertion updates only the new physical row/column. Even the first
insertion computes its self norm, so subsequent steps need not revisit it.
Logical wrap or dependent retirement never shifts complete history tensors.
The small augmented DIIS solve and weighted sum traverse the original
chronological order using physical indices. Retiring a dependent row requires
no new dot products and leaves the remaining Gram entries valid.

The shared tensor layer owns ordered Gram-row and slice-combination primitives.
No CC-local vendor call was added. Exact FP64 rounding, the existing 256-lane
dot tree, coefficient limits, pivot thresholds and final physical replay remain
unchanged. The independently prepared resident-JIT adapter retains its existing
dense-history ABI, including its compatibility shift; both owners share one
slice-combination primitive. The new transitive header is included in installed
assets and JIT dependency identity rather than relying on checkout-only paths.

## Resource and failure invariants

- All histories and the physical Gram remain on device. Device array capacity
  is unchanged; no temporary chronological copy or extra full vector is added.
- New insertions write `2*N*sizeof(double)` destination bytes; ring retirement
  writes zero history bytes. A live count `k` computes `k*N` dot summands and
  writes `2*k-1` Gram scalars. Successful combine work is `k*N` summands.
  Device-status-guarded rejected combinations count as launches, not executed
  arithmetic. Counters distinguish these quantities explicitly.
- Exact-zero Gram preserves the trial and live history. Singular/nonfinite Gram
  preserves the existing retire-oldest/retry policy. First-row and failed solve
  paths preserve existing host drains and trial-time attribution.
- The public `diis_size=1` validation remains rejected; tests of a single live
  row and the storage primitive's capacity-one case do not change that API.
- Later symmetry packing (#1902) must preserve the full Euclidean metric and
  cannot treat these full-layout counts as packed-coordinate counts.

## Evidence and retained alternatives

The new real-device regression compares every Gram element against an
independent host long-double dot and the documented exact FP64 reduction tree.
It compares coefficients/status with a freshly reconstructed chronological
Gram and compares weighted sums in chronological order. An intentionally altered
old-old entry must survive a row update unchanged; ordinary numerical agreement
alone would not prove that old dots were skipped. Capacities 1/2/3/8/20,
lengths 1/17/256/259/4097, wraps, clears and exact-zero/singular cases are covered.

n2 job2275 passed these 75 GPU cases and 26 pytest cases in total. Its one
failure was an extracted constructor test missing the new ring header; the
test adapter was corrected, without changing the production constructor.
The corrected constructor/asset checks passed in job2276 (three tests; one
clang++-dependent test skipped). Baseline and candidate complete Release CUDA
builds passed in jobs2273/2274. Job2277 passed all 14 enabled native DF CUDA
solver cases, including independent determinant comparisons; its separate
resident-JIT compatibility tests are still running. Job2279 passed 13 CPU
solver cases (one CUDA-provider-only case skipped), compiler/SCF/cross-method
structure, default-promotion inventory, method metadata, native complexity and
CUDA ownership checks.

Complete same-allocation endpoint comparisons are in progress in job2280 on
an n2 RTX PRO 6000 Blackwell, with independent processes, alternating baseline/
candidate order, history eight, water7 and ethane230. No endpoint gain,
tiny-domain promotion or issue completion is claimed from these tests alone.
If the complete tiny endpoint favors the old schedule, retain a specifically
bounded fallback with its evidence.

Revisit dot parallelism only with explicit deterministic/reduction-order
qualification: changing the tree is a separate numerical change and should
not be hidden inside a bookkeeping optimization. Keep conventional and DF
solver evidence separate from orbital Z, packed state, Q batching and derived
denominator qualifications (#1901–#1904).

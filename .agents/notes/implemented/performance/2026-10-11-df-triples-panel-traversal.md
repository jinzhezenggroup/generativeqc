# Decision: bounded serpentine traversal for FP64 DF triples panels

Status: implemented
Date: 2026-10-11

## Problem

After distinct occupied W reuse (#2232), integral panels still account for 152
GEMMs in the o9/v221/Q488 fixture. The three-panel LRU repeatedly evicts lower
occupied indices in the original lexicographic traversal. W and epilogue work
must remain complete; adding a resident full-ovvv frontier would change the
resource contract unnecessarily.

## Decision

Use a constant-storage serpentine triangular visitor only for actual FP64
storage/compute/accumulation and exactly three panel slots. Keep the maximum
occupied index outermost, reverse the lower row with its parity, and alternate
the intermediate index direction by the lower row's parity. Native allocation,
LRU, panel/W products and scalar kernels remain unchanged. One/two-panel,
non-FP64 and force/response execution preserve original traversal.

Each tile publishes to its original flat occupied address, not visit count.
The final energy reduction therefore consumes exactly the original canonical
array order. This separates production scheduling from floating reduction
ordering. Layout preflight already bounds the cubic/triangular integer products.
The helper visits every tile once in O(T) work with constant host state; it
never sorts or retains an O(o^3) tile list.

## Rejected alternatives

- cuBLASLt ladder selection retains an unresolved attributable-resource and
  concurrency boundary; do not repeat its old pilots to justify this schedule.
- Retaining all occupied panels would need additional numeric storage/admission.
- Greedy global tile ordering can add O(T^2) host work and a tile list.
- Changing final energy reduction order is unnecessary numerical drift.
- Protecting current-tile cache residents does not improve the best offline
  schedule, so do not add that policy complexity.

## Evidence

The actual compiled helper covers all tiles/canonical slots for occupied sizes
1..32. Six independent CPU cases execute already qualified emitted leaf kernels;
each original/candidate canonical tile-energy array has identical bits and
passes the independent original energy oracle. These are not GPU timings.

Slurm's new candidate matrix has 12 outputs covering six fixtures, a >256-point
tail, deterministic repeat, one/two-panel fallback, generated-provider budget
fallback, one-byte-short refusal and all-equal overflow with unpublished
sentinels. Memcheck and initcheck each have two representative actions and zero
errors. The immutable baseline library is exactly the prior #2232 qualified
library, so its passing fixture matrix is reused rather than rerun.

A separate bounded follow-up tests only the two previously uncovered changed
domains: three-panel generated-provider fallback and an unsafe all-equal seed
in a three-panel request. Both memcheck and initcheck pass these actions with
zero errors. The generic action reports three slots, six panel builds, strict
FP64 and independent energy agreement; overflow preserves unpublished sentinels.
No endpoint or full matrix is repeated for this focused addition.

One separate candidate-then-control complete endpoint pair is
48.093351139 / 48.491323059 seconds: 0.8207% shorter wall, 1.008275x.
Triples is 3.757277946 / 4.135664454 seconds: 1.100708x. Both arms retain 19
CC observations/evaluations and all non-timing CC counters. Total/triples
energies are unchanged; original independent gates remain 1e-8 Eh total,
1e-10 Eh triples and 1e-10 physical R1/R2. Do not pool with #2232 or claim
statistical/global performance.

Panel GEMMs fall 152 -> 98; W evaluations stay 729, moment GEMMs 1458, and
total triples GEMMs 1610 -> 1556. Contraction summands fall
2610452107406 -> 2326012282334, saving 284439825072. Every occupied energy tile,
virtual point and projected contribution remains. Device numeric CC capacity
8583749632 and complete endpoint host/device capacity 8945677650 bytes are
unchanged. Process peak RSS delta is -1572864 bytes in this pair, not a universal
memory claim.

Only one native owner object rebuilds with verified ccache; scientific generated
artifacts and existing ABI types remain unchanged. Borrowed link inputs are
checksum-verified immutable. Current source formatting is separately hash-bound
to the exact compiled bytes by a retained formatter-only receipt, not silently
treated as byte-identical. The measured library freezes #2171 HF and #2221/#2232
CC closure, not a full latest-master HF/force build. Broad performance/production
stages remain not-run. Master #2218 affects only offline CPU XC AOT and does not
justify repeating passing CC/GPU suites.

## Consequences and revisit

Bounded traversal reduces executed panel work without another provider or
resource frontier. Maintain the canonical energy-address contract if tile order
changes again. Extend to other cache capacities/precision/response only with
separate independent gates and complete endpoint evidence. Old numerical
publications bind their measured source snapshots; current-owner source matches
are checked by this newer qualified publication rather than asserting that
today's source still equals an older performance snapshot.

## Publication milestone master assessment

Master advanced through 20baf5826 during preparation: generic hybrid HVP rules,
CLI/DF DFT sources, fail-closed KS matrix-provider qualification, and removal of
integral/validation device-specific defaults. None changes the DF triples owner,
generator, CC/scalar/tensor consumers or this helper's inputs. Bookkeeping edits
in the promotion checker/inventory are separate from the measured consumer.
No CC matrix, endpoint or unrelated DFT/HF/force suite is repeated for these
advances. The latest-master full-library/HF/force qualification remains not-run.

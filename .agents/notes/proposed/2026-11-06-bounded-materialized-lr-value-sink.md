# Decision: fixture-only canonical LR sink for the materialized pair recurrence

Status: proposed; implementation and host fixtures, device/native integration unqualified
Date: 2026-11-06

## Problem and decision

The current WB resident LR source uses scalar canonical quartet production. Its
profiled generic family also permits fused contraction, so the full family sum
is not proved source-production cost. Reuse the existing compiler-owned
`MaterializedDirectPairRecurrence` primitive loop with a compile-time terminal
sink instead of adding another integral algebra or recurrence owner.

The ordinary Fock wrapper preserves its signature and default behavior. The
candidate wrapper is strict FP64, LR-only, derivative-free Cartesian SPD at total
orders 3..5, with the incumbent 32-lane whole-shell schedule (1/3/6 slots). It
binds null density/Fock pointers and compiles out the Fock terminal consumer.
Every admitted scalar is written once, including exact zero. Component screening
retains the direct Schwarz product predicate and primitive/AO multiplication
order; no density screening, exchange factor or orbit multiplicity is applied.

A fixture-only map contains at most 192 component entries for one shell task.
The CTA validates capacity, screened/tail sentinels, destination bounds and local
uniqueness before publication; malformed maps refuse uniformly. This is not a
molecular task map or production allocation. The shared recurrence is unchanged
and new candidate instantiations assert at most 16 KiB of static source storage.
Actual linked shared/register/local/stack/spill resources remain unmeasured.

## Independent fixture and gates

The host fixture independently enumerates Cartesian AO pairs, sorts descending
Schwarz bounds with deterministic ties, builds inclusive screened row prefixes,
and proves inverse canonical/component inventories. Compact slots use previous
row prefix plus second rank; dense cross slots use row times second count plus
second rank; dense same-bucket slots use sorted-rank triangular offsets. A
same-bucket union of two shell pairs is covered by its three shell tasks, so
there is no untested canonical population hidden outside a task's map.

Host tests cover every SPD composition of orders 3..5, pair/AO swaps, repeated
indices, identical shell pairs, ties, exact threshold products, empty/partial/full
rows, signed raw-K orbit contraction and capacity/overflow/request refusal.
Reference preparation uses pinned PySCF 2.14.0/Libcint Cartesian `int2e` LR and
the existing radial/component normalization adapter. Inputs have signed unequal
primitive contractions, separated/coincident/reversed centers and omega
0/0.03/0.3/3.0. LR(0) is analytically zero: PySCF's omega=0 full-range convention
must not be confused with a zero long-range source.

The standalone executable contains candidate, incumbent scalar source and
ordinary Fock kernels, without linking the native library. It poisons guards
and screened holes; checks one-write coverage, source/null-matrix isolation,
actual preparation/contraction/publication work, both unchanged Libcint error
gates (2e-12 + 2e-10*abs(reference), and 3e-11 + 3e-11*abs(reference)), raw K from
both source arrays and ordinary Fock, and ordinary plus repeated graph execution.
No device pass follows from host preparation. CUDA compile, resources and
memcheck/racecheck/synccheck require a separately authorized finite execution.
No work-count ratio is called speedup.

## Retained adverse evidence and ownership gap

The old `origin/perf/wb97m-shell-component-values-20261004` proposed note
`.agents/notes/proposed/2026-10-04-shell-component-value-consumer.md` retains
initial synchronization failures, two failed revisions, later sanitizer
qualification and the complete three-atom cold/warm regression. That Fock
consumer's preparation reuse was not an endpoint win. This candidate changes
the terminal sink only; it inherits no performance or synchronization verdict.

The native resident-source-only clone does not automatically own the generated
primitive-pair cache. Before any native integration, a separate design must
borrow an already admitted cache owner, charge inverse pair ranks as O(NAO²),
retain the current output allocation, and refuse before publication on resource
or ownership failure. This fixture adds none of those production owners. It
adds no selector/default, lease, molecular task inventory, arena, quartic cache,
capacity increase, range fusion, density contraction, derivative or SCF change.
The #3020 VV10 provisional handoff and the separate shared-XY work are untouched.

## Revisit when

Fresh standalone device/resource/sanitizer gates pass, followed by a separately
approved source-only replay against actual native rows/cache/bounds/offsets and
output slots. Only actual-task cost including mapping/preparation, then complete
endpoint evidence, can support integration or performance claims.

# Proposal: retain the generated ERI component index in each Cartesian AO

Status: proposed; local qualification prototype, no performance/default promotion
Date: 2026-10-03

## Problem

The value-only s/p/d shell-quartet producer classifies four immutable Cartesian
angular triples for every canonical contracted component. The existing HF96
source census has 100 Cartesian AOs and 12,753,775 canonical components, so
51,015,100 repeated classifications can become 100 classifications during AO
expansion. This is a source-derived work opportunity, not a measured speed claim.

## Decision under qualification

Cache the unsigned result of `generated_eri_cpu::component_index` in `AoView` at
its sole construction site, `expand_cartesian_aos`. Read that result in
`prepare_value_eri_components`. Keep the generated helper as the sole mapping
and unsupported-angular sentinel authority. Keep generated `component_record`
range validation and record lookup unchanged.

Place the unsigned field before the double normalization so existing padding
can be used where the ABI permits. The regression measures old/new `sizeof` and
member offsets; it requires no growth when the previous layout has appropriately
aligned space. No cross-platform layout assumption is imposed on production.

## Invariants and boundaries

AO/shell pointers, angular triples, component ordering, normalization association,
record lookup, primitive loop ordering and floating-point reductions are
unchanged. No new recurrence, template specialization, component map, pair cache
or generated artifact is introduced. Derivative, f/g-containing, range-separated
and omitted-ERI dispatch remains unchanged. Their AO construction stores the
same unsupported sentinel when appropriate and never uses it to select a route.

The cache adds one bounded classification per Cartesian AO to paths that do not
consume generated value ERIs. Measure complete endpoints before deciding whether
that tradeoff is worthwhile. Whole-tensor and endpoint bitwise equivalence to the
frozen shell-local parent is required; earlier shell-local versus dense-transform
rounding differences are a separate, inherited decision.

## Qualification

The focused compiled regression uses the actual `AoView`, AO expansion and
component-preparation source, actual native basis enumeration/normalization and
freshly emitted generated classification/map/lookup code. Instrumentation counts
classifications without substituting their meaning. Every s/p/d assignment of
one through four shells is traversed, with component index/order/record and
bitwise normalization checks. Forward/reversed s/p/d/f/g expansion and each
unsupported/out-of-range quartet position are checked. Complete tensor,
independent recurrence/oracle, endpoint and fallback results are recorded in the
local qualification evidence; no performance timing has been authorized.

## Rejected alternatives and revisit conditions

Hard-coded s/p/d numbering would create a second authority. Skipping validation
would weaken the sentinel boundary. Pair caches and generated specializations
would materially expand scope. Revisit only if exact endpoint evidence shows a
regression or the generated domain/layout changes.

### Initial local gate (2026-10-03)

On x86-64 with GCC 14.2, `AoView` remains 32 bytes: shell offset 0, angular
triple offset 8, cached index offset 20, normalization offset 24. The actual
helper test checks 5,079 shell blocks, 661,214 prepared records and 1,420 expanded
AOs, plus all forward/reversed s/p/d/f/g components and unsupported sentinels.
The generated/libcint and projection Python controls pass 43 tests; eight linked
native controls pass, including existing independent RawSource, spherical
physical-reference, derivative and omitted-ERI gates.

Fifty-two paired binary artifacts (10,844,240 bytes per variant) match the frozen
shell-local parent bitwise. These contain complete Cartesian/spherical signed
s/p/d and dddd tensors, reversed shell storage, f/g fallback tensors,
derivatives, omitted-ERI and short/long-range results, and RHF/UHF cold/warm/moved
energies, densities, forces and convergence fields with both CPU HF selectors.
Twelve additional paired HF96/PBE24 cold/warm/moved lifecycle points match in
every recorded scalar and density byte. Against retained independently generated
exact-basis PySCF references, the maximum energy error is 3.02e-12 Ha and density
error is 3.57e-10. HF96 forces were not run. The HF-only selector is not claimed
to select the DFT solver.

An exact source reconstruction permits only the field insertion, generated
initialization and cached reads. All other production bytes, basis enumeration,
compiler emitter and generated header are unchanged. No performance endpoint
clock was collected, no promotion decision is implied, and clean timings remain
a separate reserved gate. All harnesses, raw logs, binary comparisons, identities
and failed tooling/harness attempts are retained in the sibling local evidence
directory.

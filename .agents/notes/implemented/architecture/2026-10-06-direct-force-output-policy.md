# Decision: one Direct force task for Combined and Separate outputs

Status: implemented
Date: 2026-10-06

## Problem

HF's precontracted shell tasks and stationary DFT's J'/K' tasks used the same
compiler force roots through different coefficient and primitive consumers.
That boundary prevented DFT from borrowing HF's resident-bra execution without
duplicating either the traversal or the force equations.

## Decision

`DirectForceOutputMode` selects a compile-time source layout in the existing
precontracted task: Combined contracts the caller's signed J/K coefficients
into one weight array; Separate contracts independent J and K weight arrays
with one shell/AO traversal and one geometry/Boys ladder per primitive product.
The long-range exchange consumer retains its single-source layout.

A resident view may supply the canonical bra's complete `PrimitivePairData`
array in global-cache order. Only an exact count match admits this view; missing
or partial storage uses the original global cache. The queue owns the view's
lifetime and stream ordering. No additional method coefficient is introduced.

## Invariants

- Separate source activity never depends on cancellation of J and K.
- AO Schwarz admission, raw/canonical shell orientations, repeated-atom collapse
  and translation recovery retain the historical consumers' behavior.
- Primitive multiplication/reduction order and compiler roots are unchanged.
- Each output retains the native `-dE/dR` convention. The method publisher owns
  any subsequent conversion to energy derivatives.

## Evidence

On node1, the host probe in `test_direct_low_order_sources.py` compared against
the retained independent HF scalar consumers across all low-order angular
orientations, spin layouts, repeated atoms, disabled sources and AO masks:
19,540,224 coordinates were bitwise equal. Separate geometry construction
decreased from 1,311,456 to 1,043,280 calls with unchanged 1,311,456 weighted
force calls. Both output modes also passed complete and partial resident-view
checks. This qualifies task semantics, not a complete GPU endpoint speedup.

The branch starts from #2007 (`419551e7f`), whose force active-AO cost admission
must remain the baseline for subsequent endpoint qualification.

## Rejected alternatives

Two HF traversals would repeat immutable source work and hide independent J/K
screening behind a method adapter. Copying the force equations would introduce
a second mathematical owner. Neither is needed for an output-layout policy.

## Revisit when

A different primitive layout or compiler root changes the canonical resident
identity or reduction order. Qualify that change independently before admitting
its view to this task.

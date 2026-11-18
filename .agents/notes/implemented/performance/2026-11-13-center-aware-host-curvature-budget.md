# Decision: physical-center work budgets for the host frozen matrix

Status: implemented
Date: 2026-11-13

## Problem

The source-bound ICH3I control in
`build/qualification/host-stationary-eri-orbits-20261113` finally returns a native
stationary energy/gradient that passes independent saved-state comparison. Its
next raw-Hessian request is refused before execution by the unchanged complete
one-billion-primitive budget. The matrix source already evaluates one scalar
primitive when all slots have one physical owner, or three mixed primitives
per distinct non-origin owner otherwise. Its planner nevertheless charged every
quartet the maximum number of distinct owners in the entire molecule. This is
a conservative upper bound, not proof that the actual schedule exceeds the cap.

## Decision

Keep the primitive schedule and caller's cap unchanged. Price the matrix source
using its existing physical owners. For canonical triangular AO pairs, retain
the weighted pair-prefix sum for all pairs, for each atom's complement, and
for each single-atom subspace. Radial primitive counts and all public-AO
Cartesian expansion terms remain in each pair's positive integer weight.

Let `Q` be the complete canonical quartet weight and `Q_without[a]` the weight
excluding atom `a`. Then `sum_a(Q - Q_without[a])` counts each quartet once per
distinct owner. Subtracting `Q` removes its common-origin owner; multiplying by
three gives exactly the schedule's mixed directions. Add the single-atom
quartet weights for the scalar-only visits that would otherwise count zero.
All additions and products remain overflow checked. No coordinate equality,
density screening, zero-coefficient assumption or changed scientific gate is
used to reduce this count.

The planner runs in `O(natoms * nao^2)` with `O(natoms)` integer metadata, not
an AO4 traversal. It does not retain numeric derivative storage. First-source,
column and ordered-comparison plans, map-work ceilings and execution counters
are unchanged. The latter still expose the actual completed work.

## Validation contract

Test externally observable finite-budget admission against a separate exhaustive
enumeration of small public AO recipes: Cartesian and spherical s--f,
contracted primitives, both spins, and one through 32 physical owners. A budget
containing all required visits must bind successfully; a cap below the
published resource plan must fail. Do not require the independent envelope
to be a lower bound: a future schedule may legitimately execute less work.
Coincident centers remain distinct owners. Existing independent numerical
primal, raw-matrix, translation, alias/capacity, failure/recovery and ordered
comparison tests remain required; planner changes are not a numerical waiver.

## Rejected alternatives and limits

- Raising the caller's cap hides the distinction between excessive execution
  and an overly pessimistic estimate.
- Enumerating all AO quartets solely to price admission creates a new work
  cliff before the budget check. Even grouping and crossing every atom pair
  would introduce unnecessary `O(natoms^4)` metadata work.
- This change does not accelerate the primitive consumer. A lower estimate is
  not measured endpoint work, a practical runtime result, a raw-Hessian/mode
  qualification or permission to rerun an unchanged expensive endpoint.

The positive native saddle-frame, stationary mixed Ni/ECP control, complete
25--40-atom endpoint and requested Ni transition state remain separate gates.
Both public ECP saddle guards stay closed. See the source-matched qualification
cohorts for actual counts and terminal statuses, not this architectural note.

## Source-bound metadata result

`host-center-aware-curvature-plan-20261113` binds production source
`dd2f775fb72004ebb3ed2003ba4722e1c490b6cf7971f6f12d1821fb8f17078f`
and library SHA-256
`b3381873e6997b0147782e3f487d7e974aa09982dc767010567f25e166a355d2`.
A bounded, metadata-only native probe creates the complete unchanged ICH3I
system/basis, then independently enumerates its canonical public-AO quartets.
It performs no SCF, gradient, Hessian or displaced-state calculation.

At 81 AOs, 73 radial primitives and six physical atoms, the independent
canonical quartet weights by one/two/three/four distinct owners are 11659450,
109830363, 100809248 and 15439893. The matrix source therefore requires
1084965064 primitive visits, matching the new planner, rather than its old
2139650586 worst-case allowance. Adding the single first-source sweep gives
1322704018 visits even in the optional resident matrix/panel schedule. A storage
fallback can cost more, not less. The unchanged cap is 1000000000, so this
control still cannot be admitted. No expensive electronic endpoint is rerun.

This proves both parts of the diagnosis: the old allowance was excessively
conservative, but the incumbent primitive schedule also genuinely exceeds the
requested cap. Tightening the planner does not fix the positive native raw-H
and mode gap. Another exact algorithmic reduction is required; increasing the
cap, changing the physical basis/grid or renaming work units is not a fix.

The independently qualified stationary energy/gradient belongs to the frozen
parent source `40428aca0ef6af0d4ed91caffdbb6493c270011d71f2d37045845b0fc89c8e1b`,
not to a newly executed electronic endpoint under this planner-only producer.

The source-matched final host nuclear-integral module passes all 209 cases
without skips, including independent primal/FD raw-matrix checks, both spins,
both representations, s--f contractions, distinct/coincident owners, 32-site
source coverage and capacity/alias/failure recovery. Ten native/API/CLI groups
also pass, including the shared molecular evaluator and host-Hessian owner.
The initial 240-second outer test timeout is retained as a compile-time failure,
not a scientific result; the final complete module passes in 157.34 seconds
with the verified shared compiler cache. Native groups take 45.96 seconds.
These times are local validation, not molecular endpoint performance claims.

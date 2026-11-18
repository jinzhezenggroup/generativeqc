# Decision: bounded native frozen integral Lagrangian curvature

Status: implemented (frozen integral component qualified; stationary endpoint unqualified)
Date: 2026-11-12

## Problem

The native nuclear response owner already has actual first S/F directions and
coupled D/W response. Primitive second derivatives alone cannot supply the
frozen integral term in stationary assembly: contracted normalization,
spherical expansion, atom-slot incidence, ionic charges and canonical method
weights must all participate in the same bounded source transaction.

## Decision

Add `HostKsNuclearIntegralCurvature` alongside the existing first-integral
source. Extract their common immutable normalized frame, model validation,
public AO metadata and borrowed-state alias guards into one translation-unit
owner. The existing first source retains its interface and mathematical
projection. The curvature owner additionally borrows fixed energy-weighted
density W and requires its enclosing owner to authenticate D and W contents
and epochs. It does not certify electronic stationarity itself.

Factor the canonical ordered J/K energy objective out of
`quartet_direction_program`. Both the original density-VJP/integral-JVP Fock
map and the new integral-VJP energy weight now derive from that objective.
One-electron and overlap-constraint weights come directly from
`StationaryGradientPlan.integral_block`. No native spin, occupation, exchange
or Pulay coefficient is introduced. An ordered AO pair/quartet's final weight
is computed once, before traversing its normalized primitive expansion.

The source contracts existing AOT mixed S/T/V primitives and the canonical
factorized ERI value/gradient/gradient-JVP library into four compensated
scalars. Independent mathematical slots bind to actual physical atom motion,
including repeated atoms. ERI coordinates and both motion legs use the final
mathematical slot as common origin; the S/T/V primitives already provide
their own common frame. Rigid-translation zeros do not rely on cancellation
among separately evaluated center derivatives.

The four fields are the frozen integral Lagrangian
`[L, L_left, L_right, L_mixed]`, where
`L = Tr(D (T + V_ionic)) + E_JK(D) - Tr(W S)`. The value is not the full
electronic energy: the constraint term is deliberately included, and residual
ECP, XC and classical/nuclear contributions are separate. D/W relaxation is
also separate. A stationary Hessian must compose these sources rather than
reinterpret this frozen component as a complete relaxed endpoint.

## Resource and publication invariants

- If M sums radial primitive counts times Cartesian expansion terms over
  public AOs, full source visits are `M^2 * (natoms + 1)` for combined S/T and
  ionic attraction, and `M^4` for ERIs. Every ordered AO pair/quartet is
  consumed once. Zero weights do not bypass malformed-state or resource checks.
- Owned FP64 heap staging contains four compensated output fields, bounded
  weight-map inputs/output and the largest generated weight-map arena. It
  does not grow as AO matrices or coordinate/AO tensors. Integer/AO metadata,
  borrowed state, primitive stack, allocator overhead, RSS and time are outside
  this numeric query.
- Numeric bytes, both complete primitive-visit counts and pair-plus-quartet
  generated map work are mandatory caps checked before numeric allocation.
  Explicit quartic work is not favorable scaling or endpoint qualification.
- Validate the actual owner after each AO pair, at a 1--4096 quartet cadence,
  and before final publication. Synchronous reentry refuses. External motion
  buffers may alias output; aliases into any borrowed state, including W,
  refuse. All four fields publish together only after final validation.
- No production displacement, Python/reference fallback, JIT, AO-fourth-power
  integral storage or coordinate-squared derivative storage is introduced.

## Rejected alternatives

- Native hand-coded D/W/exchange coefficients would create a second method
  equation owner, especially dangerous for RKS total versus UKS spin density.
- Materializing molecular integral Hessians before applying final weights
  would multiply derivative storage and retain unnecessary scientific data.
- Passing unauthenticated W only at the final reduction would detach the
  overlap-constraint curvature from the source's immutable owner transaction.
- Reimplementing ERI recurrences in the new consumer would split the canonical
  derivative authority. Build-only emission of the existing factor library
  preserves that ownership.

## Evidence and acceptance scope

Qualification artifacts are retained under the ignored
`build/qualification/host-integral-curvature-20261112/` directory. Final related
regression passed 167 tests without skips; nine CMake tests pass against the
rebuilt core, including actual nonzero frozen-curvature source work with Python
absent from PATH/PYTHONPATH. Six selected CPU CTests also pass. The first
shared-frame regression separately passed all 23 existing first-source tests;
their two generated quartet-map identities are unchanged.

Nineteen retained independent normalized libcint primal-energy controls include
RKS/UKS and Cartesian/spherical representations through f, contracted s/p,
single-radial d/f, and separate Ni/I/Cs **ionic-tail-only** controls. They are
not residual-ECP or complete heavy-atom endpoint qualification. The actual
normalized packed AO basis, signed D/W, prescribed legs, baseline and all
four displaced primal corners at each step are retained. At steps
`1e-2 / 3e-3 / 1e-3`, the unchanged final `atol=3e-8, rtol=2e-6` derivative
gate and multistep convergence gate pass. Maximum final left/right/mixed
errors are `1.574e-7 / 1.558e-7 / 2.397e-8`; the two larger first-direction
errors satisfy the relative gate for the higher-charge ionic controls.

Separate controls cover every resource cap, model/method rejection, finite
indefinite and empty D/W, a nonvacuum UKS empty spin, leg symmetry/bilinearity,
exact rigid translations, bad directions/W, external aliases, borrowed-state
alias rejection, authenticated D/W mutation, replacement/reentry, late/final
owner failure and recovery. Strict C++20/O2/Wall/Wextra/Werror builds use
`-fno-fast-math -ffp-contract=off` and verified sccache 0.16.0. Changed hooks,
compiler structure (610 modules, no dependency errors), type checks and the
actual warnings-as-errors Sphinx build pass. These results qualify the frozen
source, not stationary assembly or full endpoint performance.

## Revisit when

The subsequent [source-driven column decision](2026-11-12-native-frozen-integral-columns.md)
implements the column consumption requirement below, with explicitly counted
finite-slot S/T/V replay. The stationary endpoint and complete electronic
qualification remain separate.

The current scalar-pair API is a mathematical source component, not a full
Hessian scheduling decision. Do not assemble a complete molecular Hessian by
replaying the entire ordered-quartet traversal for every left/right coordinate
pair. A source-driven column/block consumer must reuse each ERI gradient-JVP
across its left-coordinate contractions, and reuse finite S/T/V center-factor
jets before physical-atom scatter. Count all fallback tile replays explicitly.
The 25--40-atom complete electronic work/time gates remain unqualified until
that enclosing schedule and the other stationary sources are exercised.

Complete-source measurements justify batching immutable prepared ERI geometry
or resident contraction data. Retain the explicit bounded fallback, owner and
complete-work accounting; do not trade them for a scalar microbenchmark win.
Higher angular momenta or new range-separated/fitted models require their own
canonical weights, source contracts and independent endpoint gates.

## Subsequent local-right matrix schedule

`../performance/2026-11-13-source-local-frozen-integral-matrix.md` adds an
explicitly reserved coordinate-only frozen matrix and compiler-owned D'/W'
contraction. The mixed and streamed-column schedules here remain bounded
fallbacks with their unchanged scientific contracts; historical source-matched
receipts are not reassigned to the new implementation.

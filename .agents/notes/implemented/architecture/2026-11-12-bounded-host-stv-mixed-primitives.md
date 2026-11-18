# Decision: factor-scheduled native mixed S/T/V primitives

Status: implemented (primitive component only)
Date: 2026-11-12

## Problem

The native first-integral source does not supply the frozen second derivative
needed by stationary Hessian assembly. The compiler already owns the S/T/V
component DAGs and second-order AD. Expanding every coordinate Hessian of every
Cartesian component repeats large Hermite/Coulomb expressions and creates a
needlessly large generated compilation unit. Introducing another recurrence or
using displaced production integrals would violate the mathematical ownership
and analytic-source contracts.

## Decision

Expose the exact Hermite factors, Coulomb factors and Coulomb geometry from
`OneElectronComponentKernel`, retaining the original component expression and
canonical builder. The existing first-order emitter does not consume the new
boundaries. A one-off before/after default CPU generation comparison retained
identical bytes; this is migration evidence, not a permanent hash-pinned test.

`integral/one_electron_mixed.py` substitutes those factors with four-field
`[value, left, right, mixed]` jets. Existing Graph AD differentiates each
Hermite factor once per `(i,j,t)` and the existing Coulomb derivative algebra
provides the attraction factors. The component DAG then composes their jets.
S/T share factors and scalar common-subexpression elimination. Nuclear
coordinates are affine: their mixed input tangent is an actual Graph zero,
not a variable merely bound to zero after emission. Exponents remain fixed.

The build-only generator produces a finite AOT C++ source for all 400
Cartesian component pairs through f. The optional native KS response target
links this source. Native execution retains fixed-size scalar/stack data and
uses the existing range-moment evaluator; no molecular derivative tensor,
runtime graph, reference provider, displaced geometry or JIT is introduced.

S/T use B as their common origin, and attraction uses the independent nuclear
center C. Both motion legs shift with that origin. This makes a rigid
translation's zero tangent exact rather than a cancellation among separately
evaluated center derivatives. Distinct mathematical slots remain distinct at
coincident centers. Attraction retains the physical minus sign for a unit
positive charge. A finite candidate is published transactionally, allowing
external input/output aliases while leaving output unchanged on refusal.

## Rejected alternatives

- A second native Hermite/Coulomb recurrence would split scientific authority.
- Emitting full coordinate Hessians would retain unnecessary derivative
  components and repeat expanded expressions instead of contracting early.
- Binding affine mixed tangents to a C++ literal zero leaves avoidable Graph
  expressions alive. Actual zero leaves eliminate them before scalar emission.
- Numerical displacements and Python/libcint evaluation remain independent
  test oracles, never a production fallback.

## Invariants and boundaries

- FP64 arithmetic, existing component ordering and unnormalized primitive
  conventions remain unchanged. Strict builds disable fast-math and FP
  contraction.
- All 400 component pairs support independent A/B nuclear legs, and attraction
  includes independent C motion. Operator values, both first contractions and
  mixed contractions share one transaction.
- Neither basis contraction/normalization nor spherical projection, ionic
  charges, density/weighted-density policy or electronic relaxation belongs
  to this primitive API.
- These functions are not a complete frozen S/T/V/J/K second source, general
  ECP second source, stationary Hessian or TS endpoint. Enclosing source work
  caps, owner authentication and full scientific assembly remain required.

## Evidence

The preliminary native suite passed 50 tests without skips. It compares all
Cartesian pairs through f against independent analytic libcint primal values,
bra/ket first derivatives and AA/AB/BB Hessian blocks, recovering C only by
translation. Separate fixtures cover coincident centers, diffuse/tight and
asymmetric exponents, leg and bra/ket symmetry, bilinearity, exact common-origin
translations, malformed/overflow refusal, aliases, recovery and a linked
nonzero f-shell executable with Python absent from PATH/PYTHONPATH.

For the retained preliminary source, making affine mixed tangents actual
Graph zeros reduced generated source from 14,160,942 bytes / 316,511 lines to
13,422,447 bytes / 299,882 lines. Both forms compiled with C++20, `-O2`,
`-Wall -Wextra -Werror -fno-fast-math -ffp-contract=off` through verified
sccache 0.16.0. This is source-size/compilation evidence, not an endpoint
performance claim. Logs and subsequent qualification live under the ignored
`build/qualification/host-integral-mixed-20261112/` directory.

Final primitive/compiler/build regression passed 184 tests without skips. The
source-matched retained controls include 132 independent analytic cases and
three multistep independent-gradient differences. Maximum per-field scaled
analytic errors were `8.18e-14 / 6.82e-14 / 5.73e-14 / 7.02e-14` for
value/left/right/mixed. Ordinary separated and coincident controls use the
unscaled `atol=4e-11, rtol=3e-11` gate; tight/diffuse controls scale each field by
`max(1, max(abs(reference)))`. At steps `1e-3 / 3e-4 / 1e-4`, independent
gradient-difference errors decrease by about 100-fold for S/T/V. Compiler
structure checks report 610 modules and no dependency errors. All changed
hooks, strict compilation, source-built root target and warnings-as-errors
Sphinx generation pass. This remains component evidence only.

An additional 28-test integration cohort passes against the rebuilt core,
including the existing contracted Cartesian/spherical first-integral source
and the CMake-linked mixed executable without Python. Six selected CPU CTests
also pass. All 132 controls additionally pass an independent per-component
Gaussian L2-normalized `atol=4e-11, rtol=3e-11` gate; the largest mixed absolute
error in those units is `6.26e-12`.

Do not replace the declared extreme-exponent scale-aware gate with a claim of
uniform unscaled absolute accuracy. An exploratory unscaled gate finds one
diffuse f/f kinetic mixed component differing by `1.38e-10` at a value near
`0.0357`, with a Gaussian L2 normalization product near `1.72e10`. The raw
diagnostic is retained alongside the controls. No declared gate was changed
to hide this cancellation-sensitive case.

## Consequences and revisit conditions

The finite compilation unit adds build cost only to the optional native
response target. A complete source should consume the four fields directly
with canonical D/W/method weights and explicit complete-work limits, rather
than turning these components into molecular coordinate/AO Hessian storage.
Revisit the finite specialization when higher angular momenta or measured
complete-source costs justify a different factor schedule; preserve the same
independent oracle gates and transactional native contract.

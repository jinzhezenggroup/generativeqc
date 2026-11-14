# Decision: source-driven frozen integral curvature columns

Status: implemented (frozen-source schedule; stationary endpoint remains separate)
Date: 2026-11-12

## Problem

The accepted frozen integral Lagrangian mixed source has the correct D/W,
normalization, spin and physical-atom semantics, but using its scalar-pair API
for every coordinate pair would replay a complete ordered ERI traversal
quadratically in the number of coordinates. Its constant scalar storage did
not make that full-matrix schedule work-bounded enough for admission.

This note implements the column scheduling requirement in
[the original frozen source decision](2026-11-12-native-frozen-integral-curvature.md).
It does not supersede that source's scientific acceptance or turn it into a
stationary molecular Hessian.

## Decision

Select the explicit `Column` schedule when planning and constructing
`HostKsNuclearIntegralCurvature`. `apply_column` consumes a right atom/xyz leg
and publishes the value, right derivative, physical gradient and mixed column
transactionally. The scalar-pair schedule remains a distinct, bounded source
operation. Calling the wrong schedule refuses before source work.

Share the existing source traversal, immutable owner, primitive normalization,
canonical method weights, compensated accumulation and resource census. Every
ordered AO pair/quartet weight is evaluated once per right direction. Each
canonical ERI value/gradient/gradient-JVP is evaluated once and immediately
scattered across all physical left coordinates. A common final-slot origin
eliminates rigid right translation algebraically; coincident atom/origin
identities omit opposing scatter updates.

S/T/V reuse the existing canonical mixed primitive without another recurrence
or differentiation authority. Three relative A/B Cartesian legs suffice for
S/T; six relative A/C and B/C legs suffice for attraction. These finite slot
evaluations precede physical scatter and never replay an entire molecular
source for each physical left coordinate. They cost three joint S/T and six
attraction primitive evaluations, not one: the mandatory work plan and actual
census include all of them, even for zero D/W or rigid directions. A future
shared-factor full-slot lowering can reduce this constant without changing
the public scientific column contract.

## Invariants

- One complete source sweep per right coordinate, never per left/right pair.
- ERI primitive visits M^4 and canonical map work do not depend on left output
  coordinate count. One-electron visits are M^2 times (3 + 6*natoms).
- Owned numeric heap grows only as compensated physical-coordinate output,
  fixed weight panels and arena. Borrowed memory, AO metadata, scalar stack,
  allocator overhead/RSS and complete endpoint time remain separate limits.
- Every output field publishes together after the final authenticated owner
  check. Failure, reentry, stale D/W and borrowed-state aliases never publish.
- The frozen integral Lagrangian includes the negative W/S constraint and
  ionic Coulomb tail, not residual ECP, XC, classical terms or D/W relaxation.
- A 32-site synthetic frozen-source sweep is scheduling evidence only; it is
  not a 25--40-atom complete electronic endpoint or transition-state gate.

## Rejected alternatives

- Replaying the scalar source for each left coordinate has the exact work
  amplification this schedule must avoid, despite its small numeric storage.
- Storing AO-fourth-power or coordinate/AO Hessians retains unneeded data and
  changes the bounded resource contract.
- Reimplementing Cartesian recurrences or method weights natively would split
  scientific ownership without solving the outer source scheduling problem.
- Describing finite-slot primitive replay as a single evaluation would hide
  actual semantic work. Keep it explicitly charged until genuinely removed.

## Validation scope

`tests/python/test_host_nuclear_integrals.py` qualifies complete frozen columns
against independent normalized libcint primal energies in Cartesian/spherical
RKS/UKS through f, including ionic-tail-only Ni/I/Cs controls. Transactional
failures, resource caps, immutable owners, aliasing, physical translations,
linearity and larger-coordinate semantic work are separate executable gates.
The CMake consumer exercises both spin layouts with Python absent at runtime.

The source-matched cohort is retained under
`build/qualification/host-integral-column-20261112/`. The final related
regression passes 194 tests without skips, including 27 new column tests;
all nine CMake integration tests and six selected CPU CTests pass. Changed
hooks, the 610-module compiler structure check, strict warnings-as-errors
Sphinx and the root native rebuild pass. Native test probes use verified
sccache 0.16.0 with strict C++20/O2/Wall/Wextra/Werror, no fast math and
disabled FP contraction. An initial test-probe shell-index narrowing error
was fixed explicitly; it was not a numerical failure or relaxed gate.

Nineteen two-site primal-energy controls cover both spins and representations
through f, with Ni/I/Cs controls limited to ionic tails. Three independent
finite-difference steps are 3e-3, 1e-3 and 3e-4; final gradient/mixed acceptance
uses atol 3e-8 and rtol 2e-6, together with an explicit convergence-ratio gate.
Maximum final component errors are 4.080858806432275e-8 for the gradient and
4.863626490436701e-8 for the mixed column: the combined relative/absolute gate
passes, not an assertion that every component error is below the absolute
term alone. A three-site independent attraction-slot control also passes;
its raw frozen matrix is symmetric and agrees with the arbitrary-right
column without symmetrization.

Synthetic 3-, 8- and 32-site s-basis source sweeps verify the actual work
census at larger coordinate counts. The 32-site right-direction sweep visits
1,048,576 ERI primitives once, 199,680 finite-slot S/T/V primitives and
19,935,232 generated map terms, retaining 3,240 owned numeric heap bytes.
Its 2.9504-second wall time is a complete synthetic frozen-source measurement,
not a complete electronic Hessian or TS endpoint measurement. Borrowed D/W,
AO metadata, scalar stack, allocator overhead and RSS are outside those bytes.

## Revisit when

General residual-ECP and other stationary sources can compose these columns
with accepted S/F/D/W response, and the complete electronic TS endpoint can
be qualified. Source reuse removes one structural amplification; it does not
establish practical full-range ordered-quartic endpoint performance.

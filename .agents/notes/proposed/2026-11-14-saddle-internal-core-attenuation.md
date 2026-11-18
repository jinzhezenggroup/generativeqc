# Proposal: distinguish moving-core attenuation from occupied-root changes

Status: proposed; production root policy is unchanged
Date: 2026-11-14

## Problem

The translation registration in
`../implemented/numerics/2026-11-14-saddle-root-translation-registration.md`
removes a common laboratory-frame displacement. It does not remove attenuation
when different heavy centers move relative to each other. Its stated condition
for revisiting more general transport is now met by an actual native trajectory.

The installed NiO/H2 request with explicit maximum radius 0.2 Bohr terminated
after five accepted steps on its sixth attempted step. The occupied overlap
was 0.87076049143499257, below the unchanged 0.9 gate. It retained its last
accepted geometry, gradient and raw Hessian, but was not stationary: gradient
maximum 0.004448131323676302 Eh/Bohr and two negative curvatures. Native wall
time was 3508.408712 seconds. This is a genuine failed request, not a TS.

Four independent cold states reproduce the accepted point, rejected trial and
half/quarter trials, without any native or neighbor-state seeding. The accepted
point agrees with native energy and gradient to 7.05e-12 Eh and 7.91e-13
Eh/Bohr. The failed overlap is independently 0.8707604914350011; half and
quarter steps give 0.9640427601135783 and 0.9907051790984056. Actual physical
orbital-gradient norms remain below the predeclared 1e-11 gate.

The lost principal vector has 0.899616853 weight on occupied orbital 5
(zero-based), whose energy is -19.083951724 Eh. Independent AO-metric
Mulliken localization in `worst-orbital-locality.json` identifies it with O.
Thus merely shrinking a global radius repeatedly can constrain compact-core
motion rather than distinguish a change of the selected electronic state.

## Candidate, not a production decision

Let `S_current`, `S_previous` be the positive-definite same-geometry AO metrics
and `X` the existing translation-registered cross-geometry AO overlap. Define
`T = S_current^(-1/2) X S_previous^(-1/2)`. Its full-rank polar factor `U`
is the nearest orthogonal cross-AO map. An occupied-space comparison could use
`C_current^T S_current^(1/2) U S_previous^(1/2) C_previous`.

This retains every occupied orbital and each spin/population separately. It
changes only the comparison metric, never physical coordinates, densities,
SCFs, forces, Hessians, finite angular-grid orientation or displacement RMS.
It is nevertheless a broader policy change than common translation and must
not be installed merely because it makes this failed trial pass.

The offline independent prototype gives transported overlap 0.9830920581 for
the actual failed trial, with full-AO cross minimum singular value 0.8676868507,
AO metric condition below 172 and isometry residual below 5e-15. Half/quarter
overlaps are 0.9958593107/0.9989720015. A deliberate occupied/virtual rotation
remains rejected, with overlaps near 0.5. These are diagnostic mathematical
controls, not sufficient native or molecular qualification.

## Required before promotion

- Define conditioning, rank, finite-value and numeric-capacity failures; retain
  bounded fail-closed behavior when an AO transport is ambiguous.
- Qualify the same algorithm independently in Python and the native CLI owner,
  including RKS/UKS, one bad spin, population changes, incorrect roots, basis
  identity, rejected-trial lifetime and all resource/error paths.
- Retain/report physical cross-overlap attenuation separately from transported
  root overlap. Do not silently rename the old observable.
- Freeze the new policy and gates before a new actual molecular request, and
  independently match its final state and raw derivatives. Keep ECP saddle
  admission closed until its own complete consumer qualification.

## Current action and evidence

Production keeps the qualified translation-only policy and threshold 0.9.
A distinct request from the original NiO/H2 hypothesis explicitly reduces the
maximum radius to 0.1 Bohr, justified by the independently admissible half-step.
It does not resume a reference-optimized geometry, change the model or hide
the preceding failed work. The full #2844--#2848 goal remains incomplete.

Evidence:
`build/qualification/host-nio-h2-bounded-native-ts-20261114/`,
`build/qualification/host-nio-h2-sixth-trial-diagnostic-20261114/`, and
`build/qualification/host-nio-h2-small-step-native-ts-20261114/`.

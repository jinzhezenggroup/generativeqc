# Decision: Compare saddle roots in a translation-registered frame

Status: implemented
Date: 2026-11-14

## Problem

The actual installed NiO/H2 native saddle attempt completed its initial cold
energy/gradient and full Hessian, then correctly stopped at the first trial's
root-continuity gate. Its reported minimum occupied overlap was
0.8479754752960551, below the unchanged 0.9 gate. That failure is retained in
`build/qualification/host-nio-h2-native-ts-20261114`; it is not a successful TS.

One additional cold independent reference reproduced the rejected geometry's
overlap as 0.8479754752960567. Crucially, a pure translation of the original
reference by the Ni displacement, without any change of orbital coefficients
or internal geometry, already produced 0.8479742743157409. Compact Ni core
orbitals dominated this apparent change. The laboratory-coordinate overlap was
testing nuclear translation as well as electronic-root continuity.

## Decision

Translate only the current AO centres to the previous accepted mass centroid
before computing the native cross-geometry overlap. Translation leaves the
AO coefficients unchanged. This defines the local root comparison frame; it
does not change physical input coordinates, SCFs, derivatives, optimizer steps,
attempt counting, displacement convergence or final vibrational coordinates.
Use the same positive masses already required by the molecular saddle API.
Normalize their common scale before averaging to avoid an avoidable overflow.

Retain the smallest principal overlap independently in every occupied spin,
the exact population checks, last-accepted-state ownership, finite resource
limits and the existing default 0.9 gate. Each successful electronic evaluation
reports its comparison frame and actual registration translation. Do not
register rotations: discrete angular-grid orientation is not a gauge that this
change is authorized to remove.

The Python adapter and shared native CLI evaluator use the same convention.
The native evaluator now requires the already validated masses explicitly;
all repository callers migrate directly, without a compatibility constructor.
The real native RKS/UKS consumer regression treats a common one-eighth-Bohr translation
as an invariant, and retains an internal bond displacement as a failure control.
An earlier six-Bohr UKS control failed the existing native physical-SCF
certification before root comparison; that failed observation is retained in
`host-nio-h2-root-diagnostic-20261114/native-consumer-contract-v3.log`. No SCF
gate is relaxed, and arbitrary translated SCF convergence is not qualified by
the root-comparison change.

## Rejected alternatives

- Lowering the threshold or ignoring core orbitals would weaken the policy and
  could hide an electronic root change.
- A unitary polar transport of the entire AO space changes the comparison more
  broadly and needs separate qualification; no such transport is introduced.
- Changing the returned geometry or native optimizer step would alter physical
  state and convergence accounting to fix a comparison-coordinate problem.
- Restarting with smaller steps does not make root identity invariant to a
  physically irrelevant whole-molecule translation.

## Evidence and limits

`build/qualification/host-nio-h2-root-diagnostic-20261114` freezes the actual
failed receipt, original independent reference, basis, source hashes and new
trial before its one new cold SCF. The SCF took 9.131753 s including overlap
controls; its physical orbital-gradient norm is 6.2791e-13. The registered
trial overlap is 0.9867724034835272, and the registered pure-translation control
is 0.9999999999999453. These are diagnostic observations, not TS qualification.

Permanent regression checks use analytic compact-Gaussian translation controls
for RKS and UKS, preserve internal motion and finite-grid rotation, and exercise
the real adapter's accepted/rejected lifetime without substituting molecular
chemistry. Existing per-spin wrong-root and population failures remain required.
The installed modified consumer and any continued Ni reaction still require
their own source-bound real electronic receipts. Neither ECP saddle guard opens.

## Revisit when

Internal multi-heavy-atom motion, rather than a common translation, demonstrably
causes the same core attenuation. Any more general transport convention must
preserve independently qualified root discrimination and explicit diagnostics.

# Decision: automatically use the qualified PBE0 resident final-state proof

Status: implemented; bounded default promotion, not broad KS qualification
Date: 2026-10-11

## Problem

The [initial implementation](2026-10-10-ks-resident-final-validation.md) kept the
selector off for every owner, even though all four submission-cohort complete
PBE0 energy-plus-force comparisons passed the unchanged robust performance gate.
That conservative publication choice confused lack of universal KS evidence
with a reason not to use the already-qualified product owner.

## Decision

An unset selector automatically requests resident proof for exact-direct,
full-precision PBE0/RKS with at least 384 AOs. The exact composition is PBE,
0.75 semilocal exchange, unit correlation and 0.25 full-range exact exchange;
no fitted provider, range correction or nonlocal composition is promoted.
The native restricted Fock contracts total density, so its signed full-range
exchange coefficient is `-0.25/2 = -0.125`; it is not the positive physical
exact-exchange fraction. Production and the host probe share that matcher.
Explicit `0` remains an opt-out and `1` retains the original experimental scope.
All handle, charged-packet and published-scratch-lease checks remain independent
of selector choice. Invalid evidence still fails the shared numerical gates;
exceptions are never hidden by a retry on the reference route.

The lower AO boundary avoids extending the endpoint performance conclusion to
small owners, where the additional 112-byte packet and two drains can dominate.
The 384/768-AO endpoints expose the same fixed nine-product schedule. Larger
owners have no new schedule/planner switch or dense allocation in this adapter;
the charged packet count remains bounded. This is an algorithm-informed size
guard, not an atom-count or benchmark-geometry whitelist. It does not claim
measured performance on larger systems or other GPU architectures.

## Evidence and preserved history

The original source-matched bundle is
`benchmarks/results/ks-resident-final-validation-20261010/`. Its submission
cohort has five interleaved pairs per geometry: 48-atom warm/moved reductions
are 2.783%/2.020%, and 96-atom reductions are 6.224%/6.092%. All pass
`max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))`. All corresponding
numerical records pass independent 1e-8 Eh / 1e-7 Eh/Bohr gates. The historical
48-atom warm 1.195% below-gate result remains separate and retained.

The bundle's original `selector_default=0`, numerical publication decision and
formal performance/production `not-run` fields describe that source freeze;
they are not rewritten to fabricate a qualification of this later policy edit.
The product implementation and all shared scientific gates are unchanged.
Host policy tests execute production selection across size/spin/method/precision
boundaries, strict parser values and both explicit selectors. Host staging probes
retain both-spin/repeated-lease coverage for both selection outcomes.

## Rejected first admission probe

GPU job 7266 used source `4628029dd` and native library SHA256
`4dd35c60b04ff286baf44f5e17c4ca9c5a54fccfd46185474675c4cd3b33af9d`.
Both 384/768-AO unset-selector setup exports retained one drain and no added
packet: the first matcher incorrectly compared the native Fock coefficient to
positive `0.25`. The route assertion failed before timed samples. Both failed
records are retained; neither supplies resident timing or numerical promotion.
The corrected shared matcher checks signed restricted `-0.125`, and its host
regression explicitly rejects positive `0.25` and unrestricted `-0.25`.

## Revisit when

Independent complete endpoints justify extending automatic admission to small
owners, UKS, fitted providers or other functional compositions, or demonstrate
a genuine resource/schedule cliff requiring a more precise lower/upper guard.

# Proposal: guarded FP32 Boys tables in the shared Direct evaluator

Status: implementation staged; exact-head GPU/endpoints/performance pending
Date: 2026-10-16

## Motivation and ownership

Mixed Direct J/K still computes its Boys table in double and only then casts
each moment. The alternating series used by the strict owner cannot simply be
narrowed near its cancellation boundary. Reuse the existing positive incomplete-
gamma series and stable recurrences from the range-moment owner at boundary one.
Add a float instantiation rather than a second scientific recurrence.

Only MixedPrecisionFloat selects the float exp/sqrt/erf, series and recurrences.
Double/AD use their existing arithmetic and stopping rules. Primitive geometry,
integral contraction, density products, Fock scatter/reductions, derivative and
SCF policies are unchanged. No functional-name dispatch or new FP64 performance
optimization is introduced. Gaussian and SR/LR precision trials are excluded.

## Representability guard

An isolated unguarded candidate passes raw-moment accuracy and standard complete
endpoint gates but fails 87 of 420 additional independent scaled-Coulomb seed
tests near FP32 subnormal thresholds. A one-subnormal raw-moment error can be
amplified by the established seed factor. Preserve that failed population.

If the highest requested float moment is subnormal, retain the established
double Boys owner before the original narrowing. Normal tables stay FP32. This
primitive representability fallback is not a strict SCF recovery or another
solve. With the guard, all 420 actual full-Coulomb seed cases pass the unchanged
`2e-6` relative /one-FP32-subnormal gate against a 100-digit incomplete-gamma
oracle. Sampled strict and complete Dual3 seed jets are byte-identical.

Permanent CPU/CUDA-FMA/CUDA-no-FMA tests include every table order zero through
thirteen, branch-adjacent arguments, logarithmic extremes, invalid table bounds
and subnormal moments before seed scaling. Host results are 28 passed and 56
CUDA opt-in skips. The all-order native Direct J/K and exchange-only fixture
extends the previously reviewed independent component oracle to orders 0–12;
that test-only extension is shared with axis PR #2569, not its production schedule.

## Evidence boundaries

The historical unguarded v9 candidate at base `072ba0646` independently qualifies
complete 96-atom endpoints. Paired old/new medians are approximately 1.0260 for
PBE, 1.02471 for PBE0 and 1.066572 for WB97M-V. The WB population retains one
regression and two positive pairs, identical Fock counts but slight recurrence
work drift. These motivate new qualification; they are NOT speed claims for this
guarded branch based on `e4d00688e`. Do not relabel historical binaries or use a
cross-device timing denominator.

Before publication, build and attest the exact source/core and all official force
manifests, independently qualify moved small-system matrices and 96-atom complete
E/F endpoints, verify genuine mixed work/budget1/retry0, and measure guarded Fast
against its source-matched e4 master control on the same clean physical GPU.
Use unchanged Fast E/max-force/RMS gates `1e-5 /1e-4 /3e-5` and strict gates
`1e-7 /1e-6 /3e-6`. Keep all slower/failed observations and first preparation costs.
No Release, external backup or raw evidence bundle is authorized or published.

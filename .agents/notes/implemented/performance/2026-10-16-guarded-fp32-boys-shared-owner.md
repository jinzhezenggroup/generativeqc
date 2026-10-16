# Guarded FP32 Boys tables in the shared Direct evaluator

Status: qualified isolated speed population and exact integrated-source endpoints
Date: 2026-10-17

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

## Completed guarded qualification

The isolated guarded v16 population at e4 has all 27 complete returns: three
baseline-Fast/guarded-Fast/guarded-strict triples per method. Median baseline
Fast/guarded Fast is PBE 1.024647, PBE0 1.024906, WB97M-V 1.051289; all nine
pairs improve. Fock counts match. WB recurrence work differs by -5431 out of
approximately 20.39 billion; do not assert identical operator work. PBE/PBE0
precision work matches; their force-semantic comparison remains false because
provenance/artifact fields, streams and nested timers differ. An exhaustive
audit finds no arithmetic-counter differences; do not widen its exclusions.

Exact head `17fc57d69`, integrating master `8141cc7db`, has production identity
`3f82094939da7d22ad91e99139d8716767686f36efb7f9addae4e0c7aa6e76ee`.
The core's compiled identity and all six official force manifests authenticate.
Its Boys/range header bytes match v16. Job3283 passes 157 native tests without
skips, memcheck has zero errors, and CPU/CUDA-FMA/CUDA-no-FMA each qualify 420
scaled seeds. Job3284 passes six moved matrices and 32 small family E/F
returns. Clean RTX5090 job1083 independently qualifies all six PBE/PBE0/
WB97M-V96 strict/Fast integration returns at the unchanged gates.

Retain the additional job3284 timeout: PBE/PBE0 complete, but WB's independent
reference does not return before the 60-minute limit, so no native WB sample
starts. This is not evidence of a Fast timeout, and it is not erased or
replaced. Node2 evidence is numerical only because of its known NVML mismatch.
The separately predeclared clean integration is not v16 performance evidence.

Compact receipts: `benchmarks/results/fast-boys-fp32-20261017/summary.json`.
Raw observations, rejected alternatives and complete work comparisons remain
local. No generic FP64 optimization or functional-specific policy is added.

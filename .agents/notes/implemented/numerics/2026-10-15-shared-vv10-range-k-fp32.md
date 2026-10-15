# Decision: lower shared nonlocal pair and range-K value modules

Status: implemented
Date: 2026-10-15

## Problem

The shared Fast-Fock policy generalized J/full-range K, but the independent
VV10/rVV10 and SR/LR K value consumers remained strict. A generic XC policy
alone cannot supply missing lower-precision arithmetic in those modules.

## Decision

Extend the scalar TensorIR C++ emitter with explicit float arithmetic/parameters
and float constants; its default double emission and mathematical IR stay
unchanged. The shared nonlocal generator emits bounded FP32 closures from the
same roots, including both bilateral feature legs. This is not a functional
name dispatch or a second handwritten pair formula.

Only pair algebra narrows. Coordinates/differences, local scales, storage,
weighted chains and reductions stay FP64. The sufficient domain is r2=0 or
2^-16 <= r2 <= 2^12, and all supplied omega/kappa scales in [2^-8,2^8]. Prepared
rVV10 feature work also bounds inverse raw kappa to that interval. These
bounds keep positive denominator products and their reciprocals in normal
FP32 range. Cancellation-sensitive derivatives still require independent
observable gates; these bounds are not an observable-error guarantee. Rejected pairs use
the unchanged strict closure; no density, weight, distance or grid is clipped.

Ordered, bilateral and positive-zero-row consumers collect actual FP32/strict
pair counts. Screened negative-zero rows retain their prior semantics. The
SCF census records mixed nonlocal applications only after actual FP32 pair
work, and strict work when fallback executes. The exact ingredient cache also
keys on arithmetic, preventing a mixed result from satisfying a strict audit.

SR/LR K selects the existing MixedPrecisionFloat recurrence in both whole-shell
and bounded canonical-AO schedules. The compiler-owned Cartesian leaf and generic
contracted-ERI entry point accept ranged FP32 values through the same shared
range-moment owner, rather than rejecting them or substituting strict execution.
Radial moments remain FP64, while recurrence arithmetic follows the selected
scalar. Whole-shell cached Gaussian geometry and contraction, density products
and scatter retain their existing FP64 policy. Range value
capability requires restricted, exact, finite, positive-omega, derivative-zero
CUDA inputs; no XC identity or canonical fraction is involved. The main solver
accounts range recurrence work independently from full-range J/K, reusing dead
AUTO scalar-record storage without increasing the numeric arena.

The application resource/basis domain and strict final physical Focks remain
unchanged. Ordinary AUTO without the application budget, fitted modules,
stationary derivatives and response retain their prior precision contracts.

## Rejected alternatives

- Changing only the precision schedule: labels are not executed FP32 work.
- Casting all VV10 ingredients or reductions: local scales, tails and weighted
  chains have a separate numerical domain and need not narrow with pair algebra.
- Permitting mixed-cache reuse in strict closure: equal density is insufficient
  when the arithmetic identity differs.
- AO-quartet-only range K: retain the shared whole-shell source amortization and
  an explicit bounded AO fallback instead of multiplying primitive work.
- Calling mixed dispatch sufficient: the independent GPU matrix campaign caught
  two explicit NaN rejections in the old mixed Cartesian/contracted leaves when
  optional shell ordering was removed. Both shared leaves must support FP32;
  forcing this schedule to FP64 would not complete the requested capability.
- Claiming universal FP32 accuracy or speed: independent complete endpoints and
  larger source domains still need measured qualification.

## Validation

Independent Decimal pair/derivative checks cover both variants, live-output
demands, both bilateral legs and numeric-domain rejection. CUDA tests exercise
ordered, parallel and wide schedules, geometry outputs and strict-domain
fallback, observing actual work counts. Independent PySCF SR/LR matrix gates
cover signed and zero densities, both shared schedules, screening, changed
geometry and 24/96-AO inputs, requiring actual positive mixed recurrence work
with nonzero density in both schedules. Host compiled Cartesian tests compare
FP32 SR/LR values through d shells with independent quadrature in repeated,
diffuse and separated geometries; strict value/jet tests retain their gates.
Complete Fast endpoints retain independent moving
grid energy/force acceptance gates of 1e-5 Eh, 1e-4 Eh/bohr max and 3e-5 RMS.

Selected source-matched receipts and complete endpoint measurements are retained
in `benchmarks/results/shared-vv10-range-fp32-20261015/`. They include a slower
96-AO Fast result, not a universal speedup claim. Full raw campaign evidence,
including rejected intermediate attempts, remains outside the checkout in
`../evidence-vv10-range-fp32-20261015/`.

## Optimization parity audit

The first complete 96-AO benchmark regressed from 24.485 s strict to 29.200 s
Fast despite fewer Focks. Arithmetic support was not optimization parity:
KS excluded mixed range sources; their lease and preparation APIs admitted only
complete full-range values; the launch hardcoded the radial operator; and replay
rejected range identities. The nonlocal initializer/frozen-potential schedule
also rejected every lower-precision schedule, indirectly excluding automatic
range residency. FP32 pair emission additionally used division-heavy ordered
derivatives while FP64 reused a rational energy denominator.

Mixed sources now share full/SR/LR identities, bounded angular selection and
compensated replay. Partial coverage uses mixed direct recurrences; a strict
consumer cannot borrow the source. Independent signed/zero-density matrices
exercise both screening regimes, partial/complete residency, changed geometry,
idempotence, byte-ceiling refusal and operator/omega/mask/precision mismatch.

The common nonlocal initializer admits the application budget. Provisional
residuals cannot authorize target switching or publication; initialization still
consumes the existing mixed iteration cap. Precision handoff takes precedence
over the outer solver, clears provisional/frozen state and resumes fresh FP64
target evaluation. FP32 pair roots reuse the shared rational/unit-reciprocal
closure under the existing representable-input domain, with a new lowering
identity and independent Decimal checks at every domain corner.

Hermite reassociation and canonical fused-RSH experiments also exclude mixed
arithmetic, but both are opt-in and disabled in both measured default endpoints.
They are not evidence for this default regression; enabling those experiments
requires their own numerical/resource qualification, not blindly deleting gates.
Retain the slower baseline, then qualify the default reuse changes with new
source-matched complete endpoints before claiming a speedup.

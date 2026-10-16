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

Numerical-failure retries obey the same handoff: clear seed/frozen/outer state
and reset the exact-target epoch before preparing the strict source. Changing
only the precision flag would leave a seed retry without NLC, or reuse a mixed
frozen potential in the alleged strict retry. Host failure injection executes
the production retry and next device-XC dispatch for all mixed components,
both provisional stages and absent/mixed/strict leases. It requires fresh strict
pair work at the unchanged density, reset history/energy epochs and truthful
retry accounting; GPU observable acceptance remains a separate requirement.

The first composed 96-AO run passed observable gates but spent all twelve mixed
iterations in the semilocal initializer: its positive-FP32-nonlocal-work assertion
correctly failed. Do not weaken that check or report the missing work as mixed.
Reserve the last existing mixed slot for a complete target evaluation. This
uses actual FP32 pair work before handoff without increasing the cap, relaxing
final tolerances or accepting a provisional operator.

Hermite reassociation and canonical fused-RSH experiments also exclude mixed
arithmetic, but both are opt-in and disabled in both measured default endpoints.
They are not evidence for this default regression; enabling those experiments
requires their own numerical/resource qualification, not blindly deleting gates.
Retain the slower baseline, then qualify the default reuse changes with new
source-matched complete endpoints before claiming a speedup.

## Follow-up qualification

Implementation `ed4eff117` / core SHA256
`1a4b2f0bbd4bc6c69ca3535bde9ea91fc4fdfb9aa93c12c84d160d2cdc5e4147`
passes 832 host tests (406 resource skips), 354 generated-pair GPU tests and
19 native integration tests. Pair memcheck covers 24 mixed cases; four
changed/unmoved 24/96-AO streaming/resident range probes have zero memcheck
errors and maximum independent matrix error 2.057e-7.

The combined six-endpoint memcheck is **not clean**: all numerical assertions
pass, but two optional recurrence-stack reservations report OOM through
cudaDeviceSetLimit and cudaGetLastError (four API reports; sanitizer exit 99).
This is the retained optional fallback in direct_coulomb.cpp, not an invalid
read/write report. Separate fresh-process complete PBE0 and WB97M-V reruns
both pass with zero sanitizer errors. Keep the combined failure instead of
reporting those reruns as six clean tests or changing unrelated stack policy.

Three alternating steady complete-endpoint medians (FP64/Fast, seconds),
with first calls retained separately and the unchanged 48/16/32 grid:

- 3 atoms / 24 AOs: 6.783615 / 4.987146, 1.360220x.
- 12 atoms / 96 AOs: 24.434694 / 18.176775, 1.344281x.

The 96-AO maximum independent errors are 6.385e-10 Eh energy,
3.939e-6 Eh/bohr maximum force and 2.285e-6 force RMS. It executes one solve,
twelve mixed and four strict Focks, zero retries and five nonlocal sweeps.
Single-call ablations are not paired speedups: default 19.566 s/five sweeps,
range residency disabled 19.600 s/five sweeps, nonlocal initialization disabled
29.001 s/twelve sweeps. Initialization is the dominant recovered optimization
in this fixture; no isolated range-cache timing benefit is established.

The independent 96-atom/768-AO reference exceeded its finite 25-minute job.
There is no measured native 96-atom speedup and these small-system results
must not be extrapolated to one. PR #2515 was already merged on October 15,
2026 at 18:58 Shanghai time; this follow-up is not part of that merged PR.

## Clarification: 96-atom timeout versus README

Slurm job 8634 exhausted its 25-minute allocation while running only the
independent GPU4PySCF WB97M-V reference. Neither native FP64 nor native Fast
was executed by that command. The output JSON is written after both SCF and
moving-grid forces return; the interrupted log contains no iteration or phase
telemetry. It cannot identify an individual slow kernel, an SCF convergence
failure, or a native Fast regression.

The README is consistent with this outcome. Its direct timing table reports
native FP64 cold 1336.273 s and fixed-density warm 59.897 s at 96 atoms / 768
AOs, but explicitly marks the GPU4PySCF reference as `timeout`. The historical
reference job 7965 reached its 3600-second Slurm limit in `cold/0` without a
complete record. Separate accuracy-only independent geometry receipts in
`benchmarks/results/readme-wb97mv-20261014/summary.json.xz` took 4617.113 s and
4578.170 s. Those diagnostic intervals are not published reference cold/warm
timings and must not be used to manufacture a speedup.

A host-side input comparison confirms identical original geometry, spherical
H/O def2-SVP basis elements (subset identity
`fad2c7433fe5fac19525d4c80fbd995a7ebed7816b7ea4bb88547e4f6a58a3d8`),
and the complete 48/16/32 unpruned moving grid: 2,359,296 points on each of
the semilocal and VV10 grids. The new reference uses a one-electron guess,
four CPU threads and 180 maximum cycles; the historical accuracy receipts
use the reference's default atomic guess, eight threads and 100 maximum
cycles. These are not identical timing protocols even though their physical
target and stopping tolerances agree.

The timed-out reference is WB97M-V, not PBE0. In the same README table,
PBE0/96 has native cold 69.710 s and warm 13.259 s. Those different-functional
observations cannot be substituted for WB97M-V's 96-atom timings.

The 25-minute reference allowance was therefore poorly chosen relative to
already retained diagnostic evidence. This is not a new 96-atom Fast timeout.
Native cold versus native fixed-density warm, direct versus DF, and 96 AOs
versus 96 atoms must remain separate comparisons. A future 96-atom Fast
qualification may reuse the retained independent accuracy receipt after
explicit input verification rather than rerun an inadequately budgeted oracle.

## Integration with the subsequent defaults

The follow-up branch starts at master `4e28477e9`, after the guarded DF
warm-factor and VV10 warp-packet defaults landed. Preserve both defaults.
The packet traversal now uses the same bounded FP32 bilateral closure and
strict fallback as the ordered and previous parallel traversals; its
arithmetic counters count evaluations, not rotated partials or publications.
FP64 packet coordinates, shuffle payloads, chains and accumulation remain
unchanged. Retain partial-warp cases added by the default-promotion work.
The schedule identity advances to `vv10-parallel-bilateral-warp-packet-mixed-v5`.
Historical endpoint timings above retain their original source identities;
they are not measurements of this integration revision.

Integrated host suites pass 1032 tests with 42 resource skips. The independent
generated-runtime GPU replay passes 784 tests across ordered, previous
parallel, wide and packet traversals. Its 32 mixed cases also pass memcheck
with zero errors, including actual arithmetic census and strict fallback.

The integrated Release core has source identity
`99dfe2fbe094816743ee13c9eab9dae1ca2d182667694a1aa9677f4cff073b92`
and SHA256 `b89a605b87de5cdf959fc2f0fe2d208d95f155937021702005f31d429bb754d6`.
All twelve complete Fast endpoint cases and the full-range mixed source
matrix case pass. Six range matrix fixtures pass in a fresh process, retaining
both whole-shell and canonical-AO schedules, mixed residency and moved inputs.
The earlier combined run is retained: fifteen pass, but four mixed range
fixtures fail their whole-shell admission assertion before numerical checks.
Do not report that combined run as nineteen passes or infer a particular
allocation failure without telemetry. The fresh replay uses the identical
core and unchanged admission assertions, not a weakened test or fallback-only
substitute. Fresh-process complete PBE0 and WB97M-V endpoint memchecks each
pass with zero errors.

Three host fault-injection fixtures needed the extended shared provider API:
mixed range/nonlocal state reset, strict range-provider dispatch with explicit
mixed/counter arguments, and mixed source operator/mask defaults. Their 95
cases pass after updating the doubles. Strict profiling still rejects mixed
counter use, and existing event/error/retry assertions remain in place.

## Post-review strict retry qualification

Preserve remote review fix `5f0fc6606`: numerical-failure recovery must leave
the provisional nonlocal initializer/frozen operator before its fresh strict
evaluation, just like the normal precision handoff. Its executable host
failure-injection coverage and authenticated live benchmark runner are retained,
not overwritten by the provider-fixture updates.

The rebuilt core at code revision `74a852c26` has source identity
`0ba780ead15c9ca55aeb438c5f858aaedfcd64260666a35ad7dda795be203669`
and SHA256 `6b263637770ab3be53cf5b8ed31080a8da1605afe2587e2c2bc3a7bb72fabded`.
The complete host selection passes 1144 tests with 42 resource skips, including
the strict retry and live-runner authentication cases. Slurm job 8778 uses
this new core and matching force artifacts: twelve complete endpoint cases
pass, followed by seven matrix cases in a separate fresh Python process.
Both whole-shell and canonical-AO range paths keep their original assertions.
No new paired timing or 96-atom Fast speedup is inferred from test durations.

The 784 pair-runtime cases and 32 clean mixed memchecks exercise unchanged
generated/kernel sources. The complete-endpoint memchecks recorded in the
previous section belong to the pre-retry-fix core `b89a605b...`, not this new
core; do not relabel those observations. All raw logs, XMLs, source freezes
and both earlier combined failures remain in the external evidence directory.

The [2026-10-16 handoff follow-up](../performance/2026-10-16-nonlocal-strict-handoff-reuse.md)
retains the already-admitted outer solver across a normal precision handoff,
after discarding and rebuilding its mixed anchor. Numerical-failure recovery
still clears the provisional solver entirely; the retry fix above is retained.

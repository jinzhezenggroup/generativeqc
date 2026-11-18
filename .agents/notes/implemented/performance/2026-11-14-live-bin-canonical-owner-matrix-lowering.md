# Decision: initialize live ERI bins and specialize canonical owner labels

Status: implemented in the compiler; complete molecular qualification remains open
Date: 2026-11-14

## Problem

The fresh physical I–CH3–I RKS total carbon-z HVP spent 469.355001782 s
preparing its frozen integral matrix, including 456.920 s in the ERI phase.
Its actual frozen work was 136,320,861 primitives. Further small changes to
moving-grid XC did not address this dominant preparation cost.

The existing physical-owner generator already shares scalar factors, reduces
owner dimensions, and respects Cartesian support. Do not introduce another
derivative algebra or infer common physical ownership from equal coordinates.

## Decision

Retain the existing scalar Graph/AD, ordered products and reduction order, but
expose two execution facts to the ordinary portable C++ compiler:

1. Every live pair coefficient jet has every value/gradient/ordered-matrix field
   assigned by its producer, including structural zeros. Its unused tail need
   not be initialized. Only coefficient reduction bins need initial zeros, and
   only bins through the exact per-axis order can be consumed. Zero those live
   bins before reducing into them. Other Jet accumulators retain explicit value
   initialization. Removing aggregate default member initializers must never
   turn a reduction accumulator into uninitialized storage.
2. Four mathematical slots, with the fourth fixed to the origin and non-origin
   labels assigned by first occurrence, have fourteen nontrivial partitions.
   Dispatch the already validated label encoding to template constants. Share
   the same pair, Coulomb, prefactor and product algebra across these instances.

The maximum s–f axis order is twelve, within the existing thirteen coefficient
bins. The producer assigns pair orders through `left_power + right_power`;
the reduction and consumer visit only their declared live ranges. Neither
transposes nor numerical-zero guesses replace any ordered mixed partial.

Public pointer, component, exponent, canonical-label, same-owner coordinate and
finite-result checks remain. Distinct coincident owners remain distinct. No
public ABI, numerical precision, CPU ISA requirement, allocation bound, work
cap, response tolerance or fallback policy changes.

## Evidence

Private experiments use the immutable baseline core with source identity
`07282cde840d1a731549f8b59c25ca956b8c3f560aa1fb740c919cc5856fbac7`.
All primitive status/output comparisons include all 91 buffer doubles, including
unwritten sentinels. The interrupted historical prefix is reauthenticated
against that core before use; it is diagnostic input, not a derivative oracle.

- Incidence-only, live-initialization-only and combined candidates each pass
  21,532 valid bitwise ABI checks and 460 explicit refusal controls, covering all
  fourteen owner partitions, 256 ordered s–f angular tuples, two exponent
  patterns, component selections, coincident centers and invalid bindings.
- The combined private binary passes 534 existing independent scientific tests,
  including analytic libcint ordered Hessians and multistep primitive-gradient
  references. Existing signed-zero, near/coincident, translated, diffuse/tight,
  overflow and exact output-size controls are retained.
- A fresh composed physical carbon-z HVP recomputes first sources, the frozen
  matrix and coupled response. It does not inject saved response or integral
  panels. Its actual complete request takes **691.805405203 s**, within its
  original 900 s cap. First sources take **194.133953082 s**, and the frozen
  matrix takes **425.577642039 s**. Separate earlier frozen preparation took
  469.355001782 s: a diagnostic 9.327% reduction, not matched-host or whole-
  endpoint speedup evidence.
- All 38 composed fields match the earlier accepted HVP bitwise. Independent
  fourth-order HVP error remains **3.495323606939049e-9**, and the physical
  response residual remains **8.406008203576136e-14**. Both original gates pass.
- An explicit private override receipt confirms actual matrix primitive calls
  by one/two/three owners: **58,747,446 / 61,158,798 / 10,690,785**, with no
  refusals. Native complete work and resource plans match the earlier request.
  Peak process RSS is **534,664 KiB**, distinct from scoped numeric reservations.

The production emission is compared once, opt-in, with the scientifically tested
private source after removing private symbol names. This migration comparison
is not a permanent required-CI source fingerprint. Production tests compile the
actual emitter and exercise observable results and output bounds; an additional
all-angular-support sweep covers every partition without pinning source spelling.

Actual production-emitter validation passes **548** primitive/scientific/support
tests, plus the isolated build-only generation test. A further **73** native
consumer tests pass, including projected/public-basis and independent primal
comparisons, RKS/UKS density channels, mixed shells/owners, primitive admission,
unsafe-scaling fallback, aliasing, ownership invalidation and recovery. The
compiler structure audit reports 612 modules with no dependency errors.

The rebuilt production core has source identity
`2bbcc04c3142115739d58db9389b8d4d4ae2881eedb206237ccb4f3b64e61fd9`
and SHA256 `0ab7928b126aa821568ecdb45ac9e6e6cf43278c3107331693017bd839ac8263`.
Its 118,066,960 bytes are recoverable bitwise using a retained zstd patch against
the exact earlier core. The active library lives in tmpfs because the filesystem
is nearly full; a verified durable patch, not a tmpfs pathname alone, is required
before treating that artifact as retained.

The private execution is explicitly a baseline core plus a separately hashed
override. It must not be described as execution of a rebuilt or installed new
production core. New native artifacts need their own current source binding.

## Rejected alternatives and limits

Three earlier moving-XC private experiments reduced complete leaf medians by
only 2.651% (eight rows), 2.426% (four rows/eight columns), and 0.580% (exact-zero
dual-adjoint reverse skipping). All remain unpromoted. Zero primal partitions
must not be skipped: their derivatives can be nonzero.

An unchanged-DAG private AVX2 target reduced equally weighted actual-prefix
primitive timings by 5.043%, with larger synthetic-owner improvements. This
does not justify global ISA flags or an AVX2-only production artifact. Such a
path would require portable runtime dispatch and its own endpoint qualification.

The combined portable microexperiment improved its equally weighted prefix by
23.658%, but actual full frozen preparation improved only about 9.327% against
the separate earlier request. The prefix is neither visit-weighted nor a
representative complete workload. Do not project microbenchmark ratios onto
the molecular endpoint or add independent candidate percentages together.

The prior full-matrix planning estimate was 1871.264713377 s, above the original
1800 s cap. Even subtracting the separately observed frozen preparation saving
leaves an explicitly unqualified estimate above that cap. This work does not
authorize an unchanged full-matrix retry, raising a cap, splicing columns,
accepting the old failed raw reference, or opening either public ECP saddle
guard. Complete raw Hessians/modes, installed endpoints, mixed Ni/I/Cs and
representative native Ni saddle qualification remain separate requirements.

## Changed-source complete request: failed, not a matrix qualification

One actual full request uses the rebuilt production core without the private
override, under the original 1800 s cap and strict relative0 / absolute1e-12
response controls. It times out with exit124 after **17/18** completed columns.
Actual wall time is **1800.03 s**, last completed-column publication
**1758.202759794 s**, first-source preparation **192.422431961 s**, and frozen
matrix preparation **431.281009335 s**. Peak RSS is **537,340 KiB**.

All seventeen completed response residuals match the accepted response sweep,
with maximum **6.067583283002547e-13**. Against the existing independent fourth-
order reference, their maximum column error is **1.0839430530235061e-8**. The
completed carbon-z fields match the earlier accepted total HVP bitwise. These
are prefix observations only: coordinate17 is missing, no complete raw matrix
is published, and no full-matrix symmetry/mode classification is accepted.

The negative receipt and the interrupted exact-residual event are retained in
`host-live-incidence-full-matrix-20261114`. No column splicing, symmetry repair,
cap increase or unchanged retry is allowed to turn it into a successful endpoint.
The separate estimate's warning was real; further material work reduction and
scientific qualification are needed before another complete-request attempt.

## Retention

`build/qualification/host-frozen-live-incidence-candidate-20261114/`
contains the private execution, validators, controls and source provenance.
Its read-only continuation delta is **30,965,995 bytes / 411 members**, with all
member hashes verified, SHA256:

`7fdd0df32b075bdaeaedf034dcb0696d824157cf499ffd1daf0e6bd02987427d`

The delta is explicitly **not self-contained**. It requires the retained
`host-current-frozen-total-column-20261114/host-all-coordinate-response-and-total-column.tar.gz`,
SHA256 `5ca49c15cabae5e4305a79bb152c4fa50bd02b554a13ca68b32477f10185ddca`.
The 335 MB parent is not copied into the delta. Keep both archives. Earlier
compile-only failures and unpromoted experiments remain distinguishable from
successful scientific execution.

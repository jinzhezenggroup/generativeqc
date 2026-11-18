# Proposal: reuse packaged value/gradient algebra for native coordinate first sources

Status: proposed
Date: 2026-11-13

## Measured problem

The last complete attempt spent315.834755903 seconds preparing coordinate
first sources. Do not rerun that expensive unchanged endpoint or extrapolate
from faster stationary gradients: it is a different consumer.

A new bounded probe calls the actual `HostKsNuclearIntegrals::prepare_coordinates`
in CoordinatePanel/SymmetryUnique mode, on the exact original six-atom/81-AO
spherical basis and native saved C->D, RKS with exact exchange0.2. It uses the
actual linked method identity and immutable-owner validation, not a dummy
callback. Full-source admission remains conservative:2,216,680 owned numeric
bytes,297,052 one-electron primitives,237,738,954 ERI primitives and688,747,536
map terms. The ERI bound fits the original explicit1e9 qualification cap.
ECP residuals, XC, D3, response and the full Hessian are not this component.

Transparent interposition stops after exactly8,000,000 successful actual
factorized primitive calls. The native counter correctly reports8,000,001
attempts because it increments before the controlled interruption. No coordinate
panel is published. The complete interrupted preparation takes10.443103321
seconds; actual primitive leaf scopes take8.373145949 seconds (about80%).
It also executes297,052 one-electron primitives,28,430,784 map terms and
3,483,414 projection lanes. This prefix visits78 angular/type classes, not all
Cartesian components and not the entire planned source. No SCF runs.

The probe binds source
`b3a5ae19e3bd83c41f382c988d097fe3a9913798018fe7f502002d9aa4a8856a`,
core SHA256 `057ac20ee7fc8dddfb6b123555031e23867ad943b09a56606d3163e369be3326`.
Its raw receipts belong to `build/qualification/host-first-source-prefix-20261113`.
The saved native producer remains the original aefd4636 source; this does not
invent a newly converged state or establish stationarity/full endpoint success.

## Candidate to test, not yet production behavior

The first-source owner calls the shared generic factorized scalar value/gradient
primitive. The packaged stationary s/p/d emitters already produce the same
mathematical ERI value and all center gradients in their internal `Gradient`
object, but their exported gradient-only dispatcher publishes only12 fields.
Test a private CPU value/gradient adapter over that existing compiler-owned
weighted DAG, publishing13 fields (value plus12 derivatives), with exact
component canonicalization and slot/axis restoration. Do not reconstruct a
value from derivatives or author another ERI/AD implementation.

The existing gradient-only consumer has a different useful contract; a distinct
value/gradient entry would not be a compatibility alias. Do not silently change
its output width, ABI or all callers merely to expose one additional consumer.
Any production proposal must preserve the original generic route for unsupported
classes, including through-f behavior, and refuse numerical failures rather than
masking them with a scientific fallback. A private prototype must first show a
win on the actual source consumer, not just isolated primitive calls.

Validate every supported Cartesian class against independent libcint at
separated, coincident and diffuse/tight nontrivial scales, plus the actual saved
records. Preserve positive-Full bounded evaluation, normalization,12 independent
slots, scalar value, finite/domain/transactionality gates, conservative source
budgets and semantic record counts. Preserve the complete source-map/scatter
and projected/frozen/response numerical qualification. None of those complete
endpoints is qualified by the current prefix.

## Independent-oracle caveat discovered by this probe

The first two actual records captured in each angular/type class are unusually
tight/distant. All156 sampled field norms are at most3.097661016724561e-15;
some are genuinely zero or subnormal. A fixed3e-11 absolute gate alone would
accept a bogus zero implementation for these samples. Add norm-scaled checks
instead of treating the original absolute-only pass as strong relative evidence.

Default libcint screening returns zeros for representable native fields around
2.7e-239. An initial stricter gate therefore fails for the independent oracle's
screening, not because the native result has been proven wrong. The retained
retry changes only the test oracle's exponential cutoff to1000 using PySCF's
`PTR_EXPCUTOFF` field, whose units are confirmed by its `with_integral_screen`
implementation. Production screening, physics and resource gates do not change.

With screening removed, all156 raw value/gradient cases pass both the original
3e-11 absolute/relative gate and the additional scale-aware gate. The latter
uses3e-11 times the oracle field norm plus relative3e-11, with an explicit
64-smallest-subnormal-ULP absolute floor (3.16e-322). Maximum absolute error is
7.454735554338562e-29. Maximum reported norm-scaled error is1.2347020074022917e-10
in the underflow regime, where the explicit ULP floor, not impossible relative
accuracy, is authoritative. Retain the failed screened validator and both raw
attempts. This special treatment is test-only and does not weaken normal-scale
scientific gates. Broad nontrivial-scale oracle coverage remains mandatory for
any proposed production adapter.

## Subsequent consumer-use audit: prefer the existing gradient-only contract

The archived proposal snapshot predates this use audit. In
`HostKsNuclearIntegrals::Impl::apply`, the returned scalar at first[0] is never
consumed: only the twelve gradient fields feed CoordinatePanel preparation and
directional projection. Prefer a private prototype of a gradient-only packaged
adapter for this consumer before adding any13-field ABI or regenerating shards.
The existing packaged s/p/d dispatchers already expose exactly that contract.
Extract/share canonical component and slot restoration from its current native
owner instead of duplicating it. Keep public-AO/radial scale in the caller.

The similarly spelled first-only path in `HostKsNuclearIntegralCurvature` is
different: it really consumes the scalar value for stationary/moving-weight
terms. Do not globally replace or fake `full_range_primitive_first_cpu` outputs
with a zero scalar, even in an interposed performance prototype. Keep that
value/gradient consumer unchanged unless separately qualified. The older
13-field proposal above is relevant only if that distinct consumer needs it.

For unsupported angular classes or an unavailable packaged artifact, retain the
existing factorized first source, including its through-f operation and work
counts. Do not substitute the other three-pass generic gradient recurrence for
that fallback or hide numerical failures with fallback. Capability fallback
and failed finite/domain evaluation are different cases.

## Superseding implementation

The gradient-only candidate is independently qualified and integrated without
a thirteen-field packaged ABI. See
[the implementation decision](../implemented/performance/2026-11-13-gradient-only-packaged-first-source.md)
for shared native ownership, common-origin numerical qualification, distinct
capability/error outcomes and the unchanged curvature/fallback boundaries.
Production qualification remains separate from the sealed private evidence.

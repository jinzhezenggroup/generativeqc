# Decision: packaged gradients for native nuclear first sources

Status: implemented; component-qualified, complete-endpoint qualification pending
Date: 2026-11-13

## Decision and ownership

`HostKsNuclearIntegrals` consumes twelve ERI center partials, never a scalar
integral. It now reuses existing generated s/p/d packaged artifacts. Native
canonical component/axis restoration is extracted from `s_integrals.cpp` and
shared with stationary shell contractions, not duplicated in DFT. The adapter
accepts four exponents plus twelve coordinates and returns unscaled gradients;
normalization, spherical expansion and radial weights stay in the host loop.
The original factorized first operation also ignores trailing record scale:
a contrary private test assumption was corrected, not made production behavior.

Missing artifacts or unsupported components explicitly request the original
factorized first source. This includes f, preserving its one first evaluation
and work counts, not the stationary generic f API's three directional passes.
Invalid-domain or nonfinite evaluations propagate; they are not capability
misses. Output publication remains transactional. No generated shard ABI,
new scientific recurrence, numerical cache, resource cap or optional storage
precondition changes. The stationary route retains its original weighting,
zero-primal scalar AD arithmetic, geometry and through-f implementation.

The distinct curvature consumer really needs the scalar value and keeps its
original thirteen-field factorized operation. Never fake its primal or replace
the shared first symbol globally with gradient-only output.

## Small separations and independent oracle conditioning

Only the new gradient-only adapter translates coordinates by mathematical slot D
before the packaged binder. All four independent partials survive until physical
owner projection; translation is not a reason to discard a derivative slot.

A deliberately strict near-coincident/nonzero-origin control initially fails:
both old native factorized geometry and raw-coordinate libcint lose digits in
product-center subtraction, with errors around 2e-16 to 3e-16 for fields of
order 1e-6. Reorigining only the candidate leaves oracle error. Independent
75-digit, test-only Hermite recurrence checks of both diagnosed cases find
candidate errors at most 5.93e-21 and reorigined libcint errors at most 6.78e-21.
The independent oracle therefore also uses a common origin, preserving physical
separations and every tolerance. All failed attempts are retained. No gate is
relaxed to admit the candidate.

## Private evidence before production integration

Sealed cohort `build/qualification/host-first-gradient-candidate-20261113`, archive
SHA256 `9d65d15ea1aacd71bb24012668382ff63bc7f467dceb2835fee79bee604a07bd`:

- Baseline source `b3a5ae19e3bd83c41f382c988d097fe3a9913798018fe7f502002d9aa4a8856a`,
  core `057ac20ee7fc8dddfb6b123555031e23867ad943b09a56606d3163e369be3326`.
  Private adapter/overridden consumer have separate hashes, not that identity.
- All 100,000 independent libcint gradient checks pass 3e-11 abs/rel plus the
  additional 3e-11 norm-scaled gate with 64-min-subnormal-ULP floor. They cover
  every s/p/d Cartesian component, positive/signed external weights, separated,
  coincident, diffuse-tight, near-coincident and translated inputs. 70,615 field
  norms exceed 1e-5; max absolute error 2.132e-14, norm-scaled 5.028e-13.
- 525 through-f private fallback cases match original first outputs bitwise;
  fifteen domain/refusal controls leave outputs unchanged.
- Three interleaved actual first-preparation prefix pairs use original native
  saved density, six atoms/81 AOs, unchanged Hamiltonian/strategy/admission and
  genuine immutable validators. Eight million successful records, eight million
  plus one native attempts, identical histogram/map work. Median interrupted
  preparation: 10.446125226 to 5.996013474 s; leaves: 8.339714797 to 3.923465950 s.
  All 156 actual tiny/subnormal samples pass independent unscreened, scale-aware
  checks. No coordinate panel, SCF, response or Hessian is published.

Production integration needs its own matched build, complete-consumer tests,
stationary shell parity and measurements; private evidence is not relabeled.
Both public ECP saddle guards remain closed. Complete raw H/response/modes,
mixed Ni/I/Cs, stationary Ni/Cs, practical 25–40 atoms and native Ni TS under
all five criteria within 100 attempts are still outstanding.

## Production qualification, November 14, 2026

The separate production cohort is
`build/qualification/host-first-gradient-source-20261113`; its `retention.json`
binds the sealed archive. Source identity is
`8ce495a15ae52d1b293013f577a24314373b3d2ceced384c77df67275e0530a6`,
core SHA256
`e5a0e54408d57ff5a119b724819cd3e7136e4ed6545bcb45f0aa0528f7b3cb61`.
Only four production owners differ from b3a5: the shared integral source/header,
the first-source caller, and the factorized primitive header's accurate input
documentation. Neither generated primitives nor the curvature route change.

All 1,952 Python tests pass, including the complete host-integral module and
every supported Cartesian component under the independent fixed and scaled
gates. All six native CTest groups, six documentation tests plus six subtests,
warnings-as-errors Sphinx and scoped formatting pass: 1,964 distinct groups.
Thirty-six complete stationary shell contractions preserve both packaged and
reference output bits and semantic work; maximum independent recurrence error
is 2.665e-15 under the unchanged 2e-10 abs/rel gate.

Focused ASan/UBSan instruments the actual complete `s_integrals.cpp` and
`host_nuclear_integrals.cpp`, not copied implementations. It covers 10,000
primitive components with exactly sixteen input doubles, six complete s/d/f
Cartesian/spherical first-source systems, and native MP2 contracts. Leak
detection and halt-on-error pass. Other shared-core/generated dependencies are
not instrumented. Actual-consumer backend injection checks twelve complete
normal/missing-artifact source pairs and twelve transactional domain failures;
the latter forbid any factorized fallback.

Three interleaved, equal-work actual first-source prefix pairs complete eight
million records apiece, retain eight million plus one native attempts and the
same 78-class histogram/map work, and publish no coordinate panel. Median
interrupted prepare is 10.493868975 s baseline versus 6.240395064 s production
(40.5% less time); leaf scopes are 8.428614242 versus 4.205490420 s. All 156
actual tiny/subnormal oracle cases pass, and production sample gradients match
the qualified private candidate bitwise. These remain bounded component
measurements, not a complete first preparation or an endpoint speedup claim.

The shared helper's stationary regression also uses three interleaved,
equal-work eight-million-record pairs across 263 kinds. Normal value preparation
medians are 4.378970241 versus 4.347618164 s and interrupted stationary apply
5.706714477 versus 5.729804081 s: effectively unchanged, not a claimed speedup.
These probes deliberately use zero W, publish no whole gradient, and do not
replace complete endpoint qualification. Failed compile/link/test assumptions
from the private stage and the corrected sanitizer command lookup are retained.

Revisit a packaged value-plus-gradient ABI only for a measured consumer that
really needs its scalar, not this gradient-only source.

## References

- [Original proposal and consumer-use correction](../../proposed/2026-11-13-packaged-value-first-source.md)
- [Current contract](../../../../docs/developer/dft_hessian.md)

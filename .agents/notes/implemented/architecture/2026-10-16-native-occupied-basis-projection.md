# Decision: one native occupied-basis transport owner

Status: implemented
Date: 2026-10-16

## Problem

The progressive HF consumer reconstructed a pure occupied source from its density,
validated metrics and projected with NumPy. Native
`initial_guess::project_occupied_density` already implemented occupied transport.
Translating the projection again would create a second native scientific owner,
not solve #2227 R4/R6. A provisional duplicate was therefore discarded.

## Decision

The existing transport serves both contracts. Its original entry retains the
historical relative Gram cutoff, roundoff residual snapping and acceptance.
Explicit controls select the public basis policy's absolute Gram cutoff,
validation tolerance and unsnapped residual. Optional normal-equation diagnostics
use its actual unorthogonalized image. Typed scientific refusals let the adapter
present public diagnostics without matching internal error strings or duplicating
numerical gates.

The new adapter validates PSD spectra and discarded source components,
reconstructs integer-occupied pure HF densities, and publishes detached
coefficients/densities with invariant diagnostics. It pads rectangular source
columns only at the historical square-frame boundary. Coefficient gauges may
differ; compare density and metric invariants instead of signs/rotations.
Scaled A-transpose-A provides the original spectral-norm symmetry gate without
squaring the physical input scale. There is no new eigensolver: the new route
explicitly supplies the ordinary shared CPU eigen operation, not a hidden
reference-oracle computation. Legacy defaults stay distinct and unchanged.

An installed versioned C ABI provides metadata-only capacity and execution.
Numeric admission precedes numerical-array reads. All output records/arrays stay
unchanged on refusal, aliases are rejected, and unused buffer tails stay intact.
The conservative native capacity includes owner-visible numeric state/results
and ordinary eigen scratch, not caller arrays, metadata, allocator or retained
provider overhead. This is not complete physical allocation accounting or RSS.

Plain Python policies use that owner. Python keeps shape/finite conversion and
immutable dataclasses. Missing additive protocol, custom policy/scalar semantics
and unbounded integers remain explicit compatibility. Selected-library loader,
ABI and execution errors never retry Python projection science. Library bindings
are cached only by immutable resolved path; per-call CPU contexts keep error
detail and cleanup independent rather than sharing mutable failure state.

## Invariants

- Projection is only an initial proposal, never target SCF convergence/model
  authentication; source/target solve placement is independent of this CPU setup.
- Empty spin channels remain valid even when an inclusive cutoff discards every
  metric mode. Missing occupied orbitals are never invented.
- Public and historical native cutoff/residual conventions must not be silently
  interchanged. Nonfinite contractions reject rather than disappearing in max.
- No new SCF, force, least-squares projection or eigensolver equation owner.

## Evidence and remaining qualification

Hand-derived ambient projection/rotation, pure-density reconstruction, metric
rank, empty spin, inclusive/one-ULP gates, scales1e-200/1e200, detached output and
refusal tests exercise the private core. The legacy overflow/finite fixture now
compiles real translation units through the cached compiler fixture instead of
extracting production text or injecting a fake eigen oracle.

The provisional compiled core passed200 seeded ambient Euclidean-lstsq fixtures
in both input modes (400 projections), max density difference
3.197442310920451e-14; scoped core sanitizer/legacy regression checks passed.
The installed-protocol development preview passed76 targeted Python tests,
including independent ambient least squares, actual progressive HF reference
E/F, native-only science, failure cleanup, old/custom/unbounded compatibility,
loader/ABI failure and exact/below-limit admission. Those previews link newly
compiled sources to the earlier installed scalar scientific library: they are
not final exact-head installed-library or endpoint qualification.

The new full native build, frozen source/install SDK, broad independent numerical
suite, original/moved complete endpoint medians/work and final sanitizer checks
remain required before publishing final qualification. Physical progressive
stage/strict-cleanup execution and #2227 R1-R6 are not completed by this slice.

The development full scalar library subsequently passed the new installed SDK
cases, including descriptor-extension aliases, and227 Python tests/22 subtests
covering progressive projection, resource admission and policy/response
regressions. This precedes the exact committed-head rebuild and is not a
replacement for that qualification. The revision endpoint runner deliberately
includes calculator construction, both prepared lifetimes, detached copies and
cleanup, preserves first use and every measured sample, and rejects a mismatched
archived Python/native source identity. Native numeric subplan declarations and
timings are retained separately from physical work and model identity.

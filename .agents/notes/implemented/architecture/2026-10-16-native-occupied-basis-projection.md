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
cases, including descriptor-extension aliases, and 227 Python tests/22 subtests
covering progressive projection, resource admission and policy/response
regressions. This precedes the exact committed-head rebuild and is not a
replacement for that qualification. The revision endpoint runner deliberately
includes calculator construction, both prepared lifetimes, detached copies and
cleanup, preserves first use and every measured sample, and rejects a mismatched
archived Python/native source identity. Native numeric subplan declarations and
timings are retained separately from physical work and model identity.
Actual execution provenance is attached to each detached proposal and retained
per spin in prepared reports; a capability query alone does not prove execution.

## Review correction: retained-metric numerical accuracy

A retained source metric with condition number 1e8 exposed a compatibility
regression in the scalar provider. Its ordinary relative eigen stopping rule
left enough eigenvector error that reconstructing a pure six-electron density
produced populations 1.999999869 and 2.000000180, outside the public 1e-7 purity
tolerance. The independent relative `D S D - 2 D` error was 1.76e-9; the earlier
NumPy route accepted the same state. Discarding supported modes or widening
purity acceptance would hide the numerical error rather than correct it.

An initial scalar-only stopping cap fixed that fixture but was insufficient for
a reduced two-AO density and other retained modes with condition number near
9e8. The correction therefore also refines the already-computed metric
eigenspace with a Rayleigh-Ritz congruence and the existing ordinary CPU
eigensolver. Two-pass accurate orthonormalization first restores the premise
that the ordinary eigenvectors form an orthonormal basis; an O(epsilon) Gram
error otherwise tipped a valid reconstruction from 9.9664452073e-8 over
the unchanged 1e-7 gate. This normalization is in place and needs no additional
matrix storage. Refinement precedes PSD/rank classification; the existing FP64
`relative_threshold * largest_eigenvalue` expression and strict retained-mode
comparison are preserved exactly. The scalar cap uses the existing solver
parameter; no shared solver or historical transport defaults change.

Adapter-private two-component FP64 sums, FMA product residuals and corrected
division retain cancellation terms through metric and density contractions.
Density congruence, coefficient reconstruction and the separate elementwise
reconstruction comparison use those components. This does not assume that
`long double` is wider than `double`, introduce another eigen equation owner,
or relax any public PSD, rank, purity, reconstruction, target or residual gate.
Finite-overflow paths still refuse before publication. Coefficients and density
remain ordinary FP64 at the transport/publication boundary.

Permanent regressions retain the exact seven-AO and reduced two-AO valid states,
small/large scales and AO sign gauges. They also retain separate ill-conditioned
states that fail purity or reconstruction even though the old NumPy path may
accept them. Independent 70/80-digit reconstruction classified fourteen of
sixteen retained native-refusal fixtures as valid: one failed reconstruction
and one failed purity. The repaired adapter accepts the fourteen and rejects
both invalid states. Removing eigenspace refinement, its accurate orthonormalization or accurate
density congruence makes a valid reproducer fail; removing either source gate
lets an invalid fixture through. These are numerical behavior checks, not source-text assertions.

The unchanged numeric reservation is 256 times the sum of source-square,
target-square and cross-overlap element counts. A scalar-provider live-storage
audit, including accurate temporaries, ordinary eigen scratch and result
publication, bounds its peak by 8 times (26 source-square +20 target-square
+2 cross-overlap) bytes. Thirty-two tracked allocation checks include highly
asymmetric AO shapes and full/empty channels; all stay below the declaration.
The reservation is not an RSS limit. LAPACKE queries provider-dependent optimal
workspace, so a scalar proof cannot establish its capacity bound; LAPACK
runtime/capacity qualification remains open.

The repaired scalar sources passed independent ambient least-squares/gauge and
transactional public-ABI checks, the fourteen valid/two invalid source fixtures,
and rank/PSD/cutoff/extreme-scale controls. A separate build with FP64-width
`long double` and compiler FMA inlining/contraction disabled reproduced those
outcomes. The runtime `std::fma` implementation was not forced to software;
software-only libm execution is not claimed.
The independent differential suite found no oracle-valid original success
lost; three new refusals were confirmed invalid by 80-digit source-gate checks.
The adapter remains a CPU proposal route, not a target-SCF convergence claim.
GPU execution, LAPACK execution and repaired-source endpoint performance are
not qualified by this bounded numerical correction.

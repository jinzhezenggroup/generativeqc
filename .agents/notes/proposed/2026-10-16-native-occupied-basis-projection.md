# Proposal: one native occupied-basis transport owner

Status: proposed
Date: 2026-10-16

## Problem

The progressive HF consumer currently reconstructs a pure occupied source from
its density, validates source/target metrics and projects with NumPy. Native
`initial_guess::project_occupied_density` already implements the occupied
transport; independently translating the same projection into another native
module would preserve two authorities rather than solve #2227 R4/R6.

## Direction under implementation

Use the existing transport for both contracts. Preserve its original signature,
relative Gram cutoff, roundoff residual snapping and acceptance as defaults.
An explicit controlled entry adds the public basis policy's absolute Gram
cutoff, configurable validation tolerance and unsnapped residual, while the
same transport computes optional normal-equation diagnostics from its actual
unorthogonalized image. There is no second least-squares projection formula.
The new adapter explicitly supplies the ordinary shared CPU eigen operation;
the original initial-guess entry retains its historical default. Reusing the
legacy transport must not silently route a newly admitted production request
through an independent reference eigensolver.

The new adapter validates PSD metrics, retains the declared spectral modes,
rejects discarded source components, reconstructs integer-occupied pure HF
densities and publishes detached target density/coefficients and diagnostics.
Rectangular coefficients are padded only at the old square-frame boundary.
Metric spectral-norm symmetry validation uses scaled A-transpose-A through the
shared CPU algebra/eigen owner; this is not a replacement eigensolver. Gauge
changes must be tested through density and metric invariants, not vector signs.

The public Python policy and legacy native initial-guess policy are distinct:
do not silently replace one threshold convention with the other. An empty spin
channel is valid even if all metric eigenmodes are discarded at an inclusive
cutoff. Nonfinite intermediate contractions must reject, not disappear through
max reductions. Projection remains a seed, never target SCF convergence.

## Evidence so far and required completion

Preliminary hand-derived nonnested projection/rotation, density reconstruction,
metric rank, empty-spin, refusal and detached-result tests pass using the new
compiled adapter and original scalar scientific library. Eight source-Gram
overflow/finite cases now compile real translation units through the existing
cached test fixture instead of extracting production body text or injecting a
fake eigensolver. These are scoped development checks, not exact-master final
library or endpoint qualification.

An additional provisional compiled bridge exercised 200 seeded random ambient
space fixtures in occupied-column and density-input modes (400 projections),
against independent NumPy Euclidean least squares with metric/electron gates.
Maximum target density difference was 3.197442310920451e-14. This preview links
the newly compiled core to the earlier installed scalar library; its result is
not transferred to a not-yet-built final master-based scientific library.

Before calling this delivered: add installed ABI/buffer transactionality and
native authoritative numeric-capacity admission; wire normal built-in Python
production to the shared owner with explicit old-library/custom compatibility;
test independent Euclidean least squares and reconverged molecular E/F; qualify
source-matched installed-library endpoints including original/moved, failures,
resource limits and cleanup. Retain physical/provider accounting limitations.
No full progressive stage/strict-cleanup execution cutover or #2227 completion
is claimed by this provisional core.

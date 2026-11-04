# Decision: qualify a bounded native local saddle corrector before task admission

Status: implemented
Date: 2026-11-05

## Problem and decision

Issue #2848 requires native saddle optimization, not a geomeTRIC bridge or a
minimum optimizer with a TS label. Its requested molecular domain still lacks
complete qualified total-method derivative endpoints. Keep algorithm validation
independent of that admission: add an internal C++ local corrector whose
callbacks explicitly supply energy/gradient and raw Cartesian Hessian at the
same coordinates. It has no installed API/CLI task exposure and does not claim
to validate electronic provenance or complete #2848.

Use per-mode rational eigenvector-following: reverse the selected negative-mode
step sign and minimize the other modes, bounding the total Cartesian step by
a trust radius. Track the target by normalized absolute eigenvector overlap,
align its sign and fail if the declared overlap threshold is lost. Reuse the
prepared CPU symmetric-eigen owner; do not add a competing eigensolver.
Initial curvature is exact; refresh the raw Hessian at every accepted trial.
There is no approximate or stale Hessian update policy in this first slice.

Accept a trial using actual/predicted reduction of the gradient-squared merit,
not energy decrease: legitimate saddle motion can be uphill in energy.
Shrink/grow the radius using this ratio. A small stationary trial has a bounded
roundoff branch but must independently meet energy, both gradient, and both
displacement tolerances. Every trial, accepted or rejected, consumes one attempt
from a cap of at most 100; preparation and fresh final verification are separate
evaluations. At the cap with unmet gates, return nonconvergence, never success.

## Numerical and ownership boundaries

This is a **local corrector with an already negative target mode**, not an
algorithm that escapes a minimum or finds a transition state from arbitrary
reactants. Gradient-merit acceptance does not justify that broader claim.
Positive target curvature is an explicit failure for a nonstationary input.
Stationary minima/higher-index points are reported separately. Curvatures within
the declared `1e-7 Eh/Bohr^2` zero band remain reported and block first-order
saddle classification; they are never deleted to manufacture one negative mode.

Five default gates are `1e-6 Eh` energy change, component gradient RMS `3e-4`
and maximum `4.5e-4 Eh/Bohr`, and component displacement RMS `1.2e-3` and
maximum `1.8e-3 Angstrom`. RMS denominators are `3N`, with no implicit rigid
alignment. Steps/coordinates are Bohr; one explicit conversion is applied to
displacements. An initially stationary point uses zero displacement and must
still pass a fresh same-geometry energy/gradient/Hessian verification.

Raw Hessians must be finite, dimensionally complete and symmetric to the fixed
declared admission tolerance; no posthoc symmetrization is performed. Dense
coordinate storage is explicitly bounded (default 300 coordinates). Failed
energy/gradient, curvature, target tracking, trust exhaustion, cancellation and
the iteration cap preserve the last fully accepted state and explicit status.
Failed final verification never returns a verified stationary point. Counts
include attempted callbacks even when they fail; history distinguishes rejected
trials from accepted geometry updates.

Do not reuse this raw Cartesian index as a molecular vibrational index: no
translation/rotation projection or physical mass analysis is performed here.
Molecular task composition must later bind a single verified MethodIR, basis,
ECP, grid, charge/spin and qualified native derivative owner to every callback,
then independently qualify the requested chemistry. No external optimizer,
oracle, finite differences, preoptimization or functional substitution enters
this native algorithm.

## Evidence and open gates

The compiled host unit test uses independent analytic polynomial derivatives
and a known saddle at zero. It exercises all five gates, reaction mode,
fresh final evaluations, exact 100-rejected-attempt exhaustion, minimum/wrong
mode/higher-index/soft-mode controls, near-degenerate target selection and
eigenvalue reordering, target-overlap loss, trust exhaustion, cancellation,
SCF/response failures, malformed raw Hessians and changed final energy.

CPU-only CMake/CTest qualification passes both the new saddle test and the
existing minimum optimizer test (2/2, 0.01 s). The new test also passes an
independent ASan/UBSan/leak-detection build with whole exit status zero.
Release compiler uses `-Wall -Wextra -Werror` in the standalone check; CMake's
verbose build commands verify the selected sccache 0.16.0 launcher. Cache
statistics and `saddle-cmake-ctest-v2.log` / `saddle-sanitizer-v2.log` are retained
locally under `/tmp/gqc-2843-2849/`. These timings are only mathematical unit
tests, not molecular endpoint or optimizer performance claims.

Independent molecular saddle/minimum controls, molecular projection and
near-degenerate subspace qualification, installed fixed-method public API/CLI,
complete endpoint work/memory evidence and Ni/ECP/total-D3 acceptance remain
open. This internal mathematical delivery is useful but cannot close #2848 or
qualify TS+frequency execution in #2849.

## References

- Issue #2848, derivative dependencies #2843 through #2847 and consumer #2849.
- `src/geometry/saddle_optimizer.hpp`.
- `tests/native/test_saddle_optimizer.cpp`.

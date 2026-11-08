# Derivative AD review integration

Status: implemented
Date: 2026-11-07

The derivative ownership migration exposed build-time and finite-range gaps in
review. Geometry package exports now load lazily, like the method package, so
scalar source generation does not import NumPy-backed runtime execution. The
GFN2 generator uses the real package rather than installing a substitute package
module. Array-only nuclear response imports remain local to that runtime path.
Bare `python -I -S` subprocess tests compare generated output with ordinary
source generation, including gCP, dispersion, nuclear repulsion and GFN2.

The generated erf tangent previously multiplied a near-maximum finite direction
by `2/sqrt(pi)` before its small Gaussian factor. At input 10 and direction
1.7e308 this overflowed although the derivative is about 7.136e264. The response
now differentiates the primal first and contracts its complete slope with the
direction. This retains compiler-owned AD and avoids that intermediate overflow.
An independent high-precision primal oracle covers large and subnormal seeds.
There is no demonstrated molecular endpoint failure at those extreme seeds.

Fully qualified atomic-grid namespaces preserve coexistence with semilocal recipe
metadata in one translation unit. Compiler type declarations reflect existing
scalar payload and generated-callable contracts without changing equations.

The original derivative campaign remains a historical source reconstruction.
Smaller JSON metadata and the original samples are losslessly XZ-compressed to
fit the unchanged aggregate evidence budget; stored checksums are refreshed.
No samples, independent oracles, or old source identities are discarded. Review
repairs have host tests and CI, not fresh GPU endpoint performance evidence.
The native Libxc snapshot must be migrated with exact source-identity changes,
retaining stale measurements and zero newly qualified runtime contexts.

Run 42410861878 exposed two host-only probes that bypass CMake: the actual DF
force owner and device ledger translation units now require generated classical
headers. A shared test fixture invokes the real generators under bare Python,
without adding checked-in fallback headers or replacing production arithmetic.

The same run exhausted core-c while two scalar-ECP HVP cases passed in 889 and
1157 seconds. The nine added AD test files had shifted 1272 of 2309 existing
module owners in the merged-base modulo inventory, collocating those cases with
core-c's molecular Hessian suite. New files now get supplemental non-modulo
slots. A one-off merged-base census verified that all 2309 old owners remain
unchanged; behavioral routing tests cover tracked, missing and untracked inputs
across all five events. Tests, priority, time caps and production CPU code remain
unchanged. This is an inventory integration correction, not timing qualification.

The subsequent [CI integration resolution](2026-11-08-derivative-ad-ci-integration.md)
keeps the nuclear array executor in `method.nuclear_repulsion_array`, with direct
caller migration rather than local NumPy imports or an old-location alias. It
retains the self-contained compressed evidence and the explicit erf numerical
repair described here.

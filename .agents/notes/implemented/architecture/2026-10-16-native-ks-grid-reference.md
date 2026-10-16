# Decision: authenticate native grid response against the prepared KS owner

Status: implemented
Date: 2026-10-16

## Problem and decision

The native fixed-grid WB97M-V component previously had only a fixture
authenticator. Copied identity labels or a detached Python snapshot cannot
prove immutable basis/quadrature identity and accepted-density lifetime.

`CudaKsPlan::resident_final_grid` reuses the exact canonical-frame gate, then
borrows packed AoBasis records/quadrature from its actual `CudaXcPlan` and D
from the accepted frame. The complete binding includes all pointers, extents,
stream/device and owner/solve/generation fields. Borrowing submits no work,
device/panel allocation, transfer or synchronization. Host identity metadata may
be copied by the existing token gate. Stale/unvalidated states clear output;
host-unfused owners without resident XC do not stage replacement storage.

The private batch/item/token port and nonmovable `CudaKsGridReference` reborrow
this exact binding at the grid component's execution/publication gates. The
batch and adapter must survive every synchronous call without concurrent batch
access. A new solve revokes the reference before numerical reads; a stale
action leaves public output untouched. Exact normalized records and actual
molecular-grid pointer identity are independently checked. Numerical quadrature
parity remains the existing quadrature fixture's responsibility.

Storage authentication is functional-neutral: it does not turn a PBE density
into a stationary WB97M-V reference. The enclosing physical provider must also
validate semilocal, exact/fitted J/K, LR correction and VV10 identity. Physical
J/K composition, nuclear motion, Hessian assembly and CLI replacement remain
incomplete. This change adds no production numerical algebra or CPU fallback.

## Independent oracle boundary: default FHC is not a signed-D extension

The probe solves actual H2 RKS and H3 UKS WB97M-V batches, including LR exchange
and VV10, then runs the native grid component against their live storage.
The whole subprocess executes with executable/Python paths unavailable.
Only test observations cross to Python. Independent normalized Gaussian jets,
pinned Libxc 7.0.0 value potentials and a separate real-space VV10 oracle
qualify response by displaced values, not compiler/AD self-comparison.

There are two distinct tests, both with genuinely indefinite directions:

- Arbitrary signed D uses isolated Libxc handles with optional FHC projection
  disabled, matching the compiler's declared rho/sigma/tau-floor extension.
  Default reference handles are never mutated.
- Physical paths retain completely unmodified default Libxc. Displacements
  follow `(I+t*A) D (I+t*A)^T` with skew A, including the quadratic term. The
  native direction is `A D + D A^T`; eigenvalues check its indefinite nature
  and PSD displaced states.

Pinned upstream `functionals.c` enables Fermi-hole-curvature projection under
`XC_ENFORCE_FERMI_HOLE_CURVATURE`; `work_mgga_inc.c` then caps sigma at
`8*rho*tau`. At rank-one reference densities, arbitrary signed D leaves the
PSD cone. Centered differences of those projected values are not the raw
functional's linear density kernel. Initial errors plateaued at 3.64e-5 RKS
and 1.216e-4 UKS as steps shrank. Separating semilocal and VV10 localized this
solely to the oracle projection. Disabling FHC on isolated upstream handles
restored about 1e-12 RKS error and quadratic UKS convergence
1.46e-9 -> 1.31e-10 at steps 1e-3 -> 3e-4. No production math changed.

Do not differentiate a reference-only projection in production, loosen gates,
or replace independent evidence with the incumbent compiler graph. Acceptance
stays value `atol=3e-10, rtol=3e-9`, response `atol=3e-9, rtol=2e-6`, at both
steps. Existing spherical d/f and quad-precision empty-spin gates remain intact.

## Evidence and consequences

Finite Slurm job 3093, node2/main/PRO6000, assigned visibility 0: six native
frame/grid lifecycle cases and four molecular response cases pass. The latter
take 7.47 s: a test duration, not an endpoint benchmark. H2/H3 have 576/882
active points, or 331776/777924 ordered VV10 pairs each for reference/response.
Finite job 3094 repeats RKS/UKS physical-path execution under memcheck,
initcheck, racecheck and synccheck: zero errors/hazards in all eight runs.
Final job 3096 reruns the six lifecycle cases and integrated native
leaf/GMRES/CPKS/grid suite: 163 passed in 67.80 s. CPU regression: 22 passed,
141 explicitly opt-in skipped; CUDA-disabled native method/adapter syntax
compiles, pinned ty 0.0.82 and prek 0.3.1 all-files checks pass. Compiler
dependency checking reports 554 modules and zero errors. These durations are
test execution evidence, not complete molecular Hessian endpoint timing.
After the final header-documentation rebuild, job 3097 requalifies the exact
library/probe bits: six lifecycle cases, 163 integrated tests in 68.08 s and all
eight sanitizer runs remain clean. Job 3098 reruns the final annotated grid
test source: 33 passed in 20.90 s. The probe SHA256 is
`8b0fbe371de4ed5a9358a8ffde435b70cb96ed03f475145556ce480080f024b9`;
the shared native library is
`c349027c129be18255f501b8c3e279bdfc8bc32aaa5f8cd7ab9f2e38dda9f5a1`.
The final type-check-only import cleanup is re-executed in job 3099: all 33
grid tests pass in 20.86 s; the final pinned all-files lint suite also passes.
Zero-transfer reborrowing, missing/foreign/stale/warm-replayed owners and
explicit invalidation retain coverage, including empty/unequal-spin UKS.

Raw build/cache/Slurm/diagnosis receipts are ignored local
`.artifacts/wb97mv-hessian-aot-20261015/grid-binding-*`. Initial failed
oracle/extra quadrature-comparison runs are diagnostics, not qualification.
This is not complete molecular Hessian or GPU4PySCF speed evidence.

Next physical J/K composition must preserve source/projection scratch lifetimes,
the borrowed producer stream and signed-D semantics. Revisit the oracle extension
only if the production work policy explicitly changes and is independently
requalified, never merely to hide a boundary mismatch.

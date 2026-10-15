# Decision: resident CUDA numerical execution of the WB97M-V Hessian

Status: implemented
Date: 2026-10-15

## Problem

The initial compiler-AD Hessian supported CUDA/UKS/DF/ECP/spherical combinations
numerically, but its CUDA contract was explicitly mixed. First integrals,
grid/XC/VV10 AD, raw DF derivatives, ECP projectors and coupled-spin response
still used host arrays. A CUDA backend label or a CUDA second-integral kernel
does not establish an all-device derivative/response endpoint.

This supersedes the placement policy, not the mathematical decisions, in
[the domain extension note](2026-10-15-wb97mv-hessian-domains.md).

## Decision

`CudaCompositeResponse` owns the numerical Hessian/response stage beginning at
the converged native snapshot. It uploads that snapshot and evaluates the
energy-weighted density on the device. The native SCF preparation contract is
unchanged; this is not a claim that native input/pre-SCF preparation contains no
host numerical work. Host shell enumeration, basis definitions, fixed quadrature
constants, compilation, provenance checks and scalar solver decisions remain
host responsibilities. No host molecular derivative evaluator is selected.

The compiler remains the differentiation owner:

- Scalar Graph lowering emits CUDA lane programs with lazy piecewise branches.
  The same generated roots define semilocal feature Hessians, VV10 pair JVPs,
  partition response, ECP radial/projector factors and nuclear-repulsion HVPs.
- TensorIR retains the existing feature, density bilinear, stationary-weight
  and full-rank DF cotangent/JVP programs. Its namespace executor binds CUDA
  storage and contractions without changing derivative rules. Device finiteness
  status can accumulate without a per-node host synchronization; publication
  must check it.
- The normalized native packed basis and existing generated Gaussian axis AD
  factors define resident AO jets through f/order three. There is no grid fit
  or reconstructed spherical convention.
- The existing second-integral emitter has an additive value/gradient/HVP
  consumer. Its default emission is unchanged. At most 16 raw AO components
  and 64 radial products are streamed per resident integral tile. DF uses its
  existing physical Hermite/Boys graph and next-moment gradient JVP.

CuPy is an optional runtime storage/CUDA-linear-algebra dependency, exposed as
`generativeqc[cuda-hessian]`; it does not own AD. Compiler source generation
never imports it or probes a GPU. A context-local execution owner prevents a
CUDA operation from changing another thread's NumPy execution. Missing CuPy is
an explicit error, not selection of the old mixed Hessian path. The public
dispatcher also rejects a non-resident CUDA owner rather than using host AD.

Coupled RKS/UKS CPKS uses independent physical occupied/virtual blocks, including
empty spin blocks. Restarted, reorthogonalized GMRES retains Krylov vectors,
Hessenberg least squares and solution updates on the device. Acceptance uses
the true physical residual and a scaled FP64 norm, including tiny RHS values.
D and energy-weighted-D reconstruction retain the canonical metric RHS and
occupation factors qualified in the original domain note.

## Resources and fallbacks

Dense primal ERIs are optional CPKS reuse, never an unconditional n^4 resource
requirement. Their conservative reserve is selected only after protecting the
full-rank DF reserve. If it does not fit, J/K streams shell/component tiles on
the device. Derivatives always stream, contracting compiler-generated
stationary weights on shell-local components rather than materializing a
nuclear Jacobian or derivative ERI tensor. Reusing an owner rechecks the
retained reserve. The electronic approximation never changes with this choice.

Every numerical allocation uses a private CUDA pool with an aggregate cap from
the integral, grid, response and publication allowances. Native SCF/provider
storage, modules, compiler processes and library handles have separate existing
contracts. The owner closes its stream work and releases its private storage.

Only discrete branch metadata and scalar control/status values may be downloaded
during execution. These transfers are explicit in `branch_metadata_bytes`.
AO/density/Fock/derivative panels remain resident. HVP components are downloaded
together after all gates. Full Hessian columns remain resident until final
publication; no post-hoc symmetrization or projection is allowed.

## Invariants

- Preserve fitted full-range J/K plus the exact LR correction, not a different
  range-separated RI-K model. Preserve physical alpha/beta spin factors.
- Preserve independent orbital/auxiliary normalized spherical maps through f.
- Preserve the full-rank metric cutoff separation and explicit truncated-metric
  rejection; do not freeze a retained eigenspace.
- Preserve effective nuclear charges and both residual ECP channels, with the
  original coarse/fine absolute HVP quadrature gate.
- No displaced SCF, finite differences, PySCF or other reference engine enters
  production. Independent references and differences are acceptance tests only.
- Generated scientific kernels use the verified compiler adapters, finite
  compilation lifetime and existing sccache/ccache caches. CuPy's in-process
  NVRTC/library caches are not cleared. Artifact replay does not require a
  compiler-cache executable when no compilation occurs.

## Evidence

Slurm n1 RTX 5090 qualification, using the consistently rebuilt native library
from the preceding domain qualification:

- Job 8621: all six endpoint tests passed in 556.67 seconds. These cover direct
  RKS/UKS, a raw UKS matrix, DF, ECP, combined DF/ECP/UKS/spherical execution and
  nonempty occupied/virtual response blocks in both spins. Gates include CPU
  parity, reconverged native-force differences and raw matrix symmetry/HVP
  consistency, without repair.
- The resident lowering run passed 14 tests in 44.59 seconds: spherical AO jets
  through f, SR/LR value/gradient/HVP lowering against an interval-quadrature
  gradient oracle, inactive scalar branches, true-residual GMRES at ordinary
  and 1e-270 RHS scales, failure gates, and a complete endpoint with CPU AO,
  scalar-graph and DF derivative evaluators poisoned. An explicitly constrained
  integral allowance selected the device-streamed fallback and agreed with
  resident dense reuse.
- A focused CPU regression passed 29 tests before the broader final run.
  Compiler structure checks report 529 modules and zero dependency errors.
- Final-source job 8628 passed all 16 resident tests in 218.69 seconds. The added
  gates verify aggregate nonfinite TensorIR rejection before publication and a
  combined three-center synthetic ECP/DF/UKS fixture with nonempty OV blocks in
  both spins and a genuinely spherical d auxiliary map (different Cartesian and
  public dimensions).
- Final-source job 8629 repeated the direct RKS, raw UKS matrix and coupled H3
  force-oracle endpoints: three passed in 228.64 seconds. This also rechecks
  device-side W construction and the final per-endpoint work diagnostics.
- The final broad CPU regression passed all 143 tests in 1088.13 seconds,
  covering compiler lowering, second-order grid response, RKS/UKS/DF/ECP
  Hessians, basis transforms, range-separated second integrals, stationary
  response plans and compiler dependency boundaries. A separate generic
  TensorIR namespace/grid-response regression passed all 33 tests in 1.57
  seconds. Logs are `wb97mv-resident-cpu-final.log` and
  `wb97mv-resident-generic-regression.log` under `/tmp/`.

Retained local logs start with `/tmp/wb97mv-resident-`. The shared sccache stats
are not a hit-rate benchmark: runtime JIT uses the required per-invocation
foreground server/slot ownership, not the shared stats server. No speedup or
scaling claim is made from these qualification timings. Work counters and
complete Calculator endpoint timing are reported separately from storage.

## Revisit when

Fuse/tile resident TensorIR and grid/ECP actions or add a native generalized
resident J/K ABI only with complete-endpoint work/transfer/timing evidence.
The current conservative device traversal is a correctness implementation, not
a claim that moving every small operation to the GPU is automatically faster.
Shared multi-RHS response, truncated-metric spectral Hessians and additional
precision modes require independent scientific qualification.

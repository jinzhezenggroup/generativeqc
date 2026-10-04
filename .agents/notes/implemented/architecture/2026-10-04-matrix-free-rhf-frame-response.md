# Decision: matrix-sized RHF frame AD around an exact resident J/K action

Status: implemented
Date: 2026-10-04

## Problem

The conventional CCSD(T) force owner materializes a full MO N^4 Hamiltonian,
then applies every occupied/virtual basis vector to assemble a dense orbital
Hessian before GMRES. Reusing that machinery would defeat the DF source's
structural savings on 230/264-AO molecules. The streamed source nuclear sink
already supplies a frame cotangent, but does not supply Z or Pulay response.

## Decision

The method compiler owns matrix-only primal density/Fock/frame maps and derives
both reverse stages and orbital tangent with the existing TensorIR AD. An
external physical exact `G(D)=J(D)-K(D)/2` separates the reverse stages and tangent
stages. Native runtime executes that signed symmetric-density action on the
same direct-provider stream, without host intermediate densities.

For `R=C.T bar_C`, the antisymmetric part is orbital stationarity and
`C[-(R+R.T)/4]C.T` is the full AO overlap cotangent. The rotation convention is
`K_ia=x, K_ai=-x`, with `A=-dFov/dx` and RHS `-(R-R.T)_ov`. After solving `A z=rhs`,
subtract z from the full Fov seed and recompute all weights. No occupied/occupied
or virtual/virtual energy-gap division is introduced.

Runtime-shaped CUDA lowering optionally uses packed matrix GEMM contracts
already recognized by the shared tensor compiler. Higher ranks, scalar products
and layouts needing packing retain generated kernels. The scalar schedule uses
the same arena and remains explicit fallback/audit. BLAS results are audited
before scratch reuse; only the enclosing complete action clears the error flag.

The native owner validates canonical Fock, C.T S C, reconstructed D and F=h+G(D),
then solves on demand with bounded GMRES. A fresh scalar-CUDA action audits the Z
residual (1e-10); the complete frame stationarity gate is 1e-8. Matrix-free local
convergence does not certify global RHF stability. The result says so explicitly
and does not publish a fictitious minimum-curvature estimate. Public full-DF
forces remain gated while the method composition is unfinished.

The exact reference two-electron derivative uses polarization of the existing
quadratic derivative provider in three passes: E2'(D+P)-E2'(D)-E2'(P). All passes
are unscreened and share the physical source. This avoids a coordinate-indexed
ERI derivative, but retains three traversals and possible cancellation. A
compiler-owned bilinear consumer may replace this fallback after qualification.

## Invariants and admission

- The conventional RHF branch must use exact G, even when correlation is DF.
- No N^4 input, full orbital Hessian, or CPU integral/response fallback.
- All frame AD tensors have rank at most two. All contraction summands have
  degree at most three in occupied/virtual/complete-orbital extents.
- Matrix control/storage grows quadratically plus bounded Krylov storage;
  physical exact J/K work still depends on quartets and iteration count.
- Complete numeric admission counts borrowed system/reference/seeds, device
  arena/provider allowance, conservative host setup/output and Krylov capacity.
- Transfer counters cover the matrix owner, not all integral-provider traffic.
- Electronic gradients require correlation-source and nuclear contributions
  before the final sign conversion to forces.

## Evidence

The 15 mathematical tests include independent NumPy nuclear-source directions,
metric transport, exact same-space degeneracy, dense existing CC-Hamiltonian JVP
agreement, tiny independent Z solutions and HF Pulay identities. Shape witnesses
include (occupied,virtual)=(9,221) and (21,243). Two runtime-lowering tests check
all five maps at four shapes and deterministic optional-BLAS/scalar emission.
Seven existing Hamiltonian codegen tests also passed (24 host tests combined).

Slurm job 12217 passed all 11 CUDA molecular tests in 9.86 seconds using library
SHA256 `7043f063215bc2c817b36ac03530a5a7cbb05cec4d068eab861b8c46fbc8295f`.
Both schedules reproduce independent complete RHF forces for H2/water/LiH and
nonzero-Z toy molecular energy derivatives at two steps for water/LiH. Tests
also reject one-byte-short admission, inconsistent reference, same-space torque,
nonfinite seeds and unconverged Z without publication. Explicit Hessian elements
are zero. These timings are qualification test times, not complete CC endpoints.
Slurm job 12218 repeated all 11 cases under compute-sanitizer memcheck in
105.66 seconds with zero errors (`--target-processes all`). Static hooks and
type checks pass. Ignored payloads are under `.artifacts/orbital-response/`.

## Rejected alternatives and remaining work

- Reuse dense raw CC Hamiltonian response: restores N^4 storage and all Hessian
  basis-column J/K applications.
- Silently use DF reference response: differentiates a different Hamiltonian.
- Replace unrestricted C by orthogonal rotation variables everywhere: loses the
  symmetric cotangent required by Pulay metric transport.
- Certify positive curvature from a few Krylov vectors: not a global guarantee.
- Rewrite CC derivatives by hand: compose the existing generated sources.

Next, combine triples/corrected Lambda/retained-factor sources and the nuclear
sink with this exact-reference branch in one admitted native force owner, then
qualify complete independent DF-CCSD(T) force gates and hundreds-AO endpoints.
References: #158, #1763–#1765, #1799, #1802, #1806.

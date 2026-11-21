# Decision: Authenticate the CUDA Hessian's SCF handoff

Status: implemented
Date: 2026-11-21

## Decision

Separate SCF proposal arithmetic from the accepted stationary model. The
native CUDA KS owner certifies the converged generation's physical FP64
Fock, not its requested precision or installed integral provider. Neither
mixed Fock components nor a seeded or frozen nonlocal iteration can certify
it. The endpoint also refuses application-budget stationarity: a fresh FP64
Fock alone does not imply the SCF was closed with the strict stopping policy.

Both the raw device endpoint and the C API check the token before preparing
response resources. Live state borrows still compare the entire token to the
current owner and solve generation. The borrowed state, the grid geometry
response, the integral/metric response and CPKS stay FP64. AUTO must not select
a different XC point domain, nonlocal grid, or fitting Hamiltonian to make a
Hessian possible.

## Direct evidence

- Slurm 2088 (`node5`, RTX 5090): 12 native AUTO probes for LDA, PBE, PBE0,
  B3LYP and WB97M-V RKS/UKS, including independent WB97M-V meshes. Hybrid
  fixtures prove mixed Focks and strict refinement ran; rejection of an
  uncertified token preserves the caller's output buffer.
- Independent reconverged energy stencils use Libcint integrals, Libxc point
  values and the independent VV10/partition reference where relevant. Compare
  the center Fock, density and energy and every matrix element after Richardson
  extrapolation. Maximum absolute matrix errors (Ha/Bohr²): PBE0 RKS
  `6.52e-10`, UKS `2.29e-9`; split-grid WB97M-V RKS `1.01e-9`, UKS `3.06e-9`.
  Compare dimensionless Becke fractions at the center rather than amplifying
  their errors with large radial weights at near-zero density.
- Slurm 2091: 14 Python/CLI AUTO, rejection and r²SCAN+rVV10 regression
  tests passed. Raw SCF/response and independent stencil receipts and build
  logs are in `build/hessian-gpu-sm120/hessian-auto-*`.

## Fitted PBE

Reuse the fitted pure-J XC proposal schedule for ordinary AUTO only for
all-electron, device-fused PBE RKS, without authorizing the force-budget
stopping policy. Only stock-grid geometries in the existing AO/atom/angular
domain use mixed XC matrix contractions. Outside that domain, ordinary AUTO
runs the same provider, metric, grid and PBE point program in FP64; it does not
switch to Direct J. The fitted integrals and all response sources never enter
the mixed schedule. Keep AUTO disabled for fitted hybrids and for r²SCAN,
where the incumbent and variational point programs are distinct.

Slurm 2095, RTX 5090: the 26-AO, 19,616-point fixture used five mixed SCF Focks
and seven strict refinement iterations, with physical CPKS residual `3.28e-13`.
All 36 Hessian entries agree with reconverged Libcint/Libxc energies at
`0.004` and `0.002` Bohr (Richardson extrapolated) within `6.52e-10 Ha/Bohr²`.
Raw symmetry, translation and matrix-free action were also checked at `3e-9`.
Eighteen frontend and regression tests passed, including mixed fitted PBE,
small-basis strict fallback and C API rejection of an actually converged
force-budget state. Slurm 2096 added 35 strict FP64 tests covering Direct,
fitted, fitted long-range exchange and ECP. The raw results and build/cache logs are in
`build/hessian-gpu-sm120/hessian-fitted-auto-*`.

## Revisit when

FP64 physical closure is not qualification of lower-precision derivative
arithmetic. A mixed response implementation must independently qualify the
final physical response residual, raw dense symmetry, translation, and both
dense and matrix-free error against reconverged energies before deployment.

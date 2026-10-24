# Decision: extend compiler-AD WB97M-V Hessians without changing the primal

Status: implemented
Date: 2026-10-15

## Problem

The initial CPU/direct/RKS/Cartesian/all-electron endpoint omitted requested
spin, representation, fitting, pseudopotential and device domains. Merely
relaxing capability checks would differentiate the wrong variational space,
Hamiltonian or fitted exchange approximation.

This extends the admission boundary in
[the initial compiler-AD Hessian decision](2026-10-15-wb97mv-compiler-ad-hessian.md).

## Decision

- Derivative workspaces use the exact native normalized Cartesian/spherical
  expansion map. Densities and energy cotangents pull back with `T D T^T`,
  operators project with `T^T F T`. The native spherical SCF space is retained.
- UKS has independent alpha/beta OV blocks and a coupled installed GMRES solve.
  All response actions retain cross-spin J/XC/VV10 coupling and occupations
  1/0. Empty spin occupied/virtual spaces are legal. The canonical nuclear
  metric connection uses `(epsilon_a + epsilon_i)/2`, not `epsilon_i` alone.
  The latter failed the reconverged HeH doublet HVP by approximately 6%.
- Preserve the prepared DF primal: fitted full-range J/K at fraction 0.15,
  plus exact LR exchange at fraction 0.85. For the existing source ABI,
  SR = fitted full-range exchange minus its exact LR part, and LR = exact LR
  at fraction 1. No separately fitted SR operator is invented.
- Generate DF primitive value/gradient/gradient-JVP DAGs from the existing
  physical metric/three-center primals. The Boys derivative consumes the next
  moment. Stream bounded component and radial-product tiles through the
  existing native contraction template. TensorIR generates source VJPs and
  their JVPs, including the full-rank inverse-metric implicit-solve rule.
- Admit only full-rank metrics separated from the native relative cutoff.
  Rank truncation needs a separately qualified spectral second-order rule;
  a frozen retained eigenbasis is not that rule. Reconstruct fitted Fock from
  compiler sources and compare with the native fitted owner before response.
- ECP differentiates both the residual local potential and harmonic-projector
  bilinears. Fixed ECP-centered radial/angular grids move with their center;
  the AO relative motion includes both that motion and the orbital center.
  Demand only the AO-density bilinear JVP rather than computing unused
  meta-GGA gradient/tau fields. Native coarse/fine quadrature and its HVP
  acceptance gate remain explicit. Effective nuclear charges supply the tail
  and nuclear repulsion exactly once.
- Native nonlocal model proof is independent of the Hamiltonian proof; ECP
  snapshots retain both identities. WB97M-V forward preparation admits ECP+DF
  without broadly promoting other methods' ECP/DF force domains.
- CUDA has an explicit mixed residency contract. Native SCF/J/K and generated
  weighted direct second integrals execute on CUDA. Grid/DF/ECP AD, first
  integral contraction and solver orchestration remain host work, reported in
  diagnostics. This is not an all-GPU residency or performance claim.

## Rejected alternatives

- Full-Coulomb/direct substitution for fitted derivatives: changes the primal.
- Treating native full-range fitted exchange as fitted SR exchange: changes
  the resolved RSH-DF approximation and fails complete-energy derivatives.
- Four-center binders with zero-exponent dummy shells: their positive-exponent
  geometry contract excludes that representation. DF has explicit two-/three-
  center physical binders instead.
- Fitting the spherical map on a grid, Cartesianizing the SCF, spin averaging,
  or dropping cross-spin blocks: changes the variational model.
- Numerical ECP/DF Hessians or production reference-engine calls: test oracles
  only, not the requested compiler-AD implementation.

## Invariants and resource boundaries

Every HVP shares one implicit nuclear response and all plan-declared sources;
ECP adds its two channels. Full matrices are raw scalar columns, never repaired
by symmetrization/projection. Occupations, quadrature topology, work floors,
VV10 active sets and the fitting active set are fixed for differentiation.

Direct integral and output budgets remain separate. DF reserves dense primal,
directional and cotangent sources and representation pullbacks before giving
the remaining allowance to TensorIR. It never stores a coordinate Jacobian or
fitted four-center tensor. ECP streams one radial layer with a separate numeric
reserve under the grid budget. Python metadata, compiler/library storage,
BLAS-private work and borrowed native providers retain their explicit exclusions.
Neither bounded storage nor GPU admission is an endpoint scaling claim.

## Evidence

- Native normalized p/d/f AO jets and bilinear cotangents preserve the public
  spherical map. Public spherical H2 HVPs agree with Cartesian results.
- HeH and H2+ doublet HVPs agree with three-step reconverged force differences,
  including a fully occupied spin space and an empty beta occupied block.
  An asymmetric H3 doublet additionally exercises nonempty OV blocks in both
  spins and their coupled response against the same three-step force oracle.
- Raw fitted RKS Hessians and fitted RKS/UKS HVPs are checked against native
  reconverged energy mixed differences; raw DF primitive gradient JVPs have
  independent displaced-primitive checks. Duplicate auxiliary shells exercise
  explicit rank-truncated rejection, not a direct fallback.
  Explicit Cartesian and spherical d-auxiliary records keep their own declared
  representation independently of the spherical orbital record and pass native
  fitted-Fock and reconverged mixed-energy checks.
- Synthetic Na-core/H ECP fixtures test both residual channels, effective
  charges, spin and fitting combinations against reconverged native total
  energies. These are mathematical fixtures, not physical sodium ECP data.
- Real CUDA qualification runs only under finite Slurm allocations on n1;
  the opt-in tests cover raw matrices, CPU parity and combined domains.
  Slurm job 8594 (RTX 5090, scheduler visibility 0) passed all five CUDA tests,
  including the combined UKS/DF/ECP/spherical case and a raw UKS matrix.
  Slurm job 8597 passed the additional asymmetric H3 CUDA/CPU/force-oracle
  gate with nonempty alpha and beta OV spaces, bringing CUDA coverage to six
  distinct endpoint tests.

The numerical gates verify derivatives of the declared discretized model,
not convergence of vibrational frequencies with basis or quadrature.

The focused CPU qualification run, repeated after the consistent native rebuild,
passed 184 tests, excluding the separately
qualified expensive asymmetric s/p water case. Its full combined ECP/DF/UKS
matrix is raw and passes symmetry and HVP consistency without repair. Native
qualification requires a consistent build: re-linking newly compiled native
adapters against stale objects is not sufficient CUDA provenance.
An incremental re-link of the previously built library produced CUDA illegal
memory accesses even in RHF. A consistent full native rebuild and recheck
resolved this on the same scheduler-visible GPU before accepting the five
CUDA endpoint tests; the failing partial build is not qualification evidence.

## Revisit when

The mixed CUDA placement recorded here is superseded by
[resident Hessian numerical execution](2026-10-15-wb97mv-resident-hessian.md).
The mathematical domain, approximation and acceptance decisions remain in force.

Add truncated-metric spectral Hessians, auxiliary shells beyond f, resident
grid/DF/ECP AD or shared multi-RHS response only with independent complete-
endpoint numerical and resource evidence. Never infer those from primitive
kernel availability alone.

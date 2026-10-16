# Decision: compiler-AD WB97M-V Hessians with implicit stationary response

Status: implemented
Date: 2026-10-15

The initial admission boundary recorded here is extended by
[the spin/representation/DF/ECP/CUDA decision](2026-10-15-wb97mv-hessian-domains.md).
The original derivative and fixed-branch rationale remains applicable.

## Problem

The native WB97M-V energy/force endpoint already composes meta-GGA, SR/LR
exchange and self-consistent VV10. The LDA/PBE Hessian owner cannot be reused
by changing its functional-name gate: that would omit kinetic-density, range
exchange and pair-coupled density response and produce an incomplete Hessian.
The user explicitly requires compiler automatic differentiation, not a
numerical Hessian assembled from displaced forces.

## Decision

Retain the installed method-neutral second-order executor, implicit CPKS and
stationary nuclear-response algebra. Add exact primitive second-order planning
rules for tau, SR/LR exchange and VV10. The molecular binding remains separately
qualified for canonical CPU direct Cartesian FP64 closed-shell WB97M-V.

Use TensorIR JVPs for AO features, with distinct left/right tangent namespaces
for mixed geometry-density derivatives. In particular the left geometry partial
has zero density tangent, whereas its right derivative includes the relaxed
density. Differentiating only AO values, or forgetting the tangent of the left
AO motion, omits terms even for a bilinear feature.

Use the shared scalar Graph for functional Hessians and VV10 pair JVPs. Keep
the VV10 primal under the existing nonlocal-pair owner; its denominator factors
are shared with the existing TensorIR native emitter. An ordered pair without
one-half is differentiated with respect to its first leg: symmetry supplies
the second energy leg. Energy integration still has one-half. Both pair legs
move in the JVP, and the beta term and weight/point partials remain explicit.
Hash the actual pair derivative DAG as well as the semilocal expression.
Keep this composition under `method.stationary_grid_second`, above the DFT
feature and XC owners. Its narrowly declared neutral `integral.expr` import
uses the existing shared scalar algebra, not integral recurrences or runtime
providers; the compiler structure audit continues to forbid those dependencies.

Preserve native Libxc worker coordinates, Cartesian coefficient conventions,
smooth LR expression and the fixed VV10 active set on both legs. A binding-time
Fock reconstruction audit prevents silently combining a response kernel with
a different SCF point model. Generated SR/LR second integrals use the same
geometry-factored nuclear chain rule as full Coulomb, but the native geometry
must receive the actual modified moments and fixed omega. Merely admitting a
range IR while retaining Full moments would silently compute the wrong physics.

## Rejected alternatives

- Displaced-force finite differences: useful independent acceptance oracles,
  but not the requested production algorithm or analytic second-order support.
- Differentiating SCF iterations: unnecessary iteration tapes and convergence-
  dependent derivatives instead of the existing implicit stationary equations.
- Full AO-rank-four weights or a coordinate-by-grid Hessian: unnecessary global
  materialization. Keep shell-local integral weights, scalar molecular columns,
  and bounded point/pair AD tiles.
- Promoting compiler rules to all public methods/backends: primitive algebra
  does not qualify a complete molecular endpoint.

## Invariants and limits

Every HVP has one shared nuclear response and all twelve plan-declared sources;
the installed executor enforces coverage before publication. The full matrix
is raw, without post-hoc symmetrization or projection. No PySCF/reference work
or finite differences are used in production. CPU qualification does not grant
CUDA, UKS, spherical, DF or ECP Hessians. The initial full matrix uses scalar
columns, not a claimed multi-RHS optimization. Separate numeric capacity and
ordered-pair work caps are mandatory; a smaller pair tile cannot reduce work.
The numeric reserve includes simultaneously retained reference and current
response partials, geometry fields and immutable-publication copies. TensorIR
feature evaluation receives the remaining explicit byte budget rather than its
larger default allowance. Cached grid partials are detached read-only arrays.

## Evidence

`test_composite_grid_second_order.py` checks independent real-space VV10
partials/JVPs, the two-leg quadrature factor and beta, signed weights, active
sets, work/capacity failures, nested AO feature derivatives and nuclear AD.
`test_rsh_second_derivatives.py` checks generated CPU SR/LR HVPs through f
against first-gradient differences with independent interval-quadrature moments.
`test_wb97mv_hessian.py` checks raw H2 all-column Hessians and asymmetric s/p
water response against reconverged complete-force differences at three steps.
The semilocal pinned-Libxc Hessian gates remain independent point oracles.

For the asymmetric STO-3G water fixture, the complete HVP's maximum absolute
errors against reconverged central force differences were 3.9114e-8, 4.3408e-9
and 3.9458e-10 at steps 3e-3, 1e-3 and 3e-4 Bohr, respectively. Its CPKS residual
was 3.1408e-15. These verify fixed-grid derivatives, not quadrature convergence.

Run with the repository Python path, a compatible native library, a verified
compiler-cache launcher and single-thread BLAS when reproducing small fixtures.
These are numerical and coverage gates, not endpoint performance claims.

## Revisit when

Add native generated pair/feature execution or shared multi-RHS response only
with complete endpoint evidence. Expand spin, representation, DF or CUDA
admission only after their complete response and raw-Hessian gates pass.

# Decision: first-order RHF AD through Gaussian integral parameters

Status: implemented
Date: 2026-10-23

## Problem

The Array API molecular operator previously composed a native energy/force
source with TensorIR algebra. That interface provides nuclear derivatives but
does not expose a differentiated integral computation or continuous basis
parameters. Attaching another framework's VJP to the same force result would
leave that scientific boundary unchanged.

## Decision

Add an explicit experimental `IntegralRHF` CPU FP64 owner and an opt-in
`MolecularModel(..., differentiation="integral-ad")` source. Coordinates,
Gaussian exponents and raw signed contraction coefficients enter one generated
integral/normalization computation. The primal S, h and ordered chemists' ERIs
are passed to a value-only `PreparedFockPlan` and the existing native RHF solver.
This plan never matches a system-only or ordinary prepared-plan cache lookup;
external tensor provenance is part of its identity.
ECP and spherical metadata are rejected before basis expansion; reconstructing
runtime primitive records must not erase the requested Hamiltonian or AO
representation and accidentally admit unsupported science.

The compiler differentiates the stationary RHF Lagrangian

    L = D:h - W:S + sum_abcd eri_abcd (D_ab D_cd / 2 - D_ac D_bd / 4)

with the shared TensorIR transpose transform. D is spin summed and W is the
physical weighted density D F[D] D / 2, evaluated at the returned determinant.
The native physical-weighted-density helper is shared with the existing force
consumer. Neither an extrapolated DIIS Fock nor old orbital energies authorize
this overlap cotangent. A fresh F[D] commutator check gates publication.
Nuclear repulsion has unit cotangent. The stationary rule is only first order
on a smooth converged branch; it is not a density Hessian or global-minimum
certificate.

Primitive partials use the existing scalar Graph derivative rules and canonical
S/T/V/ERI value owners, including the Boys argument's dependence on exponents.
The shared Boys implementation admits order 17 for this first-order through-g
algebra; existing lower-order consumers retain their emitted bound. Contracted
weights are differentiated through primitive and contraction normalization.
Zero coefficients use product-except-slot, not division by a coefficient.
Repeated mathematical centers scatter to the same physical coordinate.
The full ordered ERI layout avoids inferred symmetry factors and permits
arbitrary nonsymmetric source cotangents in `integral_vjp`.

Runtime contraction/allocation belongs to `src/integrals`; mathematical source
emission belongs to the scientific compiler. The runtime template is packaged
as a compiler asset, included in artifact identity and public provenance. Native
SCF execution belongs to the shared solver. Array API first-order jets capture
all continuous input snapshots and compose outer chain rules; they call no
preexisting force endpoint. Configuration is immutable so public shape fields
cannot invalidate generated native buffer extents.

## Bounds and rejected alternatives

Retain an explicit dense primal ERI endpoint with a numeric array budget and a
separate primitive evaluation bound. Stream primitive pullbacks rather than
materializing a coordinate-by-ERI Jacobian. Compiler work is separately charged
before symbolic interning and native compilation has a finite timeout. This is
a bounded correctness endpoint, not a large-basis performance claim. Dense
n^4 storage, ordered primitive work and large high-angular source expansion
remain real limitations.

A mixed `gggg` parameter kernel exposed recursive Graph differentiation and
postorder traversal limits. Both traversals now use explicit stacks; existing
rules and operand ordering are retained. Offline Libxc catalog regeneration
produces unchanged products and refreshes only the scalar-owner source digest.
However, fully expanded `gggg` value/partials still produced over one million
C++ lines: an -O1 compile failed and an -O0 investigation hit its 180 s timeout.
Do not label that endpoint qualified or infer large-basis readiness from the
through-g algebra. Native primitive qualification uses mixed lower-total-angular
ERIs including `(gg|ss)` and g/g one-electron cases, plus a separately compiled
Boys F0..F17 gate; large DAGs also have a symbolic budget rejection test.

Rejected alternatives are a force callback advertised as integral AD, numerical
energy differentiation in production, differentiating a frozen density without
the overlap multiplier, and unrolling a particular SCF iteration history.
Each either leaves the requested integral derivative absent or changes the
scientific derivative contract. No GPU/DF/KS/post-HF/higher-order claim is added.

## Evidence and acceptance gates

CPU Release build with CUDA disabled, main Python 3.13, ccache 4.5.1 and pinned
PySCF 2.14.0. All native compilation uses the existing compiler cache. Tests use
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and the built shared library.

- `test_integral_autodiff.py`: independent libcint S/h/ERI/nuclear values and
  reconverged PySCF energy differences for geometry, exponents and coefficients;
  three-step first-derivative convergence (final absolute error below 2e-7 for
  the two-center parameter directions), multicenter p-shell combined direction
  within 3e-6, signed/zero coefficients, scale/translation null directions,
  nonsymmetric integral cotangents, invalid domains, byte/work preflight,
  nonconvergence, lifetime and prohibition of Calculator endpoint delegation.
- `test_integral_parameter_ad.py`: emitted d/f/g one-electron and mixed ERI
  primitives against independent libcint; separate exponent/center directional
  differences within 3e-7 absolute/relative. Compiled F0..F17 moments agree with
  independent hypergeometric values at relative tolerance 3e-13, including the
  recurrence boundary at T=20.
- `test_array_api_integral_ad.py`: nonlinear energy loss, transformed coordinates
  and logarithmic exponents, joint basis/geometry AD, JVP and VJP, changed basis
  inputs, immutable replay after mutation, capture without native execution and
  explicit control/domain rejection.
- Existing Array API molecular, scalar Graph, rational identity and branch
  coalescing regressions; generated-source registry, ownership/API contracts,
  pinned compiler type check and Sphinx warnings-as-errors.

Raw local execution logs are retained under the ignored
`.artifacts/autodiff/` directory. No GPU qualification was performed.

## Revisit when

A generated integral schedule can bound source growth and full endpoint work
for larger angular bases, or a qualified implicit response rule supports second
order and additional electronic methods. Preserve explicit provenance, raw
normalization derivatives, independent oracle gates and failure-before-gradient
publication when extending this path.

## References

- `docs/user/experimental_array_api.md`: current integral AD usage and contracts.
- `python/generativeqc_compiler/integral/parameter_ad.py`: canonical integral AD.
- `python/generativeqc_compiler/method/rhf_integral_ad.py`: stationary source VJP.

# Decision: Native compiler-AD residual ECP nuclear source

Status: implemented
Date: 2026-10-16

## Problem

The raw DF migration left residual local/projector ECP response in the Python
stationary consumer. Effective ionic charges alone do not include that residual
operator. Moving an ECP potential center also changes every AO argument, not
only AOs whose basis centers coincide with the potential. A component cannot
claim an accepted molecular state through matching detached pointer labels.

## Decision

Freeze the incumbent coarse (160 radial, 32 polar) and refined (224 radial,
44 polar) quadratures during source generation. NumPy supplies fixed Legendre
constants only at build time; existing integral scalar graphs retain radial
mapping, real s/p/d/f harmonics, potential powers, projection products and the
two-grid admission policy. The 2.7 MB constants header remains an ignored build
artifact. No root solving, Python or compiler package is needed at molecular
runtime. This preserves the quadrature rule and mathematical policy, not the
exact bits of the old runtime Newton solver's roots.

Keep quadrature in `integral.ecp_fixed_grid`, not the method compiler: the latter
must not import integral equations/policy. Compose the existing stationary AO
energy bilinear with fixed measures, then apply TensorIR transpose/linearize
for physical matrices, geometry directions, AO cotangents and joint D/AO
cotangent directions. Reuse the existing AO Taylor graph's value-motion roots,
requiring only ten spatial jets through second order. Do not write a native
ECP product rule or copy a derivative recurrence.

Stream each potential-center radial layer once for each grid. Collocate AO
jets once, cache all sixteen harmonic projections, and reuse them for local
angular rows, nonlocal projector rows and every left physical coordinate.
Relative displacements are potential-center minus basis-center displacements;
indicator masks preserve both physical incidences, including coincidences.
Compiler weighted-contraction leaves combine AO cotangents and mixed Taylor
motion without a coordinate-major AO-pair derivative tensor. Right cotangent
motion includes signed dD. The Coulomb tail stays in the independent effective
charge attraction and nuclear-repulsion sources.

The accepted-state adapter borrows the actual immutable normalized grid basis,
physical spin densities and canonical frame. It packs actual normalized
primary-Fock ECP terms once and uses a separate generated physical spin-sum
map for D/dD, without occupancy/half factors. Its own arena includes every term,
sum, status and map scratch byte. The sum and generic source have separate
generated CUDA headers because non-inline CUDA kernel definitions cannot be
included in both execution translation units.

Authenticate the entire WB97M-V Direct/DF model, canonical frame, native source
identity and original noncapturing stream at preparation, execution and before
publication. Protect immutable basis/grid/density/frame/H/F storage against
output aliases. Every physical spin direction must be symmetric individually;
opposite antisymmetric errors must not disappear under the total-spin sum.
Do not borrow or mutate ordinary J/K or occupied-factor scratch.

## Admission and work

Bound total ECP terms, resident bytes, AO evaluations, literal generated affine
visits and conservative native reductions. Receipts count every submitted
radial layer, point/AO evaluation, harmonic projection summand and generated
affine term. Left-coordinate reductions reuse cached jets; memory boundedness
must not conceal repeated source work. These counts are not FLOPs or endpoint
performance measurements.

Stage local/nonlocal matrices and directions or gradients/HVPs jointly for both
grids. Retain the incumbent 2e-9 matrix and 2e-8 derivative absolute coarse/fine
gates, plus finite and output-matrix symmetry admission. Reauthenticate before
all four publication copies. Ordinary admission, numerical and revocation
failures preserve every output; a CUDA publication fault may leave partial
copies.

Initialize the entire generic arena at preparation. A numerical-abort flag may
skip generated node writers while later generated finite-input gates still
inspect private buffers. Always initialize submitted point records even after
an earlier numerical failure, because the linked AO leaf reports errors but
does not consume that abort flag. Neither rule permits publishing rejected
numerics.

## Rejected alternatives

- Generate quadrature or import compiler/Python at runtime: violates the native
  numerical runtime contract despite native CUDA matrix kernels.
- Differentiate only AO centers, or recover all potential motion from a final
  translation sum rule: loses individual physical-center derivatives.
- Recompute AO/projector work for every left coordinate: bounded scratch would
  hide a coordinate-sized source-work multiplier.
- Apply native hand-coded ECP product rules: duplicates the compiler AD and
  makes moving-density/normalization conventions harder to audit.
- Use CPU/PySCF ECP values as a production fallback: silently changes the
  requested runtime/backend contract. Nonconvergent quadrature fails closed.
- Call an arbitrary matching synthetic callback an accepted KS proof: actual
  canonical frame, model and producer rebinding are mandatory.

## Independent qualification

`test_ecp_response_compiler.py` compares signed layer matrices, AO VJPs/JVPs,
Taylor mixed motion and physical spin sums against independent NumPy algebra.
It also checks angular normalization/projector orthonormality, a known radial
integral, deterministic generation and rejection of public-runtime/GPU/oracle
imports during generation.

`test_native_ecp_nuclear_response.py` uses separate local/projector libcint
operators, independently normalized Cartesian basis columns and independently
applied real-spherical transforms. Its analytic physical gradient retains both
AO and potential-center integrals; displaced energies validate that oracle.
Two steps of displaced analytic moving-density gradients validate native HVPs.
Displacements and PySCF run only in the test driver, never in production.

Raw fixtures retain contracted signed radial coefficients, local and s/p/d/f
projectors, powers zero through four, Cartesian/real-spherical orbitals through
f and multiple ECP centers. Accepted WB97M-V fixtures cover RKS/UKS Direct/DF
with s/p Cartesian/real-spherical orbitals and all residual projector channels.
Native probes disable executable/Python paths, check deterministic replay and
ordinary induced-Fock preservation, and re-solve the real KS owner to prove
revocation without publication. A tightly localized off-center AO deliberately
fails coarse/fine admission and preserves all matrix/vector outputs.

Local evidence lives under
`.artifacts/wb97mv-hessian-aot-20261015/ecp-*`; remote source-matched runs use
`n2:/data/jzzeng/wb97mv-hessian-native-response-20261015/` under finite Slurm
allocations. Keep terminal outcomes and binary checksums alongside those logs;
an allocated/running test is not evidence of a pass.

The final source-matched checkpoint passed all 24 GPU tests in Slurm 3139:
six independent raw-domain cases, nine malformed source/action cases, one
deliberate numerical-refinement failure and eight accepted-owner combinations.
Local broad component/compiler regressions passed 145 tests with 321 explicit
optional-native/oracle skips. Both DF/ECP host admission CTests, pinned ty,
562-module compiler dependency checks, the 396-file CUDA ownership inventory
and all-files pre-commit checks passed. Slurm sanitizer runs and their terminal
summaries remain separate evidence, not inferred from these numerical passes.

## Consequences and remaining scope

This supplies the residual numerical source, not a stationary Hessian. The
production CLI still invokes the Python/CuPy worker. Full practical ERI/DF
angular closure, transactional stationary CPKS/HVP/Hessian assembly, native CLI
wiring, full endpoint qualification without Python and matched GPU4PySCF
endpoint benchmarks remain outstanding. No endpoint speed claim follows from
these component tests or sanitizer durations.

Revisit the fixed-quadrature storage/scheduling when representative complete
endpoints justify another generated quadrature or source reuse policy, while
preserving independent numerical and explicit work/resource acceptance gates.

## References

- `docs/developer/dft_hessian.md`
- `.agents/notes/implemented/numerics/2026-10-16-native-raw-df-nuclear-response.md`
- PR #2527

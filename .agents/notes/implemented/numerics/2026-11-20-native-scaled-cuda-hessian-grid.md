# Decision: reuse the scaled SCF point model in native CUDA Hessians

Status: implemented
Date: 2026-11-20

## Problem

The native stationary CUDA Hessian owner coupled its grid lifetime and
TensorIR feature maps to WB97M-V's scalar policy and VV10. LDA/PBE have a
different vacuum and spin interpolation policy. Reusing the WB97M-V floors,
switching to its unscaled symbolic graph, or delegating to the CPU would not
differentiate the converged physical CUDA Fock model.

## Decision

Bind scaled LDA/PBE value and directional point evaluators to the existing
native AO/feature maps. The restricted evaluator uses total-density directions,
while UKS retains separate physical spin directions. The moving measure adds
the weight derivative times the base potential to the directional Fock
coefficients; the contraction for the left weight motion receives the energy
density tangent. No tau or explicit position cotangent is included. Unused
VV10 discovery, workspace, and pair work are absent in these branches.

Consume the same authenticated converged model for both fixed and moving
grids. In the pure Direct candidate, reuse the compiler's Coulomb-only
quartet source for nuclear response and omit both exact exchange actions.
The native canonical Fock comparison, resource bounds, CPKS and raw matrix
publication gates are retained. The separate WB97M-V model and DF/ECP source
admission remain unchanged. A grid scale is not sufficient to admit a
corresponding hybrid molecular response.

The full-range hybrid candidate accepts canonical PBE0 by checking both the
semilocal exchange scale and the exact-exchange coefficient of the live
Fock identity. The induced Fock binds the signed coefficient already owned
by that Fock spec. The molecular nuclear path converts it back to the
dimensionless exchange fraction and uses the compiler-generated Coulomb plus unit
full-range exchange quartet graph with that fraction as a constant parameter.
This preserves the shared RKS/UKS spin factors without a second implementation
of the exchange weight or a WB97M-V coefficient rescaling.

The generated moving-grid mixed-feature program has no left or mixed density
input: it is the derivative of a fixed-density left AO partial. Update the
native call to that actual generated signature; do not reintroduce a zero
input to hide the build mismatch.

## Validation boundary

An sm_120 probe compiled using CUDA 12.9 and sccache 0.16.0 ran through a
finite Slurm allocation on n4 (RTX 5090). The fixed/moving-grid probe suites
passed 80 cases (10 skipped for separately linked molecular bindings or an
optional empty-spin oracle). Six new cases compare LDA and canonical/scaled
PBE for RKS/UKS against Libxc values and finite differences for fixed-grid
potential, moving Fock and mixed scalar response. Semilocal cases run with a
zero pair allowance. These are grid component checks, not public endpoint
qualification or timings. Molecular closure requires the independent displaced
SCF oracle and a linked AOT package containing every integral source.
Compiled quartet tests independently check energy, both Fock incidences, the
integral weight and its density response at several full-range fractions;
their passing results also do not qualify a complete molecular endpoint.

The full-range selector exposed a hard-coded five-selector quartet closure
table. Size and index that table from the generated radial selector count and
the validated shell-angular domain. The corrected sm_120 test passed all 28
independent s/p libcint checks. An expanded d-shell inventory also passed 14
checks including four-center permutations and exact-budget fallbacks. PBE0
RKS and B3LYP UKS with d orbitals passed the full matrix and HVP checks in
both Direct and DF. For d-shell B3LYP Direct, finer `0.002`/`0.001` Bohr
stencils recover the same matrix tolerance; a sweep to `0.0005` Bohr
distinguished stencil truncation from native response error.

The six Direct LDA/PBE/PBE0 molecular fixtures (both spins) passed independent
reconverged libcint/Libxc energy stencils, raw symmetry, translation and HVP
agreement. The combined molecular/J/K run passed 11 cases, including WB97M-V
H2. The public Python and compiled C++ CLI each passed six CUDA endpoint
checks covering the same methods and spin layouts on RTX 5090 through Slurm.
That initial evidence qualified only canonical Direct all-electron FP64
models. No endpoint speed or larger-basis claim is inferred from the small
fixtures.

## B3LYP production branch

B3LYP uses its existing production-tail energy and polarized potential graph,
not a new Libxc runtime or an LDA/PBE point approximation. Lower the potential's
directional derivative with both local gradient Gram motion and the derivative
of its pullback. At an exactly empty spin, only zero inactive density and
gradient directions are admitted. Restrict those features before applying AD;
the normal exchange curvature is singular even though the tangent action of
the finite potential exists. Prune the inactive B88 branch before power
constant folding so it cannot evaluate an inverse power of zero. No density
floor, finite-difference response or Hessian materialization is needed.

The native grid run passed 53 cases (four optional molecular-grid cases
skipped), including B3LYP RKS/UKS with independent fixed and moving Libxc
differences. Two complete B3LYP molecular cases passed the reconverged
energy, symmetry, translation and `H @ v` gates. These are small-basis tests.

## Fitted primary and residual ECP

Use the live Fock specification's full-range exchange fraction in the DF
energy and response graphs. A pure functional emits a J-only graph. If the
fitted primary is the whole two-electron operator, do not prepare exact
quartets or their optional frozen response columns. WB97M-V still owns the
separate exact long-range source. Full rank and cutoff separation are
required; substituting exact integrals would change the differentiated
energy. RKS RI-J now uses the existing prepublication canonical-state probe:
a small DIIS density change alone did not certify the unshifted physical
frame in the PBE public fixture. No gate tolerance changed.

Eight all-electron fitted and eight Direct scalar-ECP matrices on RTX 5090
passed independent reconverged libcint/Libxc energies, raw symmetry,
translation and HVP agreement. Direct ECP includes effective-charge core
terms plus the same residual local and nonlocal projectors as SCF. The 32
public all-electron checks covered both integral paths, both spin layouts,
all four models and both frontends. The fitted CUDA probe passed 24 checks at
zero, WB97M-V and PBE0 exchange fractions, including malformed-source rejection.
Eight public Direct ECP cases compared CUDA HVPs from both frontends and
verified native ECP provenance and the translation null mode.
The combined DF/ECP model retains the same residual projectors and
effective-charge terms without changing the fitted two-electron energy model.
Eight combined matrices (four functionals, both spin layouts) passed the
independent reconverged-energy, raw symmetry, translation and HVP gates, and
eight public HVP comparisons confirmed both CUDA provenance sources. The
combined B3LYP UKS case also published the raw full matrix through both public
frontends. These are small-inventory results, not performance qualification.

Current architecture and admission boundaries are in
[`docs/developer/dft_hessian.md`](../../../../docs/developer/dft_hessian.md).

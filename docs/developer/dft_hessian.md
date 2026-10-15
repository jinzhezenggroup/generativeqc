# DFT Hessian and HVP execution contract

This page describes second-order MethodIR source rules and the
stationary executor, including the bounded public molecular
endpoint. An available functional or first derivative never implies
a qualified second-order method/backend combination.
The [RHF term map](hessian.md) remains the conventional
Hartree–Fock derivation; [Reference](../reference/capabilities.md)
is the user-facing source for capability admission.

## Stationary DFT plan and molecular execution

DFT Hessian source planning comes from resolved `MethodIR` primitive
capabilities, not functional-name branches. `StationaryHVPPlan` admits
compiler-only direct all-electron FP64 LDA/GGA/meta-GGA RKS/UKS with the native SCF
point model and composes full-range or range-separated exchange and VV10. Semilocal families share
one-electron, Coulomb, XC AO/grid/partition, overlap/Pulay and nuclear
sources, while their `rho`/`sigma` features come from the method graph.
Global hybrids add one exchange source without PBE0/B3LYP-specific code.

The planner binds the shared #179 CPKS contract, #161/#236 XC feature-Hessian
action and #178 weighted second-integral HVP contract. Its integral blocks
execute only the bounded source algebra; native CPKS and XC/grid consumers
remain separate owners. Second-order support is now admitted through an exact
primitive-type rule registry rather than a fixed "one semilocal primitive"
condition. Each rule declares its required derivative capabilities, supported
ingredients, directional sources and any response inputs needed by bounded
integral HVP lowering. The plan derives its source inventory by composing those
rules with the stationary mean-field envelope. It fails closed when an active
primitive has no registered second-order rule. The full-range exchange rule reuses the existing
ordered-quartet stationary-gradient weight and its TensorIR JVP with both
same-spin density response inputs. It requires first-integral directions
and generated weighted second-integral HVPs. Range-separated rules preserve each
operator's immutable omega and generate separate SR/LR density-weight JVPs.
The meta-GGA rule includes `tau`; the VV10 rule declares both pair legs' feature,
point and quadrature-weight response. The ECP envelope adds independent local
and projector sources. Full-rank DF uses compiler-generated three-center/metric
cotangents and their response JVPs. rVV10 remains unsupported at this planning
boundary. A compiled source rule does not qualify a molecular endpoint.

Molecular execution is a separate method-neutral layer. Its canonical
installed owner is now `generativeqc.second_order`; the former
`tools.generativeqc_hessian.stationary_executor` module is a compatibility re-export
and contains no second implementation. `StationarySecondOrderExecutor` accepts
only a plan identity and complete ordered source inventory, one perturbation
provider, one opaque stationary response driver and exactly one contributor per
declared source. It performs one response solve, passes the same response object
to every contribution and publishes a result only after every source returns a
finite Cartesian HVP. There is no HF/RKS/UKS or functional-name dispatch in
this executor. A MethodIR-derived DFT plan and an HF second-order plan can
therefore share the same orchestration while retaining their own perturbation,
response and primitive providers. Missing or extra contributors fail before
execution.

The plan now exposes bounded `integral_block` programs for the one-electron,
Coulomb, overlap/Pulay and optional full-range exchange sources. Each block
reuses the stationary-gradient
source energy and derives its fixed integral weights, then generates an exact
TensorIR JVP for a supplied
density or weighted-density response, and contracts that response with the
first-integral directional tile plus #178's fixed-weight second-integral HVP
vector. This is an executable source-level algebra slice for both RKS and UKS;
native shell/center recovery, shared CPKS execution, XC/grid/partition motion
and molecular assembly remain owned by their qualified consumers. In particular,
compiled PBE0/B3LYP exchange HVP weights do **not** qualify hybrid molecular
Hessians: the native mixed K/XC CPKS provider, metric/Pulay relaxation and
complete molecular source inventory still need independent numerical acceptance.

The shared closed-shell CPKS operator now accepts an explicit physical
full-range exact-exchange fraction for CPU-side algebraic composition. It binds
the resulting restricted spin-summed -a/2 raw K action into the response
identity and **refuses nonzero exchange unless the backend explicitly declares
raw full-range K support**. Semilocal a=0 keeps the same identity and behavior.
This is a provider-checked, synthetic-oracle-tested response contract only;
`NativeRKSResponse` remains restricted to native LDA/PBE, and no native PBE0,
B3LYP, user-composed or CUDA hybrid molecular Hessian endpoint is promoted.

The closed-shell nuclear-perturbation consumer follows the same rule. Its
canonical installed owner is now `generativeqc.stationary_nuclear`: metric-density
RHS construction, occupied-orbital/density reconstruction and the single-/multi-
RHS consumer contract live there without importing repository `tools.*`.
Method-specific Fock physics remains injected through the response operator's
`induced_fock(delta_density)` contract. The shared #179 GMRES implementation
is now installed as `generativeqc.response_solver`; tools response/Hessian
imports are compatibility aliases or wrappers around the same production
objects. Existing `solve_rhf_nuclear_perturbation[s]` names remain strict RHF
compatibility wrappers.

The CPU direct all-electron Cartesian LDA/PBE RKS path now has installed
production owners for the live CPKS state/J/XC binding, nuclear-direction
response, NativeAO-owned generated first/second integral response, complete
seven-source molecular HVP, shared multi-RHS CPKS and raw bounded full-Hessian
assembly. The existing finite-difference, raw-symmetry and resource-admission
tests continue to qualify this same scientific slice. Repository
`tools.generativeqc_hessian.rks_directional` and `rks_molecular` are module
aliases to those installed consumers. The same owner is now exposed through
`Calculator.hessian_vector_product()` and `Calculator.hessian()` for CPU direct
all-electron Cartesian strict-FP64 closed-shell LDA/PBE RKS. Public calls retain
explicit integral/output budgets and report the executed public endpoint in
result diagnostics. WB97M-V uses the separate compiler-AD owner described below.

## Complete CPU endpoint benchmark and cold-JIT qualification

Run the committed device-free driver through the *public* Calculator methods:

```bash
PYTHONPATH=python:. python tools/benchmark_cpu_hessian.py \\
  --method pbe-rks --case h2 --repeats 3 \\
  --output benchmarks/results/hessian-cpu-h2.json
```

The JSON retains every full-call endpoint-first, same-geometry, and
changed-geometry timing sample for both HVP and raw full-Hessian paths. Each
full call includes fresh SCF and stationary response. Every timed pair, including
all same-geometry repeats, must pass finite real shape and `H @ v` parity checks;
raw full-matrix asymmetry is recorded without post-hoc symmetrization.
No CUDA acceptance is required. HVP precedes full Hessian in each pair in one
shared process/cache, and the receipt records that ordering. In particular,
the first full-Hessian call has already been warmed by one HVP; it is **not** an
independent process-first trial. Neither endpoint-first sample establishes a
compiler-cold measurement. Use separate fresh processes *and* independently
empty cache directories for independent compiler-cold qualification.
For large-domain compilation investigations, opt in with `--case he13`:
this uses the 13-Cartesian-AO single-He `s+d+d` fixture from the existing
resource-bound test. Its generated second-integral CPU C++ cold-JIT path has
previously exceeded the 300-second compiler limit (#2272), so this is an
explicit expensive diagnostic, not part of routine PR smoke tests. Do not infer
that passing the smaller benchmark fixes that cold-compile blocker.

## WB97M-V compiler-AD molecular execution

`generativeqc.composite_rks_hessian` binds the complete twelve-source WB97M-V
RKS/UKS plan to one converged native CPU or CUDA state. It reuses
`StationarySecondOrderExecutor` for CPU transactional source coverage and the
installed implicit CPKS nuclear-response solver. CUDA dispatches to the resident
owner in `generativeqc.cuda_composite_hessian`, with equivalent explicit source
coverage and a device-coupled CPKS solve. Neither path differentiates SCF
iterations, reconverges displaced geometries, or calls a reference engine.

`generativeqc_compiler.method.stationary_grid_second` generates the AO `rho`, Cartesian
density gradient and `tau` JVPs and mixed JVPs with TensorIR. The scalar Graph
generates the semilocal feature Hessian, weighted VV10 pair gradient/JVP and
nuclear-repulsion HVP. The shared nonlocal-pair owner supplies the VV10 primal;
both moving pair legs, local scales, beta term and quadrature measures are
included. The semilocal owner supplies the pinned WB97M-V work thresholds and
smooth long-range expression. Worker floors and the `rho >= 1e-8` VV10 active
set retain the native fixed-branch convention. These are local derivatives,
not derivatives of a discontinuous change of grid or active set.

The nuclear RHS includes J, both range-exchange operators, semilocal XC,
self-consistent VV10, AO/grid motion and Becke weights. The same full
interaction map enters every CPKS action, including the metric-density RHS.
The binding audits its reconstructed Fock against the native SCF Fock before
response. Density and energy-weighted-density relaxation feed the generated
integral-weight JVPs and overlap/Pulay source exactly once. SR/LR second
integrals use the existing geometry-factored integral AD with modified Boys
moments and fixed omega, not a full-Coulomb substitution.

Public `Calculator.hessian_vector_product()` and `Calculator.hessian()` admit
the canonical WB97M-V graph with RKS or UKS, CPU or CUDA, direct or full-rank
DF, all-electron or scalar semilocal ECP Hamiltonians, and Cartesian or real
spherical AOs through f in strict FP64. Combinations use the same stationary
endpoint. The full-matrix owner uses scalar columns
(`block_size=None` or `1`, default `strategy="recycled"`); this does not claim
multi-RHS recycling. The raw matrix is neither symmetrized nor projected.
`grid_budget_bytes` bounds numeric AO/grid and AD tile workspace;
`max_nonlocal_pairs` separately bounds the ordered active pairs **per grid
action**, not the complete Hessian's work. Default limits are 128 MiB and
16,777,216 pairs. Integral and output/publication budgets remain independent.
Compiler processes, Python object metadata, BLAS-private workspace and borrowed
SCF/native-provider storage are outside those grid bounds. A fine/default grid
can exceed the pair cap; callers must choose and converge their quadrature and
explicitly increase the cap if appropriate. Reducing a tile does not reduce
quadratic pair work. This is a bounded implementation, not a scaling claim.

### Representation, spin and fitting

`generativeqc.hessian_basis` reads native normalized AO expansion records
to construct a fixed Cartesian/public map. Integral densities and cotangents
are pulled back to Cartesian derivative workspaces; operators are projected
back. SCF, quadrature and CPKS remain in the original spherical variational
space. No angular map is fitted from grid values.

`generativeqc.spin_stationary` solves coupled alpha/beta CPKS, retaining
cross-spin Coulomb and functional kernels, 1/0 spin occupations, and empty
occupied/virtual blocks. The nuclear RHS includes the canonical metric
connection with `(epsilon_a + epsilon_i)/2`; metric density and
energy-weighted-density response enter every nuclear column.

The prepared RSH-DF approximation is **fitted full-range J/K with short fraction
0.15, plus exact LR exchange with fraction 0.85**, not separately fitted SR
exchange. The reported SR source is fitted full-range exchange minus its exact
LR part; the LR source retains fraction 1. `integral.df_second` differentiates
the existing physical two-/three-center Hermite/Boys primals. TensorIR VJP/JVP
programs in `method.stationary_df_second` generate energy cotangents, Fock
response and full-rank inverse-metric response. A second native/AD Fock audit
checks the fitted primal before executing nuclear response.

DF keeps bounded dense three-center and metric sources, never a global
coordinate Jacobian or a fitted four-center tensor. `integral_budget_bytes`
also bounds these sources and remaining TensorIR execution workspace;
compiler/library/metadata and borrowed native-provider storage remain separate.
The auxiliary AO adapter currently supports through f. Rank-truncated metrics
and metrics within 10% of the native relative cutoff fail explicitly: freezing
retained eigenvectors or silently using exact integrals is not permitted.

### ECP and CUDA ownership

The C++ executable also provides explicit `hessian`/`hvp` compiler-runtime
commands. Their interpreter dependency, protocol, controls and examples are
owned by [the native CLI guide](../user/native_cli.md#compiler-ad-wb97m-v-hessian-and-hvp).

`generativeqc.ecp_hessian` differentiates the actual residual local potential
and nonlocal harmonic projectors with TensorIR mixed AO/density JVPs. Both
AO-center and ECP-center motion are included. One radial layer is streamed,
using the native coarse/fine ECP quadrature policy; every HVP must pass the
retained absolute coarse/fine gate. Effective nuclear charges are used in
attraction and nuclear repulsion, without subtracting the ECP tail twice.

CUDA uses a **device-resident numerical endpoint**, requiring the optional
`generativeqc[cuda-hessian]` dependency. The converged native snapshot is uploaded
once. AO jets, moving Becke partitions, scalar functional/VV10 AD, first and
second direct/DF integral AD, spherical pullbacks, ECP projectors, CPKS vectors,
the Hessenberg least-squares problem and D/W reconstruction remain on one CUDA
stream. Scientific derivatives are generated from the existing scalar Graph and
TensorIR, not CuPy differentiation. CuPy supplies resident storage and CUDA
linear algebra. The compiler package never imports CuPy during generation.
The native SCF preparation contract is unchanged: the resident contract covers
the Hessian/response numerical stage beginning at the converged snapshot, not
an assertion that native input/pre-SCF preparation has no host numerical work.

Host code enumerates fixed shells/layouts, constructs fixed quadrature constants,
compiles kernels, validates provenance and makes scalar stopping decisions.
Discrete branch metadata may be downloaded; intermediate density, Fock, AO and
derivative panels are not. A full matrix retains its columns on the device until
publication. Diagnostics report `execution_residency="device-resident"`, CUDA
placement for every derivative provider, source/primitive work and pair counts.
The endpoint never silently substitutes the old host AD or changes the declared
fitting/Hamiltonian approximation. Missing dependencies, resource violations,
response failure and quadrature failure raise before publication.

Primal exact ERIs may use explicitly reserved resident dense storage to reuse
them across CPKS actions. When that reserve does not fit, a bounded device-only
shell-streamed J/K action is selected instead. Integral derivatives always stream
at most 16 components and 64 radial products, with shell-local compiler-generated
stationary weights; no coordinate Jacobian or derivative ERI tensor is retained.
This conservative schedule is not a performance/scaling claim. Numerical arrays
use an endpoint-owned CUDA memory pool with an aggregate cap derived from the
integral, grid, response and publication budgets; native SCF/provider storage,
CUDA modules, compiler processes and linear-algebra handles remain separate.

Tests in `tests/python/test_wb97mv_hessian.py` check a complete raw H2 Hessian,
all-column reconverged-force differences, translation/symmetry and an asymmetric
s/p water HVP with orbital relaxation. Compiler tests independently check VV10
against the real-space reference, nested feature JVPs, nuclear pairs, and native
SR/LR second integrals through f against interval-quadrature gradient oracles.
Additional gates in `test_wb97mv_uks_hessian.py`, `test_hessian_basis.py`,
`test_wb97mv_df_hessian.py`, `test_wb97mv_ecp_hessian.py`, and the Slurm-only
`test_wb97mv_hessian_cuda.py` cover coupled spin, representation maps, fitted
raw matrices, ECP channels, reconverged energy/force differences and combined
CUDA execution. Finite differences and reference evaluators are test oracles only.

## Qualified execution boundary

LDA/PBE retain their CPU direct all-electron Cartesian closed-shell domain.
Canonical WB97M-V additionally admits the spin, representation, backend,
fitting and ECP domains described here, subject to their resource, full-rank,
quadrature and fixed-branch gates. Other functional families, truncated DF
metrics and mixed precision are not inferred
from generated component kernels.

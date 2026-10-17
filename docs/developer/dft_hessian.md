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

The C++ CLI and public CUDA `Calculator.hessian*` numerical path use the linked
native AOT stationary owner described below. The CLI needs no Python runtime.
The Python frontend only validates/marshals inputs, prepares the native SCF batch
and publishes the detached final result; it needs neither CuPy nor a runtime
compiler for this numerical path. A library without the linked AOT package
fails before SCF preparation, rather than selecting the legacy Python owner.

`generativeqc.composite_rks_hessian` binds the complete twelve-source WB97M-V
RKS/UKS plan to one converged native state for CPU/development consumers. It reuses
`StationarySecondOrderExecutor` for CPU transactional source coverage and the
installed implicit CPKS nuclear-response solver. Its low-level CUDA development
owner in `generativeqc.cuda_composite_hessian` is not the production public CUDA
route. The public route enters `generativeqc._native_hessian`, which calls
`generativeqc_ks_hessian_cuda_v1` without creating/exporting a KS snapshot or
constructing Python AO/CPKS providers. None of these paths differentiates SCF
iterations, reconverges displaced geometries, or calls a reference engine.

The independent GPU4PySCF comparator configures its SCF, NLC and CPHF response
with the same moving atomic quadrature. Setting only `grids` and `nlcgrids`
leaves a separate default SG1 CPHF grid and does not compare derivatives of the
stated discretized model. Reference receipts record the response-grid policy
and actual point count. The full raw-matrix acceptance gate remains unchanged;
matching a quadrature does not establish chemical grid convergence.

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
(`block_size=None` or `1`). The default `strategy=None` resolves to native
`"scalar"` on CUDA and `"recycled"` on CPU. Explicit recycling/block requests
on CUDA fail rather than mislabeling independent solves. The raw matrix is
neither symmetrized nor projected. Native results expose immutable `value` or
`matrix` arrays and detached scalar/work diagnostics; source-component panels
are aggregate-only and stay on device. CPU/development result types are unchanged.
`grid_budget_bytes` bounds numeric AO/grid and AD tile workspace;
`max_nonlocal_pairs` separately bounds the ordered active pairs **per grid
action**, not the complete Hessian's work. Default limits are 128 MiB and
16,777,216 pairs. Integral and output/publication budgets remain independent.
Compiler processes, Python object metadata, BLAS-private workspace and borrowed
SCF/native-provider storage are outside those grid bounds. A fine/default grid
can exceed the pair cap; callers must choose and converge their quadrature and
explicitly increase the cap if appropriate. Reducing a tile does not reduce
quadratic pair work. This is a bounded implementation, not a scaling claim.

The CUDA frontend forwards GMRES relative/absolute tolerances, restart, iteration,
reorthogonalization and workspace limits. Native GMRES checks true residuals
itself; nondefault Python stagnation, residual-frequency or breakdown controls
are explicitly unsupported, not silently discarded. Response storage plus
caller device input/output has an independent 2 GiB endpoint cap. Host staging
and immutable publication share the output budget. `cache` remains a CPU/development
compatibility argument and does not activate compilation on CUDA.
These response arenas are not yet integrated into the global `ResourceBudget`
planner; supplying that planner with this CUDA endpoint fails explicitly before
SCF rather than claiming that its SCF-only estimate covers Hessian resources.

Auxiliary g preparation uses a source-only KS DF handle for **values**, not a
generic g orbital/HF/first-derivative capability. The native stationary owner
separately authenticates canonical WB97M-V, the live accepted state, actual DF
provider and complete linked derivative selectors. Neither this role nor a
successful SCF authorizes a missing derivative package.

### Native AOT package selection

`GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON` builds the selected
`generativeqc_hessian_aot` static library and, with native tests enabled, the
standalone `generativeqc_hessian_aot_probe` executable. Python/compiler AD runs
at build time only. The linked CUDA callbacks in `src/dft/hessian_aot.hpp` take
borrowed device pointers and a borrowed stream; they neither compile, allocate,
synchronize, launch an interpreter nor select a CPU numerical fallback.

With `GENERATIVEQC_HESSIAN_AOT_INVENTORY` empty (the default), CMake completes
all ordered orbital-through-f and auxiliary-through-g integral tiles with the
factorized schedule at configure time. The resolved inventory is written to
`generated/hessian_aot/inventory.json` in the build directory. An explicit
inventory overrides this selection, including in an existing CMake cache;
set `-DGENERATIVEQC_HESSIAN_AOT_INVENTORY=` to restore the complete default.

`manifests/hessian_aot_development.json` is a **partial development inventory**,
not a production method/basis capability manifest. It selects the existing
through-f/order-three normalized AO evaluator, RKS/UKS semilocal energy/gradient/
feature-Hessian graphs, coefficient pullbacks/JVPs, VV10 pair gradient/JVP,
mixed Becke/norm/ratio/log primitives, nuclear HVP and ECP radial/harmonic leaves.
Its integral entries explicitly select S/T/V, full/SR/LR ERI and raw DF metric/
three-center component tiles. It does not cover all s/p/d/f shell products.
Missing names fail with `cudaErrorNotSupported`, without JIT or a fallback.

An inventory may explicitly select `"integral_schedule": "factorized"` for
ERI and raw DF leaves. The compiler reuses its canonical one-dimensional
subset/Wick pair coefficients and Cartesian Coulomb states, differentiates
these factors with the existing scalar AD, and emits their product assembly.
One shared CUDA factor library covers orbital-through-f and auxiliary-through-g;
groups of at most 256 host-only metadata entries retain exact component selectors
and the established callback ABI. Scientific program identities do not change
with this schedule; executable/source identities still distinguish builds.
One-electron and scalar/AO leaves keep their existing lowering. The explicitly
selected partial development inventory retains the expanded schedule; missing linked factors
never request runtime compilation or an approximate fallback.

The finite full integral selection can be generated before compilation:

```bash
python tools/complete_hessian_aot_inventory.py \
  --inventory manifests/hessian_aot_development.json \
  --families overlap kinetic nuclear_attraction eri three_center_eri coulomb_metric \
  --max-angular 3 --max-auxiliary-angular 4 --integral-schedule factorized \
  --output build/hessian-full.json
```

Select that file with `-DGENERATIVEQC_HESSIAN_AOT_INVENTORY=<absolute-path>`
to retain an external selection equivalent to the complete default.
Enumeration, source emission and a passing primitive gate do not by themselves
qualify a complete compiled package or a molecular endpoint. The factorized
schedule preserves ordered FP64 radial contraction, but changes factoring
within each primitive and therefore has independent libcint value/gradient/JVP
gates. Its internal DF unit-Gaussian slots are compile-time explicit: every
physical primitive exponent remains strictly positive. The ordinary four-center
geometry binder continues to reject zero exponents. Native scratch bounds do
not imply that compiler-generated CUDA local/stack storage is free; inspect
compiled resource usage and complete endpoint behavior before performance claims.

Scalar panels use `[field,lane]` FP64 layout with input order in the native
descriptor. Integral records use `[exponents,center_xyz,primitive_scale]`; output
is `[component,value/gradient/gradient-JVP]` with all mathematical centers
retained. Attraction uses unit nuclear charge, so the native record scale must
include the actual charge and radial contraction weights. Component normalization
belongs to the enclosing public projection (or the scale for a single-component
consumer), exactly once; raw leaves do not normalize Cartesian components.
The caller must initialize the device error flag, keep buffers/stream alive and
discard all results after any nonzero error. Scientific descriptor identities
authenticate logical programs, not executable bytes or a converged SCF state.

The same opt-in library includes the generic native solver in
`src/response/cuda_gmres.hpp`. Its linked CUDA kernels keep Arnoldi vectors,
scaled norms, modified Gram-Schmidt projections, Hessenberg/Givens coefficients,
triangular solves and candidate/true-residual arithmetic on the device. The
host receives only discrete checkpoint status until the terminal scalar
receipt. Optional signed right preconditioning divides by the supplied diagonal;
entries at or below the requested gap floor fail. Projected convergence and
happy breakdown never bypass the true unpreconditioned residual gate.

The caller supplies the native CUDA operator, a borrowed stream and the exact
admitted workspace. The initial iterate is zero; the public solution is an
output-only panel, not an initial guess. Admission checks options, arithmetic overflow, byte budget,
alignment and nonaliasing before GPU work. There are no solver allocations or
runtime compiler/interpreter dependencies. On numerical failure, exhausted
iterations or a failed operator, the public solution remains unchanged. This
generic solver alone does **not** implement physical CPKS.

The opt-in library also supplies `src/response/cuda_cpks.hpp`: a native RKS/UKS
response controller using this GMRES and AOT TensorIR maps from
`generativeqc_compiler.method.ks_cpks`. It passes both UKS density blocks to a
single native induced-Fock callback. Metric motion, RHS/action and D/W response
are linked CUDA maps; D and W tangents are generated by compiler AD. W uses a
full occupied energy-matrix tangent, retaining off-diagonal occupied response
at same-space degeneracy. Empty/full spin populations are supported. The narrow
dynamic lowering binds full index-space identities, not witness sizes, checks
all retained primal/AD definitions and refuses unqualified primitives. It is a
materializing correctness schedule, not an optimized GEMM performance claim.

The WB97M-V grid generators additionally select checked single-consumer map
and contraction-epilogue fusion. A contraction is never duplicated by fusion.
Single-axis grid reductions use a deterministic 128-lane FP64 tree, including
one cooperative block per output element for AO-potential reductions. This
changes summation order, not precision; independent grid and complete-matrix
acceptance gates still apply. Other affine-lowering consumers retain the
materializing/ordered schedule unless they explicitly select these options.

Full native CUDA Hessians may retain a bounded CPKS operator in occupied–virtual
space. Columns are physical native CUDA actions on unit rotations; subsequent
columns still use GMRES with the requested controls. The final physical Fock
is used to recompute the compiler-generated stationarity residual, which must
pass the original unpreconditioned tolerance before D/W publication. A cached
matvec residual alone cannot qualify a solution. The cache borrows the exact
live frame/provider/device/stream and must be reset before replacing them.
`max_resident_cpks_bytes` defaults to 128 KiB and remains subordinate to the
existing response and aggregate budgets. Zero, insufficient capacity or an
optional allocation failure retains ordinary physical GMRES. Standalone HVPs
do not build this repeated-RHS cache.

The one-electron owner can replay independent precompiled primitive chains on
bounded internal CUDA streams, then fold their results in the original order.
The moving-grid owner can replay its immutable mixed affine schedule. Both
stage caller inputs into owned arena slots and retain the original numerical
and live-source publication gates. Graph resources are prepared in advance;
actions neither compile nor allocate. Explicit disables, insufficient arena
capacity and unavailable optional graph resources retain scalar CUDA launches.
These are bounded CUDA schedules, not CPU or reference-engine fallbacks.

The caller supplies a live-reference authenticator, coupled **native CUDA**
Fock provider, borrowed stream and exact bounded arena. Inputs and induced
Focks must be finite/symmetric; they are checked rather than silently repaired.
D/W outputs remain untouched on failed admission, authentication, numerical
checks, provider calls or exhausted GMRES. Publication follows a second live
authentication and true-residual acceptance. A device failure during final
copies invalidates both outputs. No numerical intermediate reaches the host.

`CudaKsPlan::prepare_resident_final_orbital_frame` separately exposes the complete
resident C/epsilon/D/F/S frame after the existing canonical physical-state proof.
Its column-major C and spin occupations are explicit, and cached leases submit
no new GPU work. Stale tokens and failed proof never publish a binding, and
preparation rejects capture before submission.
Borrowers must keep the owner alive and reacquire the exact token-bound pointers
before execution/publication; a copied binding is not a lifetime proof.
The private prepared-batch port exposes preparation and zero-transfer reborrowing
separately. `src/response/cuda_ks_cpks_reference.hpp` adapts that actual live
batch/item/token to the CPKS callback, checking all bound pointers, dimensions,
producer stream and current CUDA device. It cannot be moved while borrowed.

`CudaKsPlan::resident_final_grid` borrows that same accepted density together
with the actual XC owner's immutable packed AoBasis and quadrature. It requires
canonical preparation and returns no binding for stale tokens or a host-unfused
owner without a resident XC plan. The private prepared-batch grid-reference port
and `src/dft/cuda_ks_grid_reference.hpp` retain this live ownership boundary:
the nonmovable adapter reborrows every pointer, extent, stream and device before
grid execution/publication. Neither detached snapshots nor copying identity
labels authorize reuse. The batch must outlive the adapter and every consumer,
with no concurrent batch calls. This storage lease is intentionally functional-
neutral; evaluating WB97M-V at another model's density does not authorize a
stationary WB97M-V response. The complete molecular provider must also validate
the physical functional, exact/fitted exchange and VV10 model identity.

`src/dft/cuda_wb97mv_grid.hpp` supplies the separate native fixed-grid
WB97M-V semilocal plus bilateral VV10 density-response component. It borrows
validated packed AoBasis, immutable device quadrature and the exact reference
density under a live authenticator. RKS uses total D; UKS retains both physical
spin blocks, including semilocal cross-spin and total-density VV10 coupling.
Scalar coefficient/JVPs reuse compiler AD and the incumbent Libxc work policy;
the AO potential is the TensorIR density adjoint of the existing feature map.
Signed/indefinite directions never require an occupied/PSD factorization.

The caller-owned arena includes the complete dense first-AO-jet panel, retained
fields, staged potentials, map scratch and bounded pair tiles. The reference
fixes VV10's density mask on **both** legs, and its active-count squared must
fit the separate ordered-pair work limit before pair work starts. Tile capacity
does not bound total work. Native work receipts distinguish reference pairs,
response pairs and attempted response actions. Calls reject capture and fence
at finite/symmetry/authentication gates; failed actions do not publish outputs.
The retained native value potential and its response are qualified against
independent Libxc potentials plus a real-space VV10 oracle, including displaced
potential differences, real-spherical d/f records and unequal spin densities.
Exact empty-spin acceptance uses the original pinned upstream Maple C in
libquadmath and an explicit displacement-convergence study; binary64 Libxc
has the previously diagnosed Stoll cancellation at that boundary. The oracle
is test-only, never a runtime fallback.
The live molecular probe additionally starts with actual native WB97M-V
RKS/UKS SCF owners and executes canonical preparation, grid response and stale
publication rejection with Python/compiler paths unavailable. Arbitrary signed
density directions use isolated Libxc reference handles without optional FHC
projection, matching the declared unprojected density-kernel extension; default
handles remain untouched. Separate PSD displaced-density paths keep unmodified
default Libxc and independently qualify physical indefinite tangents. Both use
the same value/response gates as the standalone component. These tests cover
the live fixed-grid component only, not a complete molecular Hessian.
This is a correctness-first dense component, not a scaling/performance claim.
The grid adapter alone authenticates basis/grid/density storage, not a complete
physical model. `src/dft/cuda_ks_wb97mv_response.hpp` now supplies the enclosing
live-owner physical induced-Fock component. It verifies the entire native
WB97M-V identity, spin populations, primary approximation and exact LR
correction, then reconstructs **the actual accepted F** from H, full-range J/K,
exact LR exchange and the retained semilocal/VV10 value potential before exposing
an action. Copied labels do not replace reborrowing the actual source pointers.
The host-only `CudaKsResidentFockBinding` is available only after scalar canonical
validation, and is revoked by a new solve just like the orbital/grid leases.

Exact J/K and LR actions use an independent, unscreened canonical Direct source
on the borrowed producer stream, with caller-owned compensation and census
planes. Geometry-only screening makes the operator linear for arbitrary signed
directions. The primary SCF source's scratch/caches are not replayed or mutated.
For admitted single-system, full-rank DF sources, a read-only source seam borrows
the immutable raw-integral source and actual whitening operator. Raw values are
generated directly into separate response storage without device switching or
revoking occupied-projection leases. The compiler forms W-transpose-W in the
source's declared layout, then reuses the **density VJP of the existing fitted
energy**. It does not fit LR exchange, substitute full exchange or apply an
occupied-factor RI-K formula to an indefinite density direction. Missing raw
sources or device whitening, rank-truncated metrics, and mixed primary
approximations fail closed.

This correctness schedule materializes raw three-centers, inverse and affine
intermediates. Device bytes, conservative Direct ERI work and compiler-enumerated
IR scalar-term visits have separate checked bounds; the latter include retained
AD primal validation dependencies. Optional canonical Direct storage is an
explicit admission requirement, not permission for a density-screened fallback.
No endpoint performance claim follows from this schedule. Preparation allocates;
actions do not allocate, compile, run Python/CPU contractions or switch devices.
They support the coupled `solve_cuda_cpks` callback and reauthenticate before
publication. Real molecular RKS/UKS Direct/DF tests exercise both the physical
action and CPKS D/W responses with Python/compiler paths unavailable. Independent
PySCF raw/full-range/LR integrals, Libxc/VV10 potential differences and displaced
generalized-eigenprojector dense solves provide numerical acceptance; the
synthetic displaced-SCF tests remain separate regression coverage.

`src/dft/cuda_grid_geometry.hpp` supplies the separate native moving-grid
**geometry source**. It caches ordinary AO spatial jets through order three
from the existing AOT Gaussian AD and returns first/mixed-second geometry
responses for the value/gradient prefix. Its build-time Taylor AD uses
point-owner motion minus basis-center motion, including the correct mixed
spatial multiplicities for Cartesian and real-spherical records.

Moving Becke weights borrow the actual immutable CUDA pre-partition atomic
measure, owner ordering and partition policy through `CudaKsGridReference`.
Host-only quadratures retain fixed-grid capability but do not acquire moving
geometry by dividing partitioned weights or rebuilding a CPU quadrature.
Preparation verifies the supplied primal partitioned weights in dimensionless
partition units, using the actual atomic measure to account for far-radial
roundoff amplification. A zero atomic measure requires a zero reference weight.
Scalar norm,
ratio, smooth switching and mixed product derivatives are compiler-generated;
native traversal retains common binary scaling for the entire four-component
product jet. Exact-zero factors need no division by a factor. Pair-center jets
are prepared once per action and reused across bounded point tiles, without
atom-count-specific compilation. Point-pair visits and cached/output AO values
have checked semantic-work bounds independent of the workspace bound. A separate
conservative Gaussian primitive/polynomial-term bound admits preparation work;
small AO outputs alone do not bound contraction cost. Packed basis indices,
primitive ranges/exponents and supported polynomial records are device-validated
before the existing trusted AO leaf or nuclear-direction indexing executes.

The caller supplies one arena and disjoint AO/weight outputs. Both are staged,
finite-checked and reauthenticated before publication on the original ordinary
stream. Capture, coincident centers, multiatom point-center norm singularities,
invalid primal measures/weights and revoked owners reject. One-atom quadratures
need no norm differentiation and allow a point at the center. A CUDA error
during final publication invalidates both outputs. Independent polynomial
Gaussian and displaced extended-precision Becke-value tests cover mixed
derivatives, translation/swap symmetry and actual native RKS/UKS Direct/DF
grid ownership with Python/compiler paths unavailable.

The last successful geometry action also lends `[left/right,point,xyz]` motion
gathered from the actual owner-atom directions on CUDA. Any prepare/mixed
attempt, including a rejected attempt, expires that motion lease.

`CudaWb97mvGridResponse` now has an opt-in `nuclear_response` arena layout.
`prepare_nuclear_direction` consumes AO, measure, point and signed RKS/UKS
density directions and publishes the moving grid Fock derivative. It copies
right-direction inputs and caches the complete semilocal/VV10 field-partial
JVP once. `nuclear_mixed` combines those retained partials with compiler-AD
left/mixed AO features and moving weights into a device scalar. Both moving
VV10 legs, coordinate and weight partials, the local beta term and physical
tau cotangents participate. No VV10 pair traversal repeats per left coordinate.
Zero right density gives the frozen nuclear grid Fock; nonzero right density
includes its total directional response. This is not the integral Fock or a
complete stationary Hessian.

The ordinary fixed-grid layout remains the default and incurs no nuclear
storage or traversal. Nuclear admission separately bounds caller arena bytes,
ordered pair evaluations and compiler-counted materialized affine term visits,
including AD primal dependencies. All execution stays on the authenticated
original device/ordinary stream, without allocations, runtime compilation or
host numerical contraction. Any prepare/apply/right-direction attempt expires
the armed right cache. Validation, alias, capture, finite-value and live-owner
gates precede transactional publication. Independent Libxc/real-space VV10
value/potential differences qualify synthetic s/p/spherical d/f data and actual
native RKS/UKS Direct/DF geometry. Arbitrary signed molecular directions use
separate upstream Libxc handles without its optional FHC PSD projection;
interior positive-definite cases retain the default unmodified oracle.

`src/integrals/cuda_one_electron_response.hpp` supplies a separate native
compiler-AD overlap and `T + effective-point-charge attraction` nuclear source.
It consumes the actual normalized AoBasis packing, including Cartesian
component normalization and real-spherical expansions through f. Primitive
coefficients contain radial normalization only; public projection coefficients
are applied exactly once on CUDA. The method supplies finite symmetric core/
overlap weights and their right-direction responses, including signed weights.
The source publishes core/overlap values and directional matrices together with
weighted nuclear gradient/HVP vectors. It retains every mathematical center
until deterministic physical-atom scatter, including coincident centers.

The caller provides a bounded arena and a live basis/effective-charge
authenticator. Host preparation selects integer shell topology and linked
program descriptors only. Actions build radial records, evaluate AD leaves,
project public matrices and contract derivatives on CUDA, without allocation,
runtime compilation, Python or CPU numerical contraction. The linked inventory
must cover every required ordered Cartesian component exactly once; missing,
overlapping or malformed closure fails before CUDA work. Independent limits
bound jobs, primitive-component evaluations, public projection terms and arena
bytes. `projection_terms` is the conservative nine polynomial-term products per
public AO pair/component/radial tile, not an endpoint timing. A radial tile in
`[1,64]` bounds storage but does not reduce primitive work. All six outputs are
private until finite/symmetry and live-owner reauthentication gates pass. Failures
before publication leave them unchanged; a CUDA publication failure invalidates
all six. Actions reject aliases, foreign streams/devices and stream capture.

`src/dft/cuda_ks_one_electron_response.hpp` connects this source to an actual
canonical native KS batch. Integer topology/effective charges come from the
prepared Fock owner's normalized System; basis storage comes from the live grid
binding. Independent arena/charge storage never overwrites provider scratch.
Both bindings are reborrowed on every action and before publication. ECP core
removal is retained, but **residual local/nonlocal ECP projectors are not included**.
The adapter does not invent spin/HF factors or supply stationary D/W policy.

The partial development inventory is intentionally unchanged. A complete
one-electron closure can be generated **at build time** with
`tools/complete_hessian_aot_inventory.py --families overlap kinetic nuclear_attraction
--max-angular 3 --inventory manifests/hessian_aot_development.json --output <inventory.json>`,
then selected with `-DGENERATIVEQC_HESSIAN_AOT_INVENTORY=<absolute-inventory-path>`.
`tests/python/test_native_one_electron.py` compares all ordered s/p/d/f public
matrices, first directions and weighted mixed response with independent displaced
libcint/PySCF values. The standalone probe starts from the native System
normalizer/AoBasis; actual RKS/UKS Direct/DF owner tests also compare primal H/S
with the accepted native frame and verify revocation without publication.

`src/dft/cuda_wb97mv_nuclear_jk.hpp` supplies the separate native ordered-quartet
J/K nuclear source. Compiler TensorIR VJP/JVP generates the RKS/UKS energy,
integral weight and its right-density response, four density-incidence Fock
contributions, and frozen-density nuclear Fock motion. The exact operator uses
full-range primary plus exact LR correction; individual Coulomb/SR/LR parts are
available for independent qualification. Signed symmetric D and dD are allowed.
Fock nuclear motion holds D fixed, while the stationary mixed nuclear vector
retains right dD; consumers must not double count CPKS relaxation.

The mandatory bounded shell-quartet cache stores raw value/twelve gradients/
twelve gradient JVPs once per primitive component. Through f its maximum is
25*10,000 FP64 values, not a molecular ERI tensor. Public Cartesian/spherical
polynomial terms gather this cache instead of repeating integral work for every
public quartet or left coordinate. Native deterministic incidence/physical-atom
reductions publish energy, Fock, frozen-density Fock direction, weighted gradient
and HVP jointly after finite/symmetry/live-owner gates. Separate primitive,
projection-term, reduction-term and arena budgets fail before execution; actions
allocate nothing, execute no Python and perform no CPU numerical contractions.

The native source additionally admits an optional last-direction resident jet
pool, bounded by `CudaWb97mvJkOptions.max_resident_integral_bytes` (64 MiB by
default) and the existing arena/aggregate device budgets. Zero disables it;
insufficient capacity retains the original bounded shell-quartet streaming path.
All mandatory DF/ECP storage is reserved before this optional pool. Reuse requires
an authenticated immutable basis lease and an exact device comparison of every
right-direction coordinate. Pointer equality is insufficient: jets include the
right-direction HVP. Current D/dD are always reprojected, so frozen-density RHS
and relaxed contractions can share jets without sharing density-dependent
results. The pool stores one source per ERI eightfold shell-symmetry class;
projection permutes both Cartesian component strides and gradient/HVP center
slots back to each ordered group. Streaming retains the fully ordered source.
Work admission remains conservative against the ordered upper bound, while
receipts count the actually selected builds. Reprepare or a failed cache rebuild
invalidates reuse. One CTA performs
the ordered incidence reductions with barriers; physical-atom and energy outputs
are disjoint, with no floating-point atomics or changed public reduction order.
Symmetry-related primitive accumulation can differ at FP64 roundoff and remains
subject to the same independent scientific acceptance gates.
CLI work receipts expose `exact_integral_source_builds`,
`exact_resident_integral_hits` and `exact_resident_integral_bytes` alongside actual
primitive-component evaluations. Residency is not a multi-RHS solver or a
guarantee of endpoint speedup.

`src/dft/cuda_ks_wb97mv_nuclear_jk.hpp` binds actual accepted D and normalized
basis/topology from live KS grid/Fock owners. It shares complete WB97M-V model
admission with the induced-Fock provider. **DF owners admit only the exact LR
correction here**; fitted primary J/K nuclear response remains separate and is
never silently replaced by exact integrals. All ordered requested integral
components must already be linked. Build-time closure can be selected with
`tools/complete_hessian_aot_inventory.py --families eri --max-angular <0..3>
--radials full_range long_range short_range`; generation/admission alone is not
numerical angular qualification. `tests/python/test_native_wb97mv_nuclear_jk.py`
qualifies s/p/d numerical components, resident/streaming/budget fallback parity
and actual RKS/UKS Direct/DF owner lifetimes;
injected CUDA-free closure tests additionally check largest through-f storage
and work admission, not through-f integral numerics.

`src/dft/cuda_wb97mv_df_response.hpp` supplies native full-rank fitted-primary
algebra from the incumbent stationary DF TensorIR AD. Reference maps return
energy, spin Fock and raw three-center/metric cotangents. Response maps return
frozen-D Fock motion and source-weight JVPs retaining dD, dA and the implicit
inverse motion `-V^-1 dV V^-1`. The physical inverse respects the actual
whitening orientation `W^T W`; no occupied-density, frozen-metric or symmetric-W
assumption is made. Generic compiler contraction planning runs before both
emission and work counting: all scheduled nodes, including AD validation
dependencies, avoid quartic-orbital or quartic-auxiliary contraction work and
quartic intermediates. Explicit arena and scalar-term bounds fail before work.

`src/dft/cuda_ks_wb97mv_df_response.hpp` borrows actual accepted D, raw integral
source, whitening and ascending metric eigenvalues, and binds them to the
complete native WB97M-V DF model. It prepares retained raw public A on the
original stream without borrowing or revoking occupied-projection/J/K scratch.
The single device-byte bound includes retained A and all algebra storage.
Device-side numerical admission requires the smallest metric eigenvalue to
exceed `max(1.1 * relative_cutoff, 64 * eps) * largest_eigenvalue`; truncated or
cutoff-unstable metrics fail without Direct substitution. Signed symmetric dD
is accepted. Immutable reference weights are retained; actions allocate nothing,
use no Python/CPU numerical contraction and reauthenticate before all seven
outputs are published. Raw dA/dV are supplied by a separate integral consumer;
this algebra adapter does **not** supply nuclear derivatives itself.

`tests/python/test_wb97mv_df_response_compiler.py` checks the compiler maps
against independent expanded fitted ERIs and source derivatives, including a
nonsymmetric whitening gauge. `tests/python/test_native_wb97mv_df_response.py`
compares the compiled maps at runtime-sized signed RKS/UKS shapes and actual KS
owners against independent NumPy/libcint values, and exercises finite,
symmetry, alias, capture, stream, cutoff, resource and revocation gates.

`src/integrals/cuda_df_nuclear_response.hpp` supplies a separate bounded two-pass
raw DF nuclear source. The first pass publishes full ordered public A, V and
their geometry directions. After inverse/Fock/weight AD and right density
response, the second pass contracts fixed/moving raw cotangents into a nuclear
gradient and its HVP. Compiler AD remains the derivative authority; the native
owner gathers actual normalized primitives, caches one shell's mathematical
center derivatives and applies fixed public expansions exactly once. No global
coordinate-major derivative tensor or fitted four-center tensor is retained.
All coincident mathematical-center incidences survive deterministic scatter.

`src/scf/cuda_density_fitting_basis.hpp` exposes only the actual immutable DF
basis arrays and six-term-capable auxiliary-g projection ABI, not mutable
integral/J/K scratch. `src/dft/cuda_ks_wb97mv_df_nuclear.hpp` binds this source to
the actual accepted RKS/UKS WB97M-V DF owner, integer normalized orbital and
auxiliary topology, original stream and full live-state identity. Actions stage
all outputs and reauthenticate before publication. Both passes explicitly
charge primitive-component evaluations and repeated component-tile radial
records; conservative reduction bounds include primitive-field sums, cache
accumulation, projection and physical-center scatter. These receipts are not
FLOPs or complete-endpoint timings.

The AOT inventory builder accepts `--max-auxiliary-angular 4` independently of
the orbital-through-f bound. A native consumer requires complete nonoverlapping
linked tiles for every actual ordered shell class and fails closed otherwise.
Admission/generation support does not imply numerical qualification of the full
f/f/g closure. `tests/python/test_native_df_nuclear_response.py` compares raw
values, first derivatives and weighted HVPs with independently normalized
Cartesian libcint sources and independently applied Cartesian/spherical
projections. Its actual RKS/UKS source/inverse/moving-weight chain remains a
component test, not a stationary Hessian.

`src/integrals/cuda_ecp_nuclear_response.hpp` supplies a separately bounded
residual-ECP nuclear source. Build-only `integral.ecp_fixed_grid` freezes the
incumbent coarse/refined quadratures; radial potentials and the sixteen real
s/p/d/f projectors retain the existing compiler graphs. Each ECP-center radial
layer collocates ten spatial AO jets once and retains their fixed harmonic
projections. Compiler AO Taylor AD and the stationary AO bilinear VJP/JVP supply
local/nonlocal matrices, frozen-density geometry directions, total-density
gradients and joint geometry/density HVPs. Left nuclear coordinates reuse these
jets and cotangents, without replaying collocation or retaining a global
coordinate-major derivative tensor. The Coulomb tail remains in the separate
effective-charge attraction and nuclear-repulsion sources.

`src/dft/cuda_ks_wb97mv_ecp_nuclear.hpp` binds this residual source to actual
accepted WB97M-V RKS/UKS Direct/DF owners. It borrows normalized native AO basis
and physical spin densities, stages immutable normalized residual terms once,
and uses generated CUDA spin sums for D/dD without occupancy/half factors.
All staged terms, sums and scratch count against the explicit device budget;
source work receipts distinguish collocation, harmonic projection, affine work
and conservative reduction bounds. Actions reauthenticate the complete model,
canonical frame, source identity and original stream. Both quadratures must
pass the incumbent matrix/derivative convergence policy before any of the four
outputs is published. Finite, symmetry, resource, alias, stream, capture or
revocation rejection preserves all outputs; a CUDA publication fault may leave
partial copies. The source does not mutate ordinary J/K or occupied projections.

`tests/python/test_native_ecp_nuclear_response.py` independently qualifies
local/projector matrices and physical first derivatives with libcint, validates
the analytic oracle against displaced energies, and checks moving-density HVPs
against displaced analytic gradients at two steps. Synthetic Cartesian and
real-spherical through-f and multicenter sources complement accepted RKS/UKS
Direct/DF s/p owner fixtures. Nonconvergent quadrature must fail without
publication. These are source/component gates, not complete stationary molecular
Hessians or complete-endpoint speed measurements.

### Native stationary composition and CLI

`CudaKsWb97mvHessian` binds the actual accepted native state and composes H/S,
exact or fitted-primary plus exact LR J/K, residual ECP, moving XC/VV10/Becke
and nuclear-repulsion sources around one native coupled-spin CPKS solve per
right direction. Reference W reuses the primal of `method.ks_cpks`'s existing W
JVP. Compiler TensorIR owns physical D/W spin totals, negative overlap weights,
frozen nuclear Fock assembly, pair-incidence signs/reductions and final sums.
Native code owns only resource/lifetime/traversal and finite/publication gates.

The linked integral emitter assigns one CUDA block to each raw component and
evaluates up to 64 independent primitive records concurrently. Shared storage
holds scaled fields until one lane contracts them in the original radial order;
this is not a reassociated tree sum or floating-point atomic reduction. Longer
record lists stream successive bounded batches, and every lane participates in
the failure barrier before a component can publish. The derivative graphs,
normalization, shell/AO projections and global workspace contract are unchanged.

The frozen pass uses zero density/weight motion. After converged CPKS, the
second source pass includes solved dD/dW. DF retains dA/dV across that boundary,
reruns the full inverse/source-weight AD with dD, then contracts the raw source.
It never substitutes Direct J/K for a fitted primary. Nuclear grid preparation
traverses VV10 pairs once per right density direction; left-coordinate mixed
contractions reuse its retained JVP and cached spatial AO jets. Pair incidence
gathering uses O(atoms squared) work rather than scanning all pairs per atom.

Full Hessians may additionally retain frozen exact J/K Fock derivatives and
frozen HVP columns under `max_frozen_exact_bytes` and the aggregate byte budget.
The compiler-AD integral source is traversed once per nuclear coordinate.
Because exact J/K energy is quadratic in the signed spin densities, its solved
density contribution is the contraction `F_R(D) : dD`. This retains only
`3N * spin * AO^2 + (3N)^2` FP64 values, not four-index derivative jets. Standalone
HVPs do not construct this cache. Zero capacity, insufficient aggregate storage
or optional allocation failure preserves the original directional source passes.
DF uses this cache only for its exact LR correction; fitted-primary AD is unchanged.

The exact nuclear source admits a bounded canonical-source tiled schedule when
the complete ordered schedule exceeds its group, work or arena bound. Each tile
expands the distinct eightfold shell permutations with explicit Cartesian and
derivative-slot mappings, using the same linked compiler-AD leaves. Only one
tile resides on device. Adaptive subdivision preserves source reuse rather than
silently falling back to eightfold primitive replay. `max_source_groups` and all
primitive/projection/reduction bounds apply to the complete action, not per tile;
`max_groups` continues to bound a resident ordered tile. Shared publication
storage is reserved before admitting tile scratch. Disabling `source_tile` keeps
the original fail-closed ordered admission. Diagnostics expose source groups,
streaming tiles, source builds and frozen derivative contractions.

Moving-grid preparation can admit the actual shell-derived AO primitive and
polynomial incidences instead of multiplying every AO by every primitive in the
molecule. The tighter bound is checked against device AoBasis records before
the first AO jet evaluation. Callers without this topology retain the generic
conservative bound, and understated metadata fails without publication.

All independent owners, generated scratch, staged values and complete Hessian
storage share an aggregate response-byte bound. Actions allocate nothing. CUDA
success and CPKS convergence are checked separately; one failed column prevents
whole-matrix publication. Raw Hessian symmetry is checked, not repaired.
CLI commands require CUDA FP64 and linked closure, and never start Python/JIT.
Their complete model is serialized from MethodIR at build time. Obsolete worker
options are rejected, as is CPU CLI execution rather than silently retaining a
Python fallback. The public Python CUDA API marshals inputs to this same native
owner; CPU and low-level development APIs remain separate consumers.

`test_native_wb97mv_stationary.py` uses independently reconverged libcint/Libxc
and direct-space VV10 values at two displacement steps. The native child has
interpreter/compiler discovery paths unavailable; other tests exercise actual
compiled CLI input/publication. These small-grid gates are not grid convergence,
complete high-angular shell coverage or GPU4PySCF endpoint-speed evidence.
Practical complete high-angular ERI/DF AOT closure and broad molecular/benchmark
qualification remain required before claiming the full migration complete.

For a source build with a verified compiler-cache launcher configured:

```bash
cmake -S . -B build/cuda-release-sm120 \
  -DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON
cmake --build build/cuda-release-sm120 --target \
  generativeqc_hessian_aot_probe generativeqc_cuda_gmres_probe \
  generativeqc_cuda_cpks_probe generativeqc_cuda_wb97mv_grid_probe \
  generativeqc_ks_wb97mv_grid_probe \
  generativeqc_cuda_grid_geometry_probe \
  generativeqc_cuda_ecp_nuclear_probe \
  generativeqc_ks_resident_frame_tests
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_HESSIAN_AOT_PROBE="$PWD/build/cuda-release-sm120/generativeqc_hessian_aot_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_hessian_aot_cuda.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_GMRES_PROBE="$PWD/build/cuda-release-sm120/generativeqc_cuda_gmres_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_cuda_gmres.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_CPKS_PROBE="$PWD/build/cuda-release-sm120/generativeqc_cuda_cpks_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_ks_cpks.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_WB97MV_GRID_PROBE="$PWD/build/cuda-release-sm120/generativeqc_cuda_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_native_grid.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release-sm120/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_native_grid.py \
  -k live_molecular
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release-sm120/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_native_fock.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_GRID_GEOMETRY_PROBE="$PWD/build/cuda-release-sm120/generativeqc_cuda_grid_geometry_probe" \
  GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release-sm120/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_native_grid_geometry.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_WB97MV_GRID_PROBE="$PWD/build/cuda-release-sm120/generativeqc_cuda_wb97mv_grid_probe" \
  GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release-sm120/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_nuclear_grid.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:05:00 \
  build/cuda-release-sm120/generativeqc_ks_resident_frame_tests
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:30:00 \
  env GENERATIVEQC_CUDA_ECP_NUCLEAR_PROBE="$PWD/build/cuda-release-sm120/generativeqc_cuda_ecp_nuclear_probe" \
  GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release-sm120/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_native_ecp_nuclear_response.py
```

For the exact empty-spin grid tests, also set
`GENERATIVEQC_WB97MV_WIDE_ORACLE` to the independent upstream quad executable
emitted by `tools/qualify_wb97mv_tail.py::oracle_source` from the pinned archive
and built with a verified compiler cache. Without it those boundary cases are
explicitly skipped, not qualified. No quad/Python oracle runs in production.

The Python test driver supplies independent quadrature/Libxc/real-space oracles;
numerical leaves and response components execute in standalone native processes,
including tests with unavailable Python paths and an empty executable search
path. These validate components, not a complete stationary molecular endpoint.
These component receipts do not themselves qualify the native stationary
composition, complete integral closure or production CLI. Run the independent
molecular and actual CLI/public-frontend gates for the selected linked inventory;
component tests alone cannot establish complete endpoint speed.

### CPU/development representation, spin and fitting

The following Python owners serve CPU and explicit low-level development calls.
Public CUDA calls instead use the native stationary owner described above,
including its independent orbital/auxiliary layouts and linked through-g
auxiliary admission.

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

The C++ executable provides native `hessian`/`hvp` commands. Their Python-free
runtime, linked AOT build selection, controls and examples are
owned by [the native CLI guide](../user/native_cli.md#compiler-ad-wb97m-v-hessian-and-hvp).

The linked native owner amortizes fixed-geometry induced Fock actions with
optional, independently owned strict FP64 full-range/LR integral sources. Radial
operators have distinct immutable leases; DF retains its fitted primary source
and caches only the exact LR correction. Composite preparation reserves every
mandatory consumer before admitting these optional sources. Their values and
provider metadata are charged to the aggregate response budget; disabling
residency or insufficient capacity retains the compensated exact replay.

For small public shell products, the optional nuclear-integral schedule batches
independent primitive chunks sharing a linked compiler selector. Compact records
retain exponents and radial scale while borrowing immutable molecular positions.
The compiler-emitted adapter reconstructs the ordinary primitive record, without
changing its AD expression. Projection runs across independent shell groups;
sparse single-owner reductions retain the original group/incidence summation
order, including coincident centers. Projected public quartets reuse an ordered
shell-group tile, capped by `max_batch_projection_bytes` (16 MiB by default) and
the remaining arena. They do not require a full public-quartet tensor to fit.
CSR lists are group ordered, permitting binary searches for each tile's interval;
each output carries its existing FP64 sum across tiles without reassociation.
Large public shell products, missing batch
bindings and insufficient workspace retain the ordered streaming schedule.
All optional storage remains inside the existing integral and aggregate budgets;
batching does not relax component, group, projection or reduction work limits.

Moving Becke geometry uses a 1024-point tile cap to expose independent points
across several CTAs rather than repeatedly running a partially occupied CTA.
The complete per-tile distance/product/exponent scratch is included in checked
workspace admission. Explicit smaller `CudaGridGeometryOptions::point_tile`
values retain the same pointwise AD and summation order.

Native VV10 response uses a 256-point pair-tile cap (clamped to the grid extent
for storage). Smaller explicit `CudaWb97mvGridOptions::pair_tile_points` values
retain the low-memory schedule. Both schedules evaluate the same complete
reference-screened ordered-pair domain. CLI diagnostics distinguish native SCF,
response preparation and Hessian execution through the final host matrix copy;
owner-accounted bytes are not total device-memory usage.

`generativeqc.ecp_hessian` differentiates the actual residual local potential
and nonlocal harmonic projectors with TensorIR mixed AO/density JVPs. Both
AO-center and ECP-center motion are included. One radial layer is streamed,
using the native coarse/fine ECP quadrature policy; every HVP must pass the
retained absolute coarse/fine gate. Effective nuclear charges are used in
attraction and nuclear repulsion, without subtracting the ECP tail twice.

The explicit low-level `cuda_composite_hessian` development owner uses a
**device-resident numerical endpoint**, requiring the optional
`generativeqc[cuda-hessian]` dependency. This is not the public CUDA frontend or
CLI: those use the linked native owner and need neither CuPy nor snapshot export.
In this development owner, the converged native snapshot is uploaded once.
AO jets, moving Becke partitions, scalar functional/VV10 AD, first and
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

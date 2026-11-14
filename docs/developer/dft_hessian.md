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

The planner binds the shared CPKS contract, the XC feature-Hessian action and
the weighted second-integral HVP contract. Its integral blocks execute only the
bounded source algebra; native CPKS and XC/grid consumers remain separate
owners. Second-order support is admitted through an exact primitive-type rule
registry rather than a fixed "one semilocal primitive" condition. Each rule
declares its required derivative capabilities, supported ingredients,
directional sources and any response inputs needed by bounded integral HVP
lowering. The plan derives its source inventory by composing those rules with
the stationary mean-field envelope. It fails closed when an active primitive has
no registered second-order rule. The full-range exchange rule reuses the
existing ordered-quartet stationary-gradient weight and its TensorIR JVP with
both same-spin density response inputs. It requires first-integral directions
and generated weighted second-integral HVPs. Range-separated rules preserve each
operator's immutable omega and generate separate SR/LR density-weight JVPs.
The meta-GGA rule includes `tau`; the VV10 rule declares both pair legs' feature,
point and quadrature-weight response. The ECP envelope adds independent local
and projector sources. Full-rank DF uses compiler-generated three-center/metric
cotangents and their response JVPs. rVV10 remains unsupported at this planning
boundary. A compiled source rule does not qualify a molecular endpoint.

Molecular execution is a separate method-neutral layer. Its canonical installed
owner is `generativeqc.second_order`; repository Hessian tools import this owner
directly rather than retaining a second implementation.
`StationarySecondOrderExecutor` accepts only a plan identity and complete
ordered source inventory, one perturbation provider, one opaque stationary
response driver and exactly one contributor per declared source. It performs one
response solve, passes the same response object to every contribution and
publishes a result only after every source returns a finite Cartesian HVP. There
is no HF/RKS/UKS or functional-name dispatch in this executor. A
MethodIR-derived DFT plan and an HF second-order plan can therefore share the
same orchestration while retaining their own perturbation, response and
primitive providers. Missing or extra contributors fail before execution.

The plan exposes bounded `integral_block` programs for the one-electron,
Coulomb, overlap/Pulay and optional full-range exchange sources. Each block
reuses the stationary-gradient source energy and derives its fixed integral
weights, then generates an exact TensorIR JVP for a supplied density or
weighted-density response, and contracts that response with the first-integral
directional tile plus the fixed-weight second-integral HVP vector. This is
executable source-level algebra for both RKS and UKS; native shell/center
recovery, shared CPKS execution, XC/grid/partition motion and molecular assembly
remain owned by their qualified consumers. In particular, compiled PBE0/B3LYP
exchange HVP weights do **not** qualify hybrid molecular Hessians: the native
mixed K/XC CPKS provider, metric/Pulay relaxation and complete molecular source
inventory still need independent numerical acceptance.

The shared closed-shell CPKS operator now accepts an explicit physical
full-range exact-exchange fraction for CPU-side algebraic composition. It binds
the resulting restricted spin-summed -a/2 raw K action into the response
identity and **refuses nonzero exchange unless the backend explicitly declares
raw full-range K support**. Semilocal a=0 keeps the same identity and behavior.
This response contract alone does not grant a molecular capability.
`NativeRKSResponse` remains restricted to native LDA/PBE; full-range hybrid
molecular execution uses the shared composite owner described next.

The shared `CompositeRKSResponse` owner also accepts
CPU direct all-electron RKS full-range-exchange graphs with the native
LDA/PBE or B88/VWN-RPA/LYP semilocal families. It binds the physical token-verified MethodIR,
not the native carrier selector, to the mixed J/K/XC response, frozen K nuclear
RHS and compiler-owned `exact_exchange` second-order source. Cartesian and
real-spherical AOs reuse `HessianBasis`; the SCF, density and response remain in
the original public variational space. The public wrapper separately admits
strict-FP64 CPU direct all-electron PBE0 RKS and B3LYP RKS/UKS semilocal compositions
with one full-range coefficient in `(0, 1]`, Cartesian/spherical AOs through f,
and no additional electronic primitive. Admission compares resolved mathematics
with the same physical spin rather than a user-controlled identifier. An uncut
two-body D3(BJ), ATM-off correction is an independently qualified additive
source; it does not repeat electronic response. Only the exact canonical B3LYP
RKS/UKS graph additionally admits scalar ECPs through the linked native CPU
owner, with the same direct FP64, AO and resource bounds. PBE0 UKS, ECPs on
PBE0 or coefficient-variable graphs, DF and CUDA full-range variants remain
excluded. A spherical Ni/def2-SVP fixture does not
establish production-scale support for 25–40-atom Ni systems.
The low-level receipt always reports `public_calculator_endpoint=False`;
only the actual Calculator wrapper marks the endpoint public and adds native
SCF counts, energy, backend and complete endpoint time.
Canonical B3LYP now dispatches to the linked native CPU owner described below;
the shared Python composite implementation remains a development/qualification
consumer and the implementation for other currently admitted compositions.
The CPU restricted composite owner conditions its existing shared response
solve with signed occupied/virtual orbital gaps. Zero/near-zero gaps decline
this scaling without modifying the physical operator or residual gates; see
[the response contract](response.md). This is method-neutral conditioning,
not a new hybrid response solver or additional endpoint qualification.
CPU full-range integral sources reuse the compiler's factorized Hermite/Boys
jets through a bounded host consumer, rather than compiling each ERI component
tile separately. This consumer admits at most 128 primitive records and 64
components per page; Cartesian normalization and physical atom scattering stay
with their shared owners. Second-order execution reserves shell/publication
buffers before lending the remaining numeric budget, and retains the original
bounded component lowering when the factorized lease cannot fit. Generated
scalar stack temporaries, compiler processes and library mappings are excluded
from this numeric lease; it is not a complete endpoint memory/time bound.
The compiler owns disjoint AO-component partitions for expanded second-source
lowering. CPU scalar lowering uses at most eight components per artifact,
independently of coordinate-output tiles, to bound the optimizer's CSE dimension
for high-angular S/T/V and resource-fallback ERIs. The shared factorized host
program retains 64-component pages; CUDA's existing 64-component schedule is
unchanged. Every component is accumulated exactly once for each output tile.
Artifact identities already bind both subsets; smaller tiles do not relax
finite compiler deadlines, arithmetic or numeric-budget checks. Diagnostics
report the scalar-lowering component limit separately from semantic work counts.
The compiler owns the shared exact eightfold ERI permutation schedule. CPU
directional sources reuse one canonical raw jet for full-range J and K, keeping
their distinct density factors with the runtime consumer. Full-range weighted
first/second sources sum each distinct ordered shell cotangent after transposing
it into canonical component order; repeated shells do not imply a factor of
eight. Second-order folding reserves the output and two weight blocks, or keeps
the ordered bounded schedule if that additional storage does not fit. Every
ordered weight is validated, including exact translation-zero quartets.

The native CPU exact full-range final-state **gradient** consumer also reuses
the shared native `eri_symmetry.hpp` canonical index schedule. It sums the
compiler-defined Coulomb and prepared-exchange cotangents of each distinct
ordered shell orientation before one packaged derivative evaluation. Repeated
shells retain their full AO-component blocks, not a fixed multiplicity; arbitrary
finite asymmetric D/W inputs retain their ordered contraction semantics.
Independent-slot derivative providers are unchanged: only the physical nuclear
consumer folds orientations and retains its exact one-center translation
elision. Weight buffers and conservative resource allowances are unchanged.
The optional CPU journal separates hcore, Pulay and two-electron source phases;
shell/primitive counters report actual submitted work, not unfurled AO powers.
Its packaged CPU s/p/d ERI first-derivative shards explicitly select the shared
positive Full-range moment evaluator also used by the factorized CPU sources.
Only the geometry moment policy changes: the derivative graph, record weights,
Cartesian slots, finite batch bound and transactional output remain unchanged.
The generic CPU emitter defaults and CUDA generation keep their original
evaluation. This does not expand range-separated or public ECP saddle admission.

Packaged shell derivatives keep primitive values in fixed twelve-slot storage
and update already-sized channel accumulators in place. The compiler-generated
scalar-AD multiply/add retains its original zero primal, zero tangent term and
ordered FP64 accumulation; external-coefficient overflow still rejects the
whole contraction, even when the physical derivative is zero. Result and work
counts publish only on success. The independent reference recurrence and
resource allowances remain unchanged; no per-primitive numeric cache is added.

Each stationary source prepares its density/energy-weighted-density pullbacks
once per response and reuses the compiler's weight/JVP programs by term count.
Shell callbacks evaluate only the requested fixed or moving cotangent. They do
not retain rank-four weights, integrals or a geometry-replay numeric cache;
MethodIR/TensorIR still owns all spin factors and exchange coefficients.
The compiler partitions ordered terms under the same per-block symbolic element
budget for both pair and quartet weights. Runtime gathers only each tile's
indices and density feeds, preserving flattened component order and the complete
shell-local output; a large shell does not raise the compiler admission cap.
Its admitted AO-jet reserve includes native basis storage and tile/publication
buffers and is passed explicitly to the native evaluator. The remaining grid
budget excludes that reserve; raising the grid allowance does not remove the
independent point/integral work or capacity checks.

The native CPU preparation freezes the actual full-range coefficient from the
physical execution plan and passes it to the shared RKS/UKS SCF wrappers. The
wrapper still authenticates the complete prepared Fock strategy; it does not
substitute a named functional's canonical fraction. Final-state validation
binds the same coefficient through the immutable determinant, physical Fock
and eigenframe identities while retaining the fixed semilocal graph/scales.
This permits B3-family execution with a noncanonical coefficient without
widening CUDA's matched-composition restriction. Molecular tests compare canonical and 37/100 exchange
on H2 and asymmetric HeH+ against independently converged reference energies,
densities and complete energy-difference Hessians at three steps. These small
all-s fixtures do not qualify real-spherical Ni or installed endpoints.

Energy-only native CPU integral preparation projects each Cartesian shell
quartet directly into the public real-spherical tensor when every shell is
through f. The f quartets reuse the retained Hermite/Coulomb recurrence; adding
Ni's f polarization no longer selects a molecule-wide Cartesian ERI tensor and
a second global spherical transformation. Projection scratch is bounded by one
10-by-10-by-10-by-10 Cartesian block (80,000 bytes), in addition to the retained
component and recurrence storage. This does not bound complete preparation:
the public spherical ERI tensor still contains `nbf**4` doubles, and the host
resource planner retains its conservative setup allowances. Cartesian outputs,
derivatives and bases containing g or higher keep their existing schedules.
See the [through-f scheduling decision](../../.agents/notes/implemented/performance/2026-11-13-through-f-shell-local-spherical-values.md).

`StationaryGridSecondOrder` accepts an absent nonlocal primitive. In that case
it allocates no VV10 pair program, executes zero pair evaluations and does not
apply a VV10 pair-work cap to a semilocal calculation. The full-range
B88/VWN-RPA/LYP feature Hessian differentiates the same production expression
as native SCF, including its vacuum/tail convention; it does not inherit the
different WB97M-V feature floors. Primitive fractions remain part of the graph
and scientific identity. WB97M-V retains its existing work policy and VV10
admission. Neither this grid source nor a complete small-molecule low-level
HVP independently grants public capability: the complete molecular and installed
gates remain separate. These primitive providers do not independently promote
full-range ECP, DF or CUDA variants.

Native LDA/PBE grid partials use a separate identity-bound point-provider
contract inside the same grid/weight and molecular orchestration. The runtime
provider executes the actual SCF point worker and its directional potential
worker, including native scaled tail arithmetic and C2 spin continuation;
it does not replace that numerical domain with the interior PBE Maple graph.
The compiler remains independent of the runtime and reference engines. It
tiles the injected provider under an explicit workspace reserve, checks every
returned shape/value and composes grid-weight derivatives once. This provider
does not admit a nonlocal primitive. The private scaled CPU response ABI carries
the authenticated semilocal X/C weights; full-range K remains a separate source.
Frozen unscaled ABI contracts are preserved. PBE0 and changed-fraction molecular
tests use this same mixed J/K/XC path, not a named-functional Hessian solver.
Independent point/grid tests exercise both spins and changed X/C scales; this
alone is not a public endpoint qualification; the installed endpoint,
independent molecular, lifecycle and resource checks are also required.

The low-level CPU direct full-range candidate owner also admits scalar-ECP
RKS, while retaining all-electron RKS/UKS admission. Scalar-ECP UKS candidates
require the CPU analytic Gaussian specialization: radial power two and
local/s/p/d projectors, with the full native operator rechecked by the snapshot
owner. Other scalar-ECP UKS operators remain refused. The public scalar-ECP
exception is limited to exact linked B3LYP, not PBE0 or arbitrary full-range
compositions. The existing effective ionic charges and
shared ECP first/second-order sources are retained. For CPU radial-power-two
operators with local/s/p/d projectors and s--f orbitals, one compiler-owned
Gaussian operator supplies native primal, first and contracted second
derivatives. The snapshot token authenticates the immutable native system;
the response owner caches the fixed-density coordinate Hessian once and adds
the first-derivative contraction with the density response for each HVP. It
does not allocate a global coordinate-squared AO tensor or copy a native grid
to construct this source. Workspace and generated-block work are independently
bounded before execution, and stale snapshots fail before publication.

Candidate spin-response qualification distinguishes exact-input derivatives
from comparisons between independently converged states. Arbitrary symmetric
AO probes compare both spin outputs against an independent kernel at identical
Cartesian density fields; a minority-spin orbital node can amplify tiny state
differences beyond an otherwise valid SCF density tolerance. Independently
converged physical occupied/virtual responses and complete moving-grid molecular
Hessians remain separate required gates. The synthetic analytic-ECP test uses
XCFun only as this exact-input derivative oracle and PySCF/Libxc for independent
state and molecular checks; neither enters production or seeds the native SCF.

Other legal powers/projectors retain checked-grid sources. A finite coarse/fine
HVP validation mismatch on that fallback may use one bounded finer validation
grid; the returned HVP remains on the original native grid. Nonfinite values,
exhausted resources and unresolved quadrature error still reject. The retained CsH qualification
and its explicit domain limitations are documented in
[the bounded-validation rationale](../../.agents/notes/implemented/numerics/2026-11-04-scalar-ecp-hessian-bounded-validation.md).

The closed-shell nuclear-perturbation consumer follows the same rule. Its
canonical installed owner is `generativeqc.stationary_nuclear`: metric-density
RHS construction, occupied-orbital/density reconstruction and the single-/multi-
RHS consumer contract live there without importing repository `tools.*`.
Method-specific Fock physics remains injected through the response operator's
`induced_fock(delta_density)` contract. The shared GMRES implementation
is installed as `generativeqc.response_solver`. Repository response/Hessian
tools import these production owners directly. The tools perturbation adapter
binds `solve_stationary_nuclear_perturbation[s]` to that shared GMRES solver for
both RHF and RKS operators.

The CPU direct all-electron Cartesian LDA/PBE RKS path has installed
production owners for the live CPKS state/J/XC binding, nuclear-direction
response, NativeAO-owned generated first/second integral response, complete
seven-source molecular HVP, shared multi-RHS CPKS and raw bounded full-Hessian
assembly. The existing finite-difference, raw-symmetry and resource-admission
tests continue to qualify the same scientific contract. Repository consumers
import `generativeqc.rks_hessian_directional` and `generativeqc.rks_hessian`
directly. The same owner is exposed through
`Calculator.hessian_vector_product()` and `Calculator.hessian()` for CPU direct
all-electron Cartesian strict-FP64 closed-shell LDA/PBE RKS. Public calls retain
explicit integral/output budgets and report the executed public endpoint in
result diagnostics. WB97M-V uses the separate compiler-AD owner described below.

## Linked native CPU B3LYP endpoint

`GENERATIVEQC_ENABLE_HOST_KS_HESSIAN=ON` (the default) links the host scalar and
response prerequisites into the native core. It enables the private synchronous
`api/ks_host_hessian.hpp` bridge, the canonical B3LYP RKS/UKS Python Hessian/HVP
route, and the CPU CLI. This is independent of CUDA Hessian-AOT selection.
The initial linked electronic inventory is exact Gaussian/VWN-RPA B3LYP;
the native catalog also supplies its canonical two-body, ATM-off D3(BJ) total
program without duplicating electronic coefficients or dispersion fit data.

The bridge locks the prepared batch context, authenticates its current token,
borrows the actual Fock/basis/grid/correction owners and verifies the requested
elements, geometry, charge and multiplicity against them. It exports a verified
physical frame with an explicit owner-checked stationarity tolerance, then
constructs `HostKsNuclearHessian` on those same sources. No Python response
operator, reconstructed AO provider, runtime compiler, displaced production SCF
or reference engine executes. Missing electronic or total-method closure fails
before public Python SCF preparation; a standalone frontend correction cannot
authorize an electronic-only result under a corrected label.

Full matrices are staged through all columns and the raw symmetry gate before
either output or receipt publishes. Results are immutable detached aggregates,
not exported source-component panels. Scalar diagnostics include verified
electronic/total/D3 energies, source energy consistency, physical response
residuals, SCF work, induced-Fock actions, nuclear integral/XC visits and
whole-request D3/ECP work. The latter two child owners report their last source,
so the bridge accumulates them after each accepted column. No symmetry or
translation repair is applied. The native default response gate has
`rtol=1e-10, atol=1e-12` to handle symmetry-zero RHS roundoff; explicit solver
options retain their original gates, including a requested relative-only solve.

Separate native numerical/integral/grid/response/ECP/D3/output reservations are
enforced. The default aggregate native numerical cap is 512 MiB and nuclear
primitive-visit cap is 100,000,000 over the whole requested matrix or HVP.
`Calculator.hessian`, `Calculator.hessian_vector_product` and
`optimize_saddle` accept an explicit positive-int64 `max_eri_primitives` override
for the linked native CPU path; `None` retains that default. The saddle cap
applies to each complete requested Hessian, not each column or the whole search.
Other derivative paths refuse this override instead of silently ignoring it.
Increasing the cap does not qualify a domain or establish practical execution.
Diagnostics retain the selected cap alongside actual source visits. An internal
retained-batch action does not claim a public Calculator endpoint; only the
public Calculator runner labels its own endpoint receipt that way.
With `GENERATIVEQC_DF_PROGRESS_TRACE` set to a fresh JSONL path, the CPU bridge
also records admitted work and the actual frozen-integral schedule before
execution, first-source/frozen-integral preparation scopes, and completed
nuclear directions. Admitted work is a plan, not an executed-work receipt.
An unfinished scope or a returned direction is not a complete accepted Hessian;
the raw matrix/receipt still publish only after the final scientific gates.
The journal is best-effort diagnostic I/O: unavailable storage does not change
numerical results or turn a scientific failure into success. The ordinary
completed-result counters cannot measure unfinished preparation after a timeout.
Reservations conservatively overlap shared geometry storage; borrowed primal
SCF/provider storage, final-frame export scratch, caller request storage,
BLAS/allocator overhead and process RSS remain separate. Nuclear primitive
counts are not the integral work inside induced Fock actions. Ordered nuclear
quartet scheduling is still quartic: these finite caps do not qualify practical
25--40-atom electronic work, real Ni/I/Cs, an installed wheel, or a native TS
endpoint. Public CPU scalar-ECP B3LYP admission uses the same linked native
owner as the retained-batch qualification; complete public and installed
qualification must still execute their actual entry points. DF is excluded.
The saddle adapter remains a separate all-electron consumer: Hessian admission
does not promote an ECP transition-state endpoint.

### Molecular corrector frame ownership

The native CLI corrector delegates numerical execution to the internal
`cli::NativeMolecularEvaluator` in `src/cli/molecular_evaluator.hpp`. Its CLI
adapter owns admission, callback/error translation, progress and cancellation;
component qualification calls the same numerical owner without rewriting those
public policies. Public molecular ECP saddle admission remains closed.

Each energy/gradient call constructs the complete native molecular system,
including copied scalar-ECP potentials and core counts, and prepares a cold
native SCF/force batch. The matching raw Hessian consumes that retained batch,
not a second SCF. Exact geometry binding, occupied-spin populations and the
accepted-to-trial occupied-subspace overlap are checked before publishing an
energy/gradient. Only an explicit accepted-geometry observation advances the
root reference. Discarding a failed or rejected trial retains that reference.
Physical options and the linked method are borrowed immutable state that must
outlive the owner; basis records and frame state are owned. Invocation resource
controls may change only between synchronous calls.

For independent state qualification, `orbital_snapshot` copies the actual
retained native MO coefficients and spin/occupation shape at an exactly matching
current or accepted geometry. It performs no solve and supplies no derivative.
Its separate output budget bounds detached FP64 coefficient storage; requesting
an unmatched geometry or insufficient budget fails without changing the frame.
This diagnostic contract does not establish a matched electronic root by
itself: qualification must compare the exported occupied subspaces with an
independently converged same-Hamiltonian state, as well as energies, gradients,
raw Hessians and mode count/character. Synthetic ECP ownership tests do not
qualify real I/Cs molecular saddle consumers or practical electronic scale.

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
resource-bound test. Its generated second-integral CPU C++ cold-JIT path can
exceed the 300-second compiler limit, so this is an explicit expensive
diagnostic, not part of routine smoke tests. Passing the smaller benchmark does
not establish that cold-compile path.

## Build-time CPU scalar leaves

`GENERATIVEQC_ENABLE_HOST_HESSIAN_AOT_LEAVES=ON` builds the separate static
target `generativeqc_host_hessian_aot`, including in a CPU-only configuration.
`GENERATIVEQC_HOST_HESSIAN_AOT_INVENTORY` selects an explicit inventory; its
default is `manifests/host_hessian_aot_b3lyp.json`. This inventory packages
physical B3LYP semilocal energy/feature derivatives for both spin layouts,
compact XC coefficients and their response, mixed grid primitives, and nuclear
pair HVP leaves, first AO motion, weighted GGA field response, and frozen GGA
energy mixed geometry, plus uncut two-body D3(BJ) column leaves. Build-time
Python uses the existing compiler Graph roots;
the linked leaves invoke no Python, runtime compiler, device, or reference engine.

Enabling scalar leaves alone supplies prerequisites, **not a complete native CPU
Hessian or transition-state endpoint**. Complete B3LYP execution requires
`GENERATIVEQC_ENABLE_HOST_KS_HESSIAN`, which links the composed native owner
into the core and CLI. The leaf target itself supplies neither integral
assembly, occupied/virtual response, SCF state ownership nor molecular TS steps.
The CUDA `GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES` option remains separate.

Native development consumers include `dft/host_hessian_aot.hpp`. Registry
lookup requires both the exact program name and scientific identity. Inputs
are lexically ordered reachable Graph variables, recorded in each entry;
both input and root panels store contiguous point lanes. A leaf checks panel
counts, byte overflow and nonempty pointers before access. Zero lanes are a
successful no-op. A positive status identifies a nonfinite output root; a
negative status refuses the buffer ABI. Any nonzero status invalidates the
**whole** output panel, even if earlier lanes were written. Callers own distinct,
nonaliasing buffers and remain responsible for physical-state admission.

### Native D3(BJ) columns

`dft/dispersion/host_d3_column.hpp` supplies `HostD3Column`, an independently
linked CPU AOT energy/gradient/HVP consumer. It owns atomic-number and geometry
snapshots, uses the pinned production D3 tables and shared `D3Parameters`, and
admits only uncut BJ with ATM off, matching the TensorIR second-order domain.
It is a correction component, not an electronic Hessian, SCF or TS endpoint.
The caller remains responsible for matching its physical method parameters and
adding the correction exactly once.

One coordination jet per unordered atom pair feeds all left nuclear components.
A second traversal contracts the complete normalized-reference C6 response and
pair damping; a third scatters cached coordination jets with the accumulated
adjoints and their right tangents. All derivative arithmetic comes from the
incumbent dispersion scalar Graphs. Common-exponent scaling of normalized
Gaussian references avoids overflowing intermediate reciprocal powers without
changing normalized weights or their derivatives. The original unscaled
normalization must still be positive: geometries requiring the first-order
underflow fallback, finite cutoffs and ATM are explicitly refused.

`plan(atoms)` reports owned numeric capacity, shared static table bytes, three
pair visits per unordered pair, and seven reference slots per atom plus 49
reference terms per pair. All three explicit limits are admitted before
shape-dependent allocation. Owned snapshots, leaf scratch, cached pair jets and
simultaneous output staging are included; caller arrays, callback/object and
allocator overhead and total RSS are not. A supplied direction is snapshotted
and checked after final owner validation. Both gradient and HVP publish together
only on success; reentry and overlapping outputs fail, while a direction/output
alias is safe. `test_host_d3_column.py` compares complete D3-only columns and raw
Hessians against independent simple-dftd3 gradients, including Ni/I/Cs and
25/40-atom correction controls. Those controls do not qualify the electronic
work, memory or time of a molecular endpoint.

`dft/host_xc_response.hpp` supplies `HostGgaXcResponse`, a synchronous native
consumer of the linked XC and coefficient-response programs. It contracts
borrowed real AO values/first derivatives, fixed weights and stationary density
matrices into a semilocal AO Fock response. RKS uses one total-density matrix;
UKS uses two occupation-one spin matrices with cross-spin XC coupling. Both
program identities must match exactly. Spin densities must be nonnegative;
zero spin density requires zero spatial gradient. The linked production graph
owns its lazy vacuum branch and density threshold, including zero derivatives
inside that branch. Undefined/nonfinite derivatives outside it still fail;
no density or tangent floor is introduced.
No reference engine, runtime Python or runtime compilation participates.

The caller supplies a bounded consecutive-point tile reader and a live-state
validator covering geometry, grid, method, representation and spin identity.
Mandatory streamed numeric panels are admitted before allocation, including
transactional output staging. Their workspace query scales as AO-matrix plus
tile-point storage, not full-grid storage. On supported x86 CPUs, runtime-dispatched
AVX2 contractions retain each point's ordered AO-pair reduction and original
multiply/add grouping, with contraction disabled. Lanes are separate field
components or output columns, not terms of a horizontal reduction. Field packing
reuses coefficient-input scratch before those inputs become live; a small tile
or unsupported CPU retains scalar fields without extra allocation. Both paths
recompute primal fields for exact bank authentication and preserve fully ordered,
possibly nonsymmetric density/direction matrices. No global ISA flags, triangle
copy, symmetrization, or reassociation through BLAS is used.

An optional separately admitted bank retains exact linked XC inputs
and output partials for the largest whole-tile point prefix fitting its byte
allowance. The first successful action prepares this prefix; an incomplete
preparation is never reused. Later actions compare every current XC input bit
before replay. A mismatch fails rather than silently refreshing the bank.
Unretained points keep the original scalar evaluation, and insufficient capacity
or optional allocation failure retains the streamed path. No AO matrices,
response tensor or new XC formula are cached. Borrowed state, reader/provider
workspace, callback objects and process
RSS, compiled scalar stack and allocator overhead are not part of that scratch
query. Every action rereads and contracts the
full grid, with work proportional to grid points times squared AO count; bounded
scratch is not a scaling claim. Diagnostics retain attempted/completed actions
and point/tile visits, submitted scalar point slots and exact replayed slots.
A failed scalar batch can terminate before all submitted slots are evaluated.
Even a complete bank still rereads/authenticates every AO tile and recomputes
the current point inputs; it does not bypass owner or nonfinite checks. Failure
leaves output unchanged, releases the invocation
guard and permits recovery with a valid owner. Direction/output aliasing is
supported; reference-density/output aliasing and overlapping/reentrant calls
are refused. This is only fixed-grid semilocal response: direct J/K, metric and
moving-grid/integral derivatives, coupled orbital solves and a complete native
stationary owner remain separate requirements.

## Native CPU fixed-geometry CPKS

`GENERATIVEQC_ENABLE_HOST_KS_RESPONSE=ON`, together with the explicit host leaf
inventory, builds the optional static `generativeqc_host_ks_response` target.
Independently linked fixture consumers require a CPU-only native core. The
`GENERATIVEQC_ENABLE_HOST_KS_HESSIAN` integration instead links it into the
core with the core's actual backend definitions; runtime execution still
requires a CPU prepared owner. Enabling the prerequisite alone does not
authorize public Hessian or TS admission.

`dft/host_ks_response.hpp` supplies `HostKsResponse`. It borrows an authenticated
`VerifiedKsFinalState`, the matching normalized system, the ordinary CPU FP64
exact `PreparedFockPlan` and a bounded native AO/grid tile reader. The linked
registry's exact MethodIR identity binds the physical functional, spin layout,
SCF-domain version, full-range exchange fraction and both scalar program
identities. An incomplete scalar closure publishes no method binding. Grid
point count and tile size must match the verified model prescription. Occupied/
virtual gaps must exceed the explicit positive-gap admission threshold.

The consumer uses the existing shared two-spin orbital response action. RKS is
its singlet sector with total-density J/K/XC conventions; UKS retains both
occupation-one spin channels and cross-spin coupling. Induced Fock matrices
reuse the prepared native J/K source, common Fock assembly and linked fixed-grid
XC action. The ordinary native GMRES solver uses orbital-gap diagonal
conditioning without modifying the physical operator or true-residual gate.
No new SCF, runtime Python/compiler, reference engine or alternate functional
formula participates in this consumer.

Its numeric budget composes owned adapter/XC panels, a conservative bound for
shared orbital-action arrays and native GMRES storage. Optional exact XC partial
reuse is admitted before Fock actions, after actual GMRES capacity and the
composed source panels are reserved, inside the original response cap. The
native nuclear-response reader then optionally admits a whole-tile prefix of
fixed-grid AO values and first derivatives from the remaining capacity. Its
actual numeric allocation includes a packed-basis snapshot, point coordinates
and four AO components; separate capacity and evaluated/replayed-point counters
are available on the nuclear owner. Hits compare basis/coordinate bytes and the
complete floating environment and preserve the evaluator's floating flags.
Weights remain live, both frame-authentication guards remain, and generic XC
continues to read/authenticate primal fields even when its partial bank hits.
Uncached tails, input/environment misses and optional allocation failures keep
the original streamed AO evaluator, without increasing either response or
aggregate caps. Admission returns the combined newly owned cache bytes. The
perturbation and nuclear owners report the resulting dynamic owned bytes rather
than keeping a stale construction-time response estimate. Public full-Hessian
and HVP entry points attempt this allocation only after complete integral work
admission and before any numerical source preparation. Borrowed SCF/Fock/grid
storage, internal provider/BLAS workspace, metadata, scalar stack and allocator
overhead remain separately owned; this is not a process-RSS or complete endpoint
bound. Diagnostics retain J/K/XC and Krylov work. Output publication is
transactional, including nonconvergence and workspace refusals. Output may
alias its input but not any borrowed stationary frame array. Owner/epoch
invalidation and overlapping/reentrant calls fail closed and permit recovery.

Independent native response batches may share up to four physical XC actions.
Single and batched requests use one resumable GMRES numerical owner, with
separate per-RHS Krylov state and a newly evaluated physical action at every
requested true-residual checkpoint. This is not block GMRES. The common spin
density/projection maps, prepared J/K builds and compiler-owned nuclear
metric/RHS/D/W/physical-residual maps remain the scientific owners. RKS retains
the two-spin singlet embedding; UKS retains its coupled spin response. Portable
scalar contraction fallbacks and two/three-direction tails are supported.

Full raw matrices select the independent batch width before allocating XC
partials or the AO prefix. Admission includes every simultaneous Krylov state,
per-direction source/map/D/W panel, explicit XC packing workspace and staged
assembly column inside the original response/aggregate limits. Additional
assembly uses only unused output/aggregate allowance; insufficient capacity or
optional allocation failure retains the ordinary scalar column schedule.
Standalone HVPs retain that schedule. The physical column assembler and raw
symmetry gate are shared, and per-direction ECP/D3 work is retained separately
for the completed endpoint receipt. Batch failure publishes no partial group.
Neither compact component controls nor a component timing qualifies an installed
heavy-element Hessian, modes or a transition-state endpoint.

This action is **fixed-geometry electronic response only**. Small native
RKS/UKS response controls do not qualify installed Ni/ECP, mixed-element or
production-scale complete Hessian/TS endpoints.

### Native metric and external-field response

The same optional static target supplies `HostKsPerturbation` from
`dft/host_ks_perturbation.hpp`. It borrows `HostKsResponse` and accepts one
symmetric AO overlap direction plus one symmetric fixed-density Fock direction
per physical spin. Callers must provide every external contribution; these
inputs are not inferred from geometry. `HostKsResponse::apply_density` exposes
the matching physical induced Fock action for occupation-two RKS total density
or the two occupation-one UKS densities.

Build-time `tools/generate_host_ks_cpks.py` lowers the canonical KS TensorIR maps
into checked, runtime-sized FP64 CPU kernels. The compiler, not the native
consumer, owns metric orbital motion, the coupled RHS, total density motion
and the full occupied-energy-matrix weighted-density tangent. Off-diagonal
occupied energy motion is retained, including degenerate occupied spaces.
The native consumer invokes the same coupled GMRES and recomputes the physical
residual with a fresh induced Fock on the total density direction. Both D/W
outputs publish together only after that residual passes the original solve
tolerances and the borrowed frame remains authenticated.

The mandatory numeric limit composes retained D/W/map panels, a single reused
map arena, child response/action arrays and GMRES storage. The mandatory map
work limit bounds the materializing TensorIR schedule, not J/K, grid, provider
storage, process RSS or complete endpoint work. Diagnostics expose planned
capacity, scheduled map terms and attempted/completed directions. Resource
refusals, invalid inputs, nonconvergence and lifetime/reentrancy failures leave
both outputs unchanged; valid calls can recover. The child owner must outlive
this synchronous consumer and may not be rebound while it is borrowed.

This module does **not** provide nuclear fixed-density Fock directions, moving
grid/integral/ECP derivatives, second derivatives, complete Hessian assembly
or a native CPU TS CLI. Abstract overlap/field displaced-SCF controls qualify
the response composition, not a nuclear or installed molecular endpoint.

## Native CPU moving-grid XC source

The same optional native response target includes `HostGridGeometry`
(`dft/host_grid_geometry.hpp`) and `HostGgaNuclearXc`
(`dft/host_xc_nuclear.hpp`). These development consumers provide one actual
fixed-density semilocal nuclear Fock direction, not a complete stationary
Hessian. They are not installed into or enabled in public core/CLI capabilities.

The geometry reader borrows one immutable normalized `AoBasis` and
`MolecularGrid`. Native ordinary spatial jets through order two feed a linked
compiler Taylor-AD leaf for moving AO values and first spatial derivatives.
Grid points follow their owning nuclei; relative AO displacement is point-owner
motion minus AO-center motion. Weight directions reuse the
[retained partition derivative](dft_grid.md#native-weight-direction-tiles).
Exact linked identity, packed basis metadata, matching nuclear centers,
pointer/shape stability and the mandatory owner validator are checked before
publication. The validator must additionally reject in-place content/epoch
changes. A successful tile publishes all AO/AO-motion/weight/weight-motion
outputs together. External direction aliasing is allowed; output/output and
borrowed-frame aliases are not.

The XC source borrows immutable total RKS or occupation-one UKS density matrices
and requires the exact linked MethodIR/spin/field-response closure. Canonical
TensorIR feature JVPs include moving AO factors at fixed D; production scalar
XC gradients/Hessians and the shared Cartesian coefficient JVP produce weighted
primal and moving cotangents. The potential JVP includes both moving matrix AO
factors and those moving cotangents, without weighting twice. Only generic AO
contractions are reused from WB97M-V compiler maps: its point-field floors and
nonlocal policies do not enter this GGA source. The linked production GGA vacuum
and spin-endpoint branches remain unchanged. Nonfinite derivatives fail closed.
No reference engine, finite difference or runtime compiler participates.

The CPU build-only maps explicitly select the shared AO contraction schedule:
the generic TensorIR planner may use degree-neutral binary contractions, with
new intermediates bounded to one point axis and one AO axis. Exact
common-subexpression elimination runs before AD and on the resulting JVP/VJP
programs, sharing density-times-AO panels across ingredient outputs and derivative
legs. This changes FP64 reduction order, not AO normalization, spin occupancy,
signed density/response semantics or the linked XC formula/domain. Default
compiler consumers retain their canonical direct programs. Runtime admission
prices the selected generated schedule and complete live workspace at actual
AO/tile extents; the rank bound alone is neither a work nor a memory guarantee.
`tests/python/test_host_grid_maps.py` compiles direct and shared maps separately
and checks signed primal/mixed/relaxed actions, scalar-stencil controls, actual
work queries, and transactional workspace/nonfinite refusals. The physical
AO/Libxc component controls below remain required for changed reduction order.

These grid-map generators also explicitly select an ordered binary execution
schedule in `tensor/host_affine.py`. For exact-unit, rank-two binary contractions
with one reduced label, `src/tensor/ordered_binary_contraction.hpp` uses runtime
AVX2 detection to process four independent output columns. When either operand
is invariant across output rows, four rows share that operand's loads using
independent register accumulators. Small extents and layouts without a shared
row operand retain the single-row schedule, including scalar row/column tails.
Empty reductions never advance or read potentially null input pointers.
Each lane preserves
the shared graph's operand grouping and serial reduction sequence, including
vacuum-boundary decisions; there is no horizontal reduction, FMA, BLAS call,
packing allocation or change to mathematical work and arena queries. Other
nodes, nonunit coefficients and unsupported CPUs retain the original scalar
lowering. Other host-affine consumers default to that scalar schedule.
`test_host_affine_ordered.py` and `test_host_grid_maps.py` execute both schedules
and the forced-scalar fallback, including signed/nonempty density response,
subnormal and cancellation controls, vector tails, overflow, alias/capacity
refusals and transactional publication. This is component execution
qualification, not a complete Hessian or optimizer performance claim.

Mandatory limits admit composed FP64 numeric panels (including the geometry
reader), complete-source AO-point and grid-pair visits, and generated contraction
terms before execution. Scratch grows with tile size and AO matrices, not the
full grid. Metadata, borrowed basis/grid/D, provider/scalar stack, allocator
overhead and RSS remain outside that numeric query. Visit/term counts do not
claim a bound on every validation/provider instruction or endpoint wall time.
Attempted/completed directions, completed tiles/points and scheduled map terms
are exposed separately. Failures never publish a partial spin matrix and release
the invocation guard for recovery; one consumer must not overlap or reenter.

`tests/python/test_host_xc_nuclear.py` compares actual normalized spherical
s/p/d/f AO motion with independent libcint displaced-basis/point evaluations,
and complete frozen XC Fock directions with independent AO/Libxc moving-grid
differences for RKS/UKS and grid versions 1--4. These are source-component
controls, not stationary molecular or installed endpoint qualification. Integral
directions use the separate consumer below; residual ECP projectors, second
source derivatives, complete Hessian assembly and Python-free CPU TS execution
remain separate requirements.

## Native CPU mixed AO/grid geometry

The optional response target also includes `HostGridMixedGeometry`
(`dft/host_grid_mixed_geometry.hpp`). It borrows the authenticated ordinary
geometry reader and the exact exported pre-partition atomic measures. Its tile
returns primal order-one spatial AO jets, their independent left/right/mixed
nuclear directions, retained weights, and the corresponding three weight
directions. Points follow their owning atoms. The topology, atomic measure,
element radii, and radial/angular prescriptions stay fixed under differentiation;
versions 1--4 and partition iterations 1--5 retain their original policies.

Order-three normalized native AO spatial jets feed the canonical compiler Taylor
program. `tools/generate_grid_geometry_response.py --backend cpu` lowers the same
mixed norm/ratio/Becke/product graphs as the CUDA consumer, without importing a
device runtime or running a compiler at execution time. Complete product jets
share exact binary rescaling before partition normalization, including when a
zero primal has nonzero derivative fields. Normalization never divides by a
possibly zero partition weight. Every tile independently reconstructs its primal
partition and compares it with the retained weight in dimensionless units before
joint output publication. Separated nuclei and nonzero point/nucleus distances
are required; non-smooth coincidences fail rather than inventing a Hessian branch.
Signed pre-partition atomic measures, present in some default Lebedev rules,
retain their sign: only finiteness, zero consistency and the dimensionless
partition are admission gates. No clipping or absolute-value measure enters.

Mandatory caps include both geometry readers' owned FP64 panels and mixed-jet
scratch. Atomic measures remain explicitly borrowed, authenticated storage, not
a hidden full-grid temporary. Integer metadata, borrowed arrays, scalar stack,
allocator overhead and RSS are outside this numeric bound. Each direction pair
visits nuclear pairs once for center geometry, then once per requested point;
each point/AO spatial jet is evaluated once, not once per left coordinate. These
are per-tile work counts, not complete endpoint performance claims. Aliasing,
stale owners, nonfinite values and capacity failures leave all four outputs
unchanged; external direction/output aliases are supported.

Independent libcint primal AO differences and high-precision direct-product
partition differences validate this geometry component. It does **not** supply
frozen XC energy curvature, second integral/ECP sources, stationary relaxation
assembly, a complete molecular Hessian or a native CPU TS endpoint.

### Whole-column moving-grid source

`HostGridColumnGeometry` (`dft/host_grid_column_geometry.hpp`) provides the
source-driven alternative needed by complete-column consumers. A tile returns
ordinary normalized spatial AO jets through order three and partitioned measures
in `[value,right,gradient(3N),Hright(3N),point]` layout. Spatial AO jets are
evaluated once per tile. Each point records the canonical Becke source once and
reverses its right JVP once; physical nuclear leaves and the grid-point owner
receive their contributions directly. It does not replay the molecular source
for each left coordinate or retain coordinate-squared AO/weight arrays.

Local Jacobian and directional-Jacobian coefficients come from fixed-arity unit
seeds of the same compiler scalar jets. Forward products and reverse adjoints
carry independent exact binary scales. Zero product values must not erase a
nonzero derivative or force a large scale to materialize before cancellation.
The retained-partition gate, signed measures, radius/iteration prescriptions and
coincidence refusals are unchanged. Mandatory per-tile caps separately cover AO
visits, center/point pair visits and forward-plus-reverse tape nodes; tape nodes
are semantic schedule visits, not floating-point operation counts. Capacity
includes the borrowed reader's owned scratch, staged outputs, right snapshot and
bounded tape including its metadata, but not allocator overhead or RSS.

Both outputs publish only after final owner and direction authentication. The
reader supports external direction/output aliases, not output/output or borrowed
state aliases. Independent order-three libcint jets and high-precision displaced
primal partitions qualify this source. It is not itself an XC energy contraction,
a stationary Hessian, an electronic performance gate or a native TS endpoint.
The scalar-pair XC consumer below retains the mixed reader. The column XC
consumer instead contracts this reader's shared spatial jets with generated
ingredient pullbacks, without replaying geometry for each left coordinate.

## Native CPU frozen-density XC curvature

`HostGgaNuclearCurvature` (`dft/host_xc_curvature.hpp`) composes the mixed geometry
reader into the complete semilocal energy's frozen-density left/right/mixed
nuclear partials. Its four scalars are energy, left partial, right partial, and
mixed partial. Both moving AO legs, Cartesian density/gradient fields, cross-spin
XC curvature and moving quadrature measures are included exactly once. The exact
linked MethodIR/spin binding must explicitly contain `gga_energy_mixed`; missing
closure is a refusal, not runtime compilation or a different functional.

Shared canonical TensorIR AO feature JVPs explicitly exclude density response
from both mixed legs. Production scalar ingredient gradients/Hessians, Cartesian
ingredient mixed AD and the quadrature product AD supply the scalar chain rule.
The parent stationary Hessian must separately add the actual D'/W' relaxation and
all non-XC sources. This source authenticates its arbitrary symmetric frozen
density, not its stationarity. It preserves production vacuum/spin-endpoint
branches and uses compensated FP64 grid accumulation with signed weights.
The existing full ingredient-Hessian response domain remains authoritative:
nonvacuum empty-spin endpoints can refuse, even when a particular restricted
frozen-density path has a finite limit. This component does not invent or
qualify that additional directional limit; refusal remains transactional.

Mandatory caps cover the complete source's generated map terms and AO/grid visits
and the composed FP64 panels, both geometry readers and one reused map arena.
Borrowed arrays, metadata, scalar/provider stack, allocator/RSS and endpoint time
remain distinct. Stale owners, aliases, capacity failure, nonphysical Cartesian
spin fields and nonfinite derivatives refuse without publishing any scalar.
There is no displaced production evaluation, external electronic-structure
engine, Python callback or runtime compiler in this source.

Independent multistep libcint/Libxc primal energy controls use exactly the retained
grid prescription and frozen D, not displaced SCF response. They are component
qualification, not an electronic molecular Hessian or TS endpoint. Integral/ECP
second sources, stationary relaxation assembly, nuclear/D3 composition and native
installed CPU endpoint admission remain separate requirements.

## Native CPU XC nuclear columns

`HostGgaNuclearColumn` (`dft/host_xc_column.hpp`) composes the moving-grid column
reader with the exact linked `gga_energy_mixed` MethodIR/spin closure. Its output
is `[energy, energy_right, partial_R(3N), partial_R_right(3N)]`. The left nuclear
partial holds the ground density fixed. An empty `density_right` means frozen
density; a supplied finite symmetric spin/AO matrix adds the right density
response exactly once, together with the geometry direction. For supplied
`D_right`, `energy_right` includes the density contribution and need not equal
`partial_R` dotted with the nuclear direction. RKS uses total density and UKS
uses occupation-one spin densities. This source neither solves nor certifies
the supplied response or ground-state stationarity.

Canonical TensorIR feature JVPs and AO-feature VJPs/JVPs consume the retained
spatial jets. One baseline and `4*spins+1` fixed field/weight seeds of the exact
mixed-energy closure recover weighted field adjoints and their right directions.
The complete closure work is `(4*spins+2)*points`, independent of the number of
left molecular coordinates. Three fixed spatial seeds per AO scatter adjoints
to the grid-owner and basis-center coordinates with opposite incidence. Geometry
and primal/right features are shared rather than replayed for every coordinate;
no coordinate-squared AO tensor, handwritten XC Hessian, numerical displacement,
runtime compiler or reference engine is used in production.

Mandatory capacities cover composed numeric scratch, an owned `D_right`
snapshot, compensated O(atoms) output, generated map terms, scalar-point,
AO-point/motion, grid-pair and tape visits. Tape admission conservatively charges
the child reader's full per-tile bound for every tile. Semantic visits are not
scalar FLOPs or endpoint timing; borrowed arrays, provider stack, allocator
overhead and RSS are outside the numeric query. Signed weights and the linked
full ingredient-Hessian vacuum/spin domain are unchanged. Stale owners, mutation,
reentry and capacity failures publish nothing and permit recovery. Output may
alias either external direction input, but never ground density or geometry;
both direction contents are authenticated before final publication.

Independent libcint/Libxc displaced-primal controls cover all left components,
moving-grid versions 1--4, Cartesian/spherical AOs through f, RKS/UKS, and frozen
or explicitly supplied right density. Frozen controls also check raw symmetry,
linearity and rigid zeros without repair. Arbitrary supplied density responses
do not imply stationary symmetry. Synthetic source-work controls and a linked
Python-free executable are component qualification only: actual stationary
S/F/D/W response assembly, all other energy sources, complete electronic
25--40-atom endpoint resource gates and installed native CPU TS/frequencies
remain separate requirements.

## Native CPU frozen-integral nuclear source

The optional native KS response static target also provides
`HostKsNuclearIntegrals` (`dft/host_nuclear_integrals.hpp`). It returns one actual
overlap direction and the frozen-density integral Fock direction
`[T + V_eff + J + K]'` for normalized Cartesian or real-spherical s/p/d/f AOs.
This development component requires an exact linked full-range global-hybrid
MethodIR/spin and the actual CPU FP64 exact J/K model. It is not a stationary
Hessian, a public capability admission, or an installed endpoint qualification.
`V_eff` here is the ionic point-charge potential only: local and nonlocal
residual ECP projector derivatives must be added separately for an ECP system.

S/T/V gradients reuse the shared production CPU one-electron AD helpers. ERI
gradients use the existing packaged s/p/d derivative artifacts, sharing native
canonical component/axis restoration with stationary shell contractions. This
gradient-only adapter uses a common coordinate origin to retain small relative
separations, but returns all twelve independent mathematical-center partials.
Unsupported classes or unavailable artifacts retain the compiler-generated
factorized first-only specialization, including its through-f work count; a
numerical/domain failure is never silently retried as a capability fallback.
Public Cartesian/spherical normalization and radial contraction coefficients
remain in the external weights exactly once. The distinct curvature consumer
still needs the scalar integral value and retains its original thirteen-field
factorized value/gradient operation. No primal is faked and no additional
recurrence, generated shard ABI or mixed derivative is introduced. The coupled native
target still requires `GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=ON` for its
ordinary stationary force owner; this source no longer generates or borrows that
owner's older primitive-derivative declaration. Independent gradients remain separate until a
linked scalar Graph leaf projects normalization and nuclear-slot motion.
Every distinct ordered public AO quartet incidence is consumed once. Density VJPs and
integral-only JVPs of the canonical stationary objective supply all four Fock
incidences, including the method's spin, occupation and exchange factors.
There is no native duplicate of the scientific J/K equation, runtime compiler,
finite-difference path, or reference-engine fallback.

The caller supplies an immutable normalized `System`, matching packed
`AoBasis`, actual strategy, total RKS or occupation-one UKS density matrices,
and a mandatory current-owner validator. The validator authenticates all
in-place content/epoch changes and stationarity when that is required by its
enclosing owner; this integral source does not certify stationarity itself.
Finite symmetric indefinite densities are admitted. Pointer/shape, shell and
primitive layout, nuclear centers, exact method/model, finite values and
resource limits are checked. Outputs publish together only after final owner
validation. Reentry, output/output aliases and borrowed-state output aliases
fail without publishing; an external direction may alias an output.

Mandatory resource limits bound owned FP64 numeric panels, one-electron
primitive visits, ERI primitive gradients and generated quartet terms. If
`M` is the sum over public AOs of radial primitive count times Cartesian
expansion term count, the complete source schedules `M^2 * (natoms + 1)`
one-electron visits. The default `HostNuclearEriSchedule::SymmetryUnique`
evaluates one canonical real-scalar ERI per eightfold AO symmetry orbit and
preserves every distinct ordered compiler weight/scatter incidence. With
`t_i` radial/Cartesian terms on AO `i`, enumerate triangular AO pairs in order,
set `w_(ij) = t_i * t_j`, and bound primitive evaluations by
`sum_p w_p * sum_(q<=p) w_q`. Planning this bound takes O(nao^2) work, not an
AO-fourth-power traversal. For one primitive s AO per site this is
`P * (P + 1) / 2`, where `P = nao * (nao + 1) / 2`.

`HostNuclearEriSchedule::Ordered` retains the bounded ordered comparison path
with `M^4` ERI gradients. Select the same schedule in `plan` and options;
limits always admit the actual selected schedule. Both schedules retain
`nao^4` ordered compiler-map incidences and their generated per-quartet work.
First sources reuse the canonical directional integral for all incidences;
when all four physical centers have exactly equal displacement vectors, the
directional ERI is identically zero and its primitive gradients are omitted.
This uses exact equality, not an amplitude/integral threshold. Resource plans
remain conservative, direction-independent upper bounds; execution diagnostics
report actual primitive visits. The ordered comparison path remains exhaustive,
and every ordered weight/scatter and owner-validation cadence is preserved.
This omission applies only to the frozen-density first source: even with a
zero pure integral direction, curvature can need first gradients contracted
with the explicit density/energy-weighted-density response.
Curvature sources sum the generated weights and explicit moving-D weights
before evaluating and scattering the canonical physical nuclear jet. There is
no screening, resident ERI cache, new spin/exchange formula or changed operator.
Borrowed data, integer metadata,
provider/scalar stack, allocator overhead and RSS are outside the numeric
query. These explicit quartic work limits are not a favorable-scaling or
complete-endpoint performance claim. Owner validation occurs after each AO
pair, at a bounded quartet cadence, and before publication.

`HostNuclearFirstSchedule::CoordinatePanel` optionally reserves all Cartesian
S' and frozen integral F' columns. A single primitive sweep reuses each
mathematical-center gradient across its physical atom owners. The generated
integral-only JVP is linear: its unit-tangent coefficients are evaluated once
per ordered incidence, then applied to the physical partials. Repeated owners
are accumulated before scatter. A prepared panel evaluates arbitrary directions
by bounded linear combinations, with no further primitive or quartet-map work.
This is a first-source coordinate/AO panel, **not** an AO-fourth-power derivative
tensor or a resident ERI cache; curvature still streams one column at a time.

The native full-matrix endpoint attempts this schedule only before its first
source call, from unused `integral_bytes` and inside the existing composed
response reservation, including GMRES. It retains and accounts for the original
streamed child without increasing that child's cap. Insufficient panel/response
capacity, allocation failure during optional construction, or a density outside
the unit-coefficient range selects the unchanged streamed schedule. The unit
map requires `abs(D) <= 1e50`; larger finite indefinite densities remain
supported by streamed directions rather than risking new unscaled coefficient
overflow. Both selected schedules must pass complete-request primitive/map work
admission **before** integral preparation. A failed numerical preparation or
owner check propagates; it never silently switches scientific paths.

Panel storage accounting includes all columns, replay staging, the external
direction snapshot, and exact snapshots of density, packed basis and ionic
charges. Supplied D'/W' contraction additionally reserves both input snapshots
and the generated map's full AO-pair workspace. Original owner/lifetime checks
remain active throughout preparation
and replay. A partial preparation cannot be replayed. The existing C ABI and
resource fields are unchanged; receipts report actual selected first-source
primitive work and the composed fixed storage bound. HVP-only requests retain
the streamed schedule. This reuse does not qualify complete-electronic scaling
or molecular saddle consumers without separate endpoint evidence.

`tests/python/test_host_nuclear_integrals.py` compares complete frozen-D
directions against independent normalized libcint primal integral differences
at three steps for both spins and AO representations through f. Its Ni ionic
point-charge control does not test ECP projectors. Translation, linearity,
model/resource refusal, lifecycle failures and recovery are separate controls.
The optional CMake executable also runs the native integral source without
Python on PATH or PYTHONPATH. Neither control qualifies residual ECP second
derivatives or complete stationary/TS assembly.

### Bounded mixed S/T/V primitives

The optional native KS response target also links
`integrals/host_mixed_primitives.hpp`. Its `overlap_kinetic` and `attraction`
functions evaluate unnormalized Cartesian primitive pairs through f, publishing
`[value, left, right, mixed]` per operator. The attraction operator includes the
physical minus sign for a unit positive point charge. Its independent third
center is the attracting nucleus, not an AO center. Both nuclear-motion legs
use independent mathematical slots, including at coincident centers.

`tools/generate_one_electron_mixed.py` lowers exact Hermite/Coulomb factors of
the existing one-electron DAG with the shared scalar AD and Coulomb derivative
algebra. Generation is build-time only. Common-origin coordinates and motion
legs preserve exact rigid-translation zeros. Execution uses fixed-size stack
storage and existing native range moments, without runtime graphs, JIT,
displaced work or a reference engine. Invalid powers, nonpositive exponents,
nonfinite inputs or nonfinite arithmetic return `false` without publishing any
output; otherwise external input/output aliases are allowed.

These are primitive components, **not** a complete frozen integral second
source or stationary Hessian. The enclosing source still owns contracted-basis
normalization, spherical projections, ionic charges, method-authorized D/W
weights, ERI second derivatives, work/storage limits and owner authentication.
`tests/python/test_host_mixed_primitives.py` checks all Cartesian component
pairs against independent analytic libcint values, first derivatives and mixed
Hessians, including coincident, tight/diffuse and asymmetric-exponent cases.
It separately tests bilinearity, bra/ket and leg symmetry, translations,
transactional refusal, aliases, recovery and native execution with Python
absent from PATH. These gates do not admit a complete molecular endpoint.

### Integral Lagrangian curvature

`HostKsNuclearIntegralCurvature` in `dft/host_nuclear_integrals.hpp` consumes
the same authenticated normalized system, public AO basis, exact CPU
full-range strategy and linked MethodIR as the first-integral source. It also
borrows immutable energy-weighted density W with the same spin/row/column
layout as D. RKS D is total density; UKS uses occupation-one spin blocks.
Finite symmetric indefinite D/W are allowed. The enclosing owner's validator
must authenticate **both** panels, including their contents and epochs; this
component does not establish stationarity itself.

For two atom/xyz motion legs, `apply` returns four scalars
`[L, L_left, L_right, L_mixed]`, with the frozen integral Lagrangian
`L = Tr(D (T + V_ionic)) + E_JK(D) - Tr(W S)`. Canonical
`StationaryGradientPlan.integral_block` weights supply the one-electron and
constraint contractions. The integral VJP of the shared ordered J/K energy
objective supplies the quartet weight. Native code does not reimplement spin,
occupation, exchange or Pulay coefficients. Weights are evaluated once per
ordered AO pair/quartet and contracted directly into normalized primitive
jets, including the Cartesian expansion terms of real-spherical AOs through f.

S/T/V use the mixed primitives described above. ERIs use the canonical
build-time factorized value/gradient/gradient-JVP library with a common-origin
frame for both motion legs. A compensated four-field accumulator avoids a
molecular AO-fourth-power or coordinate/AO derivative tensor. Execution has no
runtime compiler, displaced integral evaluation or reference-engine fallback.

In the symmetry-unique schedule, a quartet with **exactly zero relative right
motion** explicitly requests the same compiler's value/gradient-only lowering.
Its mixed integral field is exactly zero, but its value and all twelve
mathematical-center gradients are retained: moving D/W weights can still give a
nonzero physical curvature column. No quartet or ordered weight-map incidence
is omitted. Arbitrarily small distinct right motions keep the mixed primitive;
no tolerance or screening is introduced. The ordered comparison schedule always
uses full mixed jets. This is consumer-side operation selection, not a change
of the generated primitive's output packing based on a zero direction.

When all four AOs have the **same physical atom owner**, every left nuclear
scatter also cancels identically. The symmetry-unique consumer requests the
same compiler's explicit value-only lowering for these quartets. Scalar value
and moving-density consistency fields remain; no quartet or weight incidence
is skipped. Distinct physical atoms must not be merged because their numeric
coordinates or motion happen to coincide. The ordered comparison keeps full
mixed primitives even in this case.

The internal work record separates value-only, first-only and mixed ERI calls;
their sum remains the actual primitive-call count bounded by the same conservative
admission cap. Mixed calls include directional and independent-center matrix
operations; `eri_matrix_primitives` identifies the matrix subset. These operations
return different amounts of derivative algebra, so their call counts alone are
not comparable floating-point work or endpoint timing. Curvature's public
aggregate ERI count includes all operations.
Heap/work plans and owner-validation cadence do not shrink when lowering is
possible. This reduces derivative order, not quartet enumeration or asymptotic
work, and does not independently qualify practical molecular scale.

`plan` reports owned FP64 heap staging plus generated weight-map arena, full
one-electron/ERI primitive visits and generated map work. All four bounds are
mandatory before allocation/execution. Borrowed panels, integer/AO metadata,
primitive stack, allocator overhead, RSS and endpoint timing remain outside
this numeric query. Mixed-pair one-electron visits are
`M^2 * (natoms + 1)`, where M counts radial primitives times Cartesian expansion
terms over public AOs. ERI visits follow the selected symmetry-unique or ordered
schedule described in the first-integral source contract above.
Map work includes both pair weights and quartet weights. This explicit quartic
bound is not a favorable-scaling or complete-endpoint performance claim.

For a whole physical-coordinate column, explicitly select
`HostNuclearIntegralCurvatureSchedule::Column` in **both** `plan` and the
constructor, then call
`apply_column(right, density_right, weighted_density_right, output)`; empty
response spans select frozen D/W. The result is
`[L, L_right, gradient(3*natoms), mixed(3*natoms)]`. A mixed-pair owner cannot
execute this schedule, and a column owner cannot execute scalar `apply`.
The mandatory numeric bound retains compensated O(natoms) scalar fields and
snapshots of the external nuclear direction and spin/AO D/W right responses,
not coordinate/AO derivative panels. Each ordered quartet and its generated
weight/JVP is visited once per right direction; the same ERI gradient/JVP scatters
to every physical left coordinate before proceeding to the next primitive.
The final mathematical ERI slot is the common origin. Bindings that coincide
with this origin omit opposing scatter updates; relative right legs preserve
exact rigid-right-motion zeros.

S/T/V use a finite mathematical-slot schedule: three mixed S/T evaluations
and six mixed attraction evaluations per radial Cartesian expansion, then
immediate physical scatter. Their primitive bound is
`M^2 * (3 + 6*natoms)`; ERI work is unchanged from a scalar pair, while the
column map ceiling also charges its canonical density/constraint weight JVPs.
The fixed-slot replay reuses the canonical primitive authority but is counted,
not described as a single S/T/V evaluation. No whole-source replay occurs
for each left coordinate. In this bounded column schedule, a full frozen matrix requires one source sweep per
right coordinate, rather than one per coordinate pair; this does not remove
the ordered-quartic ERI work or qualify a complete electronic endpoint.

The optional `HostNuclearIntegralCurvatureSchedule::Matrix` instead computes
`[L, gradient, raw Cartesian frozen-D/W Hessian]` in one ordered-weight sweep.
Each canonical ERI holds mathematical center D fixed. In the symmetry-unique
schedule, constant physical-owner incidence contracts the existing factor
partials before their product composition: two, three or four physical owners
require three, six or nine independent coordinates, respectively. The compiler
emits the value, those first partials and all nine, 36 or 81 ordered mixed
partials through the existing scalar Graph/AD. No transpose is copied to
construct a mixed derivative. Cartesian-axis coefficients retain their disjoint
derivative support until their three-factor scalar product is composed. The
same Graph/AD lowers each ordered derivative before materializing the dense
physical-owner jet for the unchanged Coulomb and Gaussian-prefactor products.
Numerical zero coefficients remain ordinary leaves; no new screening, recurrence,
or persistent cache is introduced. The consumer scatters each independent physical
coordinate onto its atom and the common origin. Owner labels follow first
occurrence, with D's owner last; equal labels require equal coordinates, but
coincident distinct atoms are never merged. The
symmetry-unique schedule requests only a scalar value when all four physical
owners are identical; the ordered comparison retains the full matrix operation.
No Hessian block is
symmetrized, projected or repaired. S/T and ionic-V require nine and 36 local
mixed evaluations per radial expansion respectively, with complete visit caps
and the unchanged primitive authority. No AO-fourth-power derivative tensor is
stored; only the coordinate-only frozen matrix and its compensated staging are
resident.

CPU packaged stationary ERI first derivatives, factorized value/first/mixed
and physical-owner matrix sources explicitly select the shared positive
Full-range Boys evaluator. The shared evaluator supports orders through 14;
each primitive retains its existing order and storage limits. At
unit interval boundary the existing scaled positive-series/downward or
erf/upward evaluator computes the same Full moments; its bounded failure
fallback is the existing positive quadrature. FP64, normalization and the
next-moment derivative identity remain unchanged. The explicit opt-in preserves
finite argument/rho, zero Full omega, order and storage checks. Generic geometry
defaults, CUDA generation and LR/SR evaluation retain their existing schedules;
this CPU choice does not qualify a device path or expand range-separated DF
admission.

The symmetry-unique matrix primitive budget conservatively charges each quartet
for its physical-center right directions (at most `3*min(natoms-1,3)`), rather
than multiplying every quartet by the
four-distinct-atom worst case. Weighted triangular AO-pair prefixes count the
complete quartet work and the work excluding each atom. Their differences
count distinct physical owners without an AO-fourth-power planning walk:
planning costs `O(natoms * nao^2)` integer operations and `O(natoms)` integer
metadata. Single-owner quartets retain their required scalar value visit.
The shared matrix leaf reduces repeated factor evaluation but does not shrink
this admission envelope or raise the caller's finite cap. Actual matrix calls
are counted once, separately from the conservative admitted direction slots.
`eri_matrix_coordinates` and `eri_matrix_second_partials` count the actual
requested coordinate and ordered-partial fields, including failed calls, rather
than inferring cost from primitive calls alone. The ordered comparison retains
nine coordinates and 81 mixed partials even for repeated physical owners.
First/column plans, generated-map bounds and their bounded ordinary fallbacks
remain unchanged. The optional journal reports completed matrix calls,
coordinates and second partials only after frozen-integral preparation returns;
an unfinished phase has no such completed receipt. Separate
`cpu_hessian_integral_one_electron` and `cpu_hessian_integral_eri` trace regions
distinguish these phases without treating an enclosing timeout as evidence
about which phase dominates.

`HostNuclearIntegralCurvatureSchedule::ProjectedMatrix` is an optional alternative
for the frozen matrix's ERI contraction. It groups adjacent public records with
the same physical atom, normalized radial range and homogeneous angular degree
into a Cartesian monomial workspace. If the public polynomial map is `T`, its
spin densities are `D_poly = T^T D_public T`. The native packed polynomial
coefficients already contain component normalization; internal monomials have
unit coefficients, with no second normalization. The map is geometry independent
for the fixed lab-frame real harmonics. Public AOs, SCF/CPKS, S/T/V, W, first
integral sources and moving-D/W contractions remain in their original space.
The incumbent generated quartet weights and derivative primitives consume the
projected density; there is no separate hand-written exchange formula.

This schedule replaces repeated public spherical expansion visits with one
monomial contraction, but can increase the number of AO quartet weight
incidences. Its plan therefore prices primitive visits using the monomial layout
and generated map work using that layout's actual quartic extent. Numeric
storage additionally charges both projected-density/compensation panels, the
original spin density snapshot and the packed-basis snapshot. Congruence uses
compensated FP64 accumulation and runs only after the final complete-request
work checks. The internal `density_projection_terms` observation counts scalar
contributions separately; it does not redefine ERI primitive or generated map
work units. Snapshot equality protects repeated use against input mutation.

Projection requires finite public density entries with magnitude at most `1e50`
and packed polynomial coefficients with magnitude at most 16. These are optional
representation preconditions, not restrictions on the original public source.
The endpoint prefers projection only when its primitive plan is smaller, and
otherwise retains the ordinary matrix and streamed schedules. Unsupported
scaling, insufficient optional capacity or allocation failure may decline this
representation. Numerical failures still propagate. The final work check uses
the actually selected schedule; fitting the projected plan never admits a more
expensive fallback without checking it. A lower primitive budget alone is not
complete-endpoint timing, native raw-Hessian or stationary-mode qualification.

Freezing D/W does **not** discard electronic relaxation. The prepared first
integral panel supplies the compiler-owned moving-density contraction
`<D', F'_integral> - <W', S'>` for each physical left coordinate. Commuting the
density VJP and nuclear JVP of the same quadratic stationary objective supplies
the full J/K response term; the original overlap-constraint objective supplies
its sign and occupation convention. Supplied D'/W' still come from the coupled
nuclear solve. The compiler contractions and their full-request map work are
explicitly budgeted, and both external response inputs are snapshotted before
transactional publication. Production never differentiates displaced SCFs.

The complete-matrix endpoint may reserve this additional source and coordinate
storage from leftover integral capacity and the remaining aggregate numerical
budget. It retains both original streamed children with their original caps.
Allocation-only admission precedes a fresh whole-request work check and then
numerical preparation. If the extra reservation does not fit, the first-panel
plus streamed-curvature schedule remains available; if that first panel also
does not fit, both sources stream. Each selected fallback must independently
fit the caller's original work ceiling. Numerical or owner failures propagate,
never select another scientific path. Actual receipts count the selected
primitive work and reservations. This is not, by itself, large-system or
molecular transition-state qualification.

Generated integral-map registries use content-qualified private C++ namespaces
behind the native namespace alias. Distinct inventories loaded in separate
shared objects cannot interpose each other's GNU-unique registry metadata.
The namespace includes complete generated code/provenance identity; this does
not change the operator or introduce a second compatibility interface.

The source shares frame/model/AO validation with the first-integral consumer.
Owner checks occur after every complete AO pair, at a bounded quartet cadence
and before publication. Reentry and output aliases into any borrowed state,
including W, refuse; output may alias any external right input. All three
right-input contents are authenticated before publication. Failure
does not partially publish any scalar-pair or whole-column fields.

`apply_column(right, density_right, weighted_density_right, output)` takes
explicit empty response spans for frozen D/W, or finite symmetric spin/AO
matrices supplied by an enclosing response owner. The left partial still holds
D/W fixed. Compiler JVPs of the same canonical weights add their right motion
exactly once: each left gradient multiplies the weight tangent while its
geometric right action multiplies the primal weight. No extra primitive sweep
or per-left Fock/overlap source is required. `L_right` includes D/W motion and
need not equal the nuclear gradient dotted with `right`. This source neither
solves the responses nor certifies them or stationarity; arbitrary supplied
responses do not imply a symmetric stationary Hessian. The scalar mixed-pair
API retains its frozen semantics and original map plan.

The ionic point-charge operator is not the residual local/nonlocal ECP.
Residual ECP, XC, nuclear/classical sources and actual stationary response must be composed
separately before claiming a stationary Hessian or TS endpoint. The independent
primal-energy controls in `tests/python/test_host_nuclear_integrals.py` test
frozen or explicitly supplied D/W motion, not a relaxed molecular energy
Hessian or public admission.

## Native CPU residual-ECP nuclear direction

`integrals::checked_ecp_direction` (`integrals/ecp.hpp`) supplies the separate
local and nonlocal residual-ECP matrix directions for normalized Cartesian or
real-spherical s/p/d/f AOs. It does not include the ionic Coulomb tail, which
belongs to the nuclear-attraction source above. This synchronous internal-core
component participates in the development nuclear response consumer below. It
does not provide second derivatives, a stationary Hessian or CPU TS admission.

The analytic power-two local/s/p/d-projector domain reuses the ordinary
Gaussian ECP pair/primitive jets with directional retention and compensated
accumulation. Other legal residual powers 0--4 and projectors through f use
the existing checked real-space source. Both raw and directional grid consumers
share normalized AO metadata and one radial-layer traversal; AO jets and
projections are reused across every AO pair. The compiler's scalar Graph
projects the A/B center gradients against motion relative to the ECP center.
Neither backend evaluates displaced geometries, JIT-compiles or calls a
reference engine.

The general path retains the existing coarse/fine grids (160/32 and 224/44)
and absolute acceptance gates: `2e-9` for every local/nonlocal primal matrix
element and `2e-8` for every physical-coordinate derivative. A zero direction
or cancellation in its projection cannot bypass either gate. For a fixed AO
pair, only the two AO atoms collect derivatives from every ECP site; any other
physical coordinate receives contributions from just its own site. The source
therefore checks and discards those other completed coordinates per site,
retaining only the two AO-atom accumulators across sites. Coincident A/B/site
identities are scattered before checking. This preserves the full-coordinate
gate without retaining a `3*natoms*nao^2` tensor or repeating the source for
coordinate pages.

`ecp_direction_plan` selects the actual backend and reports mandatory owned
workspace and complete-source work bounds. The checked-grid workspace includes
42 `nao^2` doubles, radial AO/projection staging and grid/AO metadata, independent
of the number of basis-free all-electron atoms. Limits also bound Gaussian
kernel blocks or grid radial layers, primitive AO-point visits and contraction
terms. These are not RSS, total instruction, favorable-scaling or endpoint-time
claims. The caller's immutable normalized system and direction are borrowed;
a mandatory validator authenticates their owner/epoch during traversal and
before returning both matrices. Limit, quadrature and validation failures do
not publish either result.

`tests/python/test_native_ecp_direction.py` compares both parts against
independent normalized libcint primal differences through f and both AO
representations. It also exercises physical-coordinate parity, translation,
linearity, explicit limits, stale/final-owner failures, recovery, and actual
derivative-only quadrature refusal even for a zero projection. Its synthetic
residual operators are component controls, not representative Ni/I/Cs or
stationary/installed endpoint qualification. A native CTest executes both
backends without a Python numerical owner.

### Frozen analytic residual-ECP mixed source

`gaussian_ecp_weighted_mixed` (`integrals/ecp_gaussian.hpp`) contracts arbitrary
finite ordered AO weights with the existing normalized analytic ECP pair jets,
returning `[value, left, right, mixed]` separately for local and nonlocal
residuals. The weights include any caller-required occupation/spin sum; the
source adds no occupancy factor. Both nuclear legs use atom/xyz order. The
compiler projects independent relative A/C and B/C slots before native
compensated accumulation, including repeated physical atom owners. Rigid
translation of either leg gives zero mixed curvature exactly.

This source retains eight scalar outputs instead of a coordinate Hessian or
coordinate/AO derivative tensor. Its mandatory plan bounds owned workspace
and the complete order-two Gaussian kernel-block traversal, not RSS, total
instructions or endpoint wall time. The authenticated immutable system,
ordered weights and directions are borrowed. Both parts publish by return only
after final owner validation; malformed inputs, numerical failure, insufficient
limits and stale owners cannot expose a partial result.

The supported operator is the existing analytic specialization: power-two
Gaussian local/s/p/d residuals with normalized Cartesian or real-spherical AOs
through f. Unsupported or absent residuals are explicit refusals of this
specialized API. This does not supply the general checked-grid second source,
ionic Coulomb tails, density/energy-weighted-density response or stationary
assembly. It neither certifies SCF stationarity nor admits a complete Hessian
or transition-state endpoint. The corresponding tests use synthetic residuals
and independent libcint primal operators, not representative Ni/I/Cs endpoint
qualification. Basis-free atom storage controls are not complete electronic
25--40 atom performance gates.

### Residual-ECP curvature columns

`ecp_curvature_plan` and `checked_ecp_curvature` (`integrals/ecp.hpp`) provide
a source-driven complete physical-coordinate column for arbitrary finite
ordered AO weights. Each local/nonlocal residual part returns
`[value, right, gradient(3N), mixed(3N)]`. The caller supplies any occupation or
spin sum in the weights; there is no additional multiplier. An explicit empty
`weights_right` span means frozen weights. A supplied finite ordered AO matrix
adds their right response exactly once while the left nuclear partial holds
the ground weights fixed. In a KS assembly, weights are total D and
`weights_right` is total D_right; this source does not solve or certify either
the response or stationarity. The right scalar includes the weight tangent
and need not equal the fixed-weight gradient dotted with the nuclear direction.
The supported
public Cartesian/real-spherical AO domain is s through f. Structural dispatch
retains the existing analytic specialization and uses the checked quadrature
for other legal residual powers 0--4 and local/s/p/d/f projectors; an absent
operator is an explicit zero-work backend. A general operator is not silently
substituted into the specialized `gaussian_ecp_weighted_mixed` API.

Both providers consume canonical relative A/C and B/C second jets once per
right direction. Compiler Graph projections supply the right derivative and
relative gradient/HVP, including the weight JVP, with immediate compensated
physical scatter. The weight tangent consumes the same raw integral gradient;
it does not add a primitive or quadrature source pass. No source
pass repeats for a left coordinate. The grid computes second AO jets from the
same Gaussian expression as first derivatives and reuses them and their
angular projections across every AO pair in a radial layer. Symmetric packing
of finite mathematical-slot Hessians is not a symmetry repair of a molecular
output matrix.

The general-grid acceptance policy checks raw, **unweighted physical** primal,
gradient and Hessian components on the incumbent 160/32 and 224/44 grids.
Primal tolerance remains 2e-9; first and second derivative tolerances are
2e-8 per component. These are empirical two-grid gates, not an error estimator.
They apply even for zero weights, zero right motion and rigid translation.
Repeated physical A/B/C identities combine before checking. A/B coordinate
components accumulate across all ECP sites; components involving any other
physical site are checked at that site's completed radial reduction, then
discarded. Cross derivatives between two such other sites are identically
zero. The failure reports the derivative order whose gate refused.

The mandatory plan bounds owned source capacity and complete semantic work.
The grid retains two 28-field relative/physical-slot acceptance panels for
both grids and residual parts (224 AO-matrix-sized FP64 fields), shared
10-field AO/projection scratch, metadata and compensated physical output
columns. The checked consumer additionally owns ground/right weight snapshots
and the nuclear direction: `2*nao^2+3*natoms` FP64 elements. This bound is
independent of coordinate-squared AO storage. Point,
radial-layer and AO-primitive visits and point/projector pair contractions
describe complete source work; each second-order pair contraction carries
43 canonical fields, not a single scalar FLOP. Borrowed inputs, scalar stack,
allocator overhead, RSS and endpoint wall time remain separate. Resource
failure refuses rather than silently replaying an unbounded tile schedule.

Immutable-system contents/epochs belong to the authenticated enclosing owner.
External weights, optional right weights and the nuclear direction are
snapshotted only after admission and authenticated before publication. The
raw Gaussian provider borrows these stable panels; its provider-only plan does
not double-count the checked consumer's snapshots. Both residual parts publish
by return only after final validation; stale state, changed inputs, failed
quadrature and insufficient caps
cannot publish a partial column. There is no runtime compiler, displaced
production calculation or reference-engine fallback. These are fixed-left
residual sources with an optional supplied right response, not ionic Coulomb
tails, a D/W response solve, stationary Hessian
assembly, Ni/I/Cs electronic qualification or a TS endpoint.

## Native CPU composed nuclear response

The optional CPU host-response target provides `HostKsNuclearResponse`
(`dft/host_ks_nuclear_response.hpp`). It composes actual overlap and frozen-D
integral Fock motion, residual local/nonlocal ECP motion and moving-grid XC
motion, then feeds that physical nuclear source into the coupled metric/CPKS
consumer for full density and energy-weighted-density directions. The shared
`method.ks_cpks` TensorIR owns the per-spin frozen-Fock sum. The integral source
already includes ionic Coulomb attraction: residual ECP is added once, without
subtracting its Coulomb tail again. Nuclear repulsion and D3 do not enter this
electronic Fock response and remain separate geometry-only sources.

The consumer takes one verified KS state, normalized system, actual prepared
Fock owner and retained `HostGridGeometry`. It constructs its own fixed-grid
response reader from that geometry's AO basis, points and weights; accepting
an arbitrary detached reader here could pair a correct moving source with a
different induced-Fock action. The immutable copied spin density comes from
that same authenticated frame, is shared by both nuclear sources and is checked
against the current frame before publication. Grid prescription, spin, method,
SCF epoch, source model and all borrowed-state validators must agree.

Mandatory integral, XC, ECP, assembly-map and perturbation-map caps accompany
the composed owned-array ceiling. The bound includes live child panels,
geometry staging, ECP workspace, copied density, transactional S/F/D/W panels
and the requested GMRES workspace. It is not a bound on borrowed SCF/provider
storage, internal BLAS workspace, scalar stack, allocator overhead, RSS or
complete endpoint time. Resource refusal occurs without publishing outputs;
source failure or physical-response nonconvergence likewise publishes neither
member of an output pair. External direction/output aliases are supported,
but output/output and borrowed-state aliases are not; calls cannot reenter.

This internal development consumer is first-order electronic nuclear response,
not a public Hessian capability or transition-state endpoint. Its independent
controls in `tests/python/test_host_ks_nuclear_response.py` use actual cold
native RKS/UKS frames and displaced libcint/Libxc primal Focks plus independently
reconverged KS D/W. The compact synthetic ECP controls do not qualify real
Ni/I/Cs chemistry, complete source second derivatives, installed endpoints or
25--40-atom work/time/resource behavior.

## Native CPU stationary nuclear columns

`HostKsNuclearHessian` (`dft/host_ks_nuclear_hessian.hpp`) composes a stationary
RKS/UKS nuclear column in the optional CPU host-response target. It binds one
verified state, prepared full-range Fock, normalized system and moving-grid
column owner. The same base geometry supplies the frozen nuclear Fock source
and induced response action. A coupled metric/CPKS solve supplies total D′ and
the **full W′**, including occupied-block energy response. Integral Lagrangian,
moving XC and residual-ECP sources consume those responses; ECP contracts total
spin density and its right response without another occupation factor. Ionic
nuclear repulsion and optional uncut two-body D3(BJ), with real atomic numbers,
are added exactly once. Every derivative remains in the incumbent compiler
Graph/TensorIR owners; native code schedules and assembles the components.

The output layout is `[energy, gradient_dot_right, gradient(3N), Hright(3N)]`.
The integral source includes `-Tr(W S)` in its Lagrangian value. Assembly restores
that trace and checks the reconstructed electronic-plus-nuclear energy against
the verified physical components using explicit finite tolerances before adding
dispersion. It does not label the raw Lagrangian value or supplied-weight right
scalar a physical energy/derivative. No symmetry, translation or frequency-mode
repair is applied to the raw stationary column.

All child numeric reservations are composed **before** shape-dependent child
allocation. The response reservation includes its GMRES allowance; integral,
XC, ECP and optional D3 reservations use their enforced source caps. This avoids
allocating several full-cap children before discovering an aggregate overflow,
without duplicating private response workspace formulas in the assembler.
The shape-only assembly query includes snapshots of D/W/F/C/orbital energies,
S/core matrices, total densities, directions and simultaneous staged outputs.
The reserved aggregate is conservative; the resource record separately reports
the actual composed fixed bound and requested GMRES workspace. Borrowed SCF,
Fock, basis/grid/geometry owners, provider/BLAS workspaces, static tables,
metadata/allocator overhead, RSS and complete endpoint time remain separate.
The immutable external validator still authenticates borrowed geometry and
provider contents/epochs; the assembler additionally checks the actual complete
electronic frame and physical component/tolerance contents.

All output fields publish together only after response convergence, finite
source/energy checks, final frame validation and actual external right-content
authentication across the entire solve and source sequence. Reentry, borrowed
output aliases, stale frames, source failures and insufficient reservations
cannot publish a partial column. A direction/output alias is safe.

This is an internal stationary-column owner, not installed public CPU Hessian/TS
admission or real Ni/I/Cs and 25--40-atom electronic qualification. Its tests in
`test_host_ks_nuclear_hessian.py` use genuine native RKS/UKS stationary states and
independent multistep reconverged libcint/Libxc energies on the identical primal
grid, with compact empty/analytic/checked-grid residual-ECP controls and optional
independent D3 correction. The native standalone exercises the assembled owner
without Python or a runtime compiler.

The internal `borrow_dft_host_derivative_sources` handoff authenticates a live
native CPU KS token and borrows that exact prepared owner's system, Fock,
normalized AO basis, molecular grid and native dispersion metadata. The owning
context must remain serialized for the entire synchronous lifetime; geometry
rebuild, replay or explicit final-state revocation invalidates the old token.
No copied snapshot, caller identity label or separately reconstructed provider
substitutes for these owners. This handoff does not itself authorize derivatives
or include a correction added independently by a frontend to its total method.

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
smooth long-range expression. Worker floors and the `rho >= 1e-10` VV10 active
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
DF with exact LR exchange, all-electron or scalar semilocal ECP Hamiltonians,
and Cartesian or real spherical AOs through f in strict FP64. Combinations use the same stationary
endpoint. For DF, explicitly set `GENERATIVEQC_DF_RANGE_EXCHANGE=direct` before
preparation: default/automatic LR fitting qualifies energy and first derivatives,
not fitted-LR Hessians. A fitted LR source is rejected rather than substituted
by exact-LR second-order equations. The full-matrix owner uses scalar columns
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
cmake -S . -B build/cuda-release \
  -DGENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON
cmake --build build/cuda-release --target \
  generativeqc_hessian_aot_probe generativeqc_cuda_gmres_probe \
  generativeqc_cuda_cpks_probe generativeqc_cuda_wb97mv_grid_probe \
  generativeqc_ks_wb97mv_grid_probe \
  generativeqc_cuda_grid_geometry_probe \
  generativeqc_cuda_ecp_nuclear_probe \
  generativeqc_ks_resident_frame_tests
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_HESSIAN_AOT_PROBE="$PWD/build/cuda-release/generativeqc_hessian_aot_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_hessian_aot_cuda.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_GMRES_PROBE="$PWD/build/cuda-release/generativeqc_cuda_gmres_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_cuda_gmres.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_CPKS_PROBE="$PWD/build/cuda-release/generativeqc_cuda_cpks_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_ks_cpks.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_WB97MV_GRID_PROBE="$PWD/build/cuda-release/generativeqc_cuda_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_native_grid.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_native_grid.py \
  -k live_molecular
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_native_fock.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_GRID_GEOMETRY_PROBE="$PWD/build/cuda-release/generativeqc_cuda_grid_geometry_probe" \
  GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_native_grid_geometry.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_CUDA_WB97MV_GRID_PROBE="$PWD/build/cuda-release/generativeqc_cuda_wb97mv_grid_probe" \
  GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release/generativeqc_ks_wb97mv_grid_probe" \
  PYTHONPATH=python:. python -m pytest -q tests/python/test_wb97mv_nuclear_grid.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:05:00 \
  build/cuda-release/generativeqc_ks_resident_frame_tests
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:30:00 \
  env GENERATIVEQC_CUDA_ECP_NUCLEAR_PROBE="$PWD/build/cuda-release/generativeqc_cuda_ecp_nuclear_probe" \
  GENERATIVEQC_KS_WB97MV_GRID_PROBE="$PWD/build/cuda-release/generativeqc_ks_wb97mv_grid_probe" \
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

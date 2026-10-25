# Shared orbital response and bounded Krylov solves

## Native correlated RI-JK reference response

The full-DF molecular owner supplies an immutable `hf::DFReference` to the
same matrix-free RHF frame maps used by conventional correlation. It retains
SCF's original RI-JK plan and metric cutoff through the same-basis companions.
Its signed-density actions use dense linear J/K, never a positive occupied
factor approximation. The independent final frame-map replay evaluates the
same physical RI operator; switching it to four-center ERIs would be a model
error, not a stronger residual check. External reconverged PySCF energies
qualify the complete gradient independently of the CUDA J/K implementation.

The nuclear cross term uses symmetric polarization of the existing generated
DF energy derivative, including three-center and fixed-rank metric response.
It differentiates the actual retained metric branch and rejects unresolved
rank crossings. One-electron/Pulay and correlation source pullbacks remain
separate. Exact-source recycled operator images are not reused for RI-JK.
`PhysicalReference::density_fitted` and immutable frame identity prevent
accidentally accepting a DF reference with an exact response (or vice versa).
The generated response-map hash describes the frame equations, not the
auxiliary basis or Hamiltonian; public model identity retains the DF selector.

The explicit full-DF API and fitting-basis contract are documented in
[the CBS guide](../user/ccsdt_cbs.md). Its source/response leases are charged
through all correlation phases; no memory refusal authorizes CPU or exact-HF
fallback. The SCF and correlation AO-factor builds are still separate owners.

The installed runtime now owns the shared closed-shell response stack:
`generativeqc.response_problem` owns problem/layout compatibility,
`generativeqc.response_operator` owns RHF/CPKS matrix-free equations,
`generativeqc.response_xc` owns the fixed-density semilocal XC response kernel,
and `generativeqc.response_solver` owns bounded true-residual GMRES plus
blocked/recycled multi-RHS execution. `generativeqc.rks_response` binds a live
native LDA/PBE RKS state to those owners using an exact J-only `FockPlan`.
`tools/generativeqc_response` retains compatibility exports, UKS/spin-specific
adapters and response backends not yet migrated. Eligible CPU host Krylov solves use the shared native GMRES controller; eligible
CPU recycle-vector storage, projection and replacement use the native recycle
owner described below. Python retains method callbacks and multi-RHS scheduling,
and selected operator backends execute native J/K. This split
lets downstream property, Hessian, and correlated-gradient code reuse one
scientific owner without making installed runtime code depend on repository
`tools.*`.
This slice is partial: the RHF response layer and the direct-CPU UHF response
layer (including `export_uhf`), host-orchestrated spin CUDA exact/DF J/K, and
native CPU/CUDA LDA/PBE RKS/UKS CPKS handoffs are delivered. Exact-RHF resident
scalar, blocked and recycled multi-RHS execution share the same solver and are
qualified for the bounded tools domain. Performance evidence describes measured
endpoints, not an automatic execution selector.

This internal tooling is not a new public electronic-structure method. It
consumes the converged native HF/KS endpoints rather than implementing SCF.

The [generated implicit-response adapter](implicit_response.md) reuses the
same installed solver through an explicit callback. It generates transposed
operators and source weights from TensorIR rather than introducing a
method-specific adjoint solver.

## Problem snapshot

The native host/resident GMRES controllers force an exact candidate residual
when the projected Hessenberg residual predicts convergence, at restart,
breakdown or iteration exhaustion, and at the configured periodic interval.
The projected norm only requests an exact action; it never accepts a solution.
The native exact-RHF frame response defaults that interval to its restart size
to amortize repeated J/K traversal. Its separate scalar-CUDA final residual
and full frame stationarity audits remain mandatory. Setting
`gmres.true_residual_every=1` retains per-iteration candidate auditing for
comparisons and difficult numerical domains.

`RHFFrameResponseOptions::resident_jk_maximum_bytes` controls immutable full-range
canonical AO source values in the shared Direct provider. Its unset default
selects a conservative automatic policy: relaxed, unscreened frames with at
least 64 public AOs may use a lease capped at 8 GiB. Explicit zero retains
ordinary bounded recomputation; a positive ceiling requests reuse independently
of that crossover, including small validation frames. Admission also respects
the complete endpoint budget; unavailable canonical storage or
allocation refusal keeps the original exact route. If a published lease crowds
out a later required allocation, the entire frame attempt unwinds before one
retry with resident values disabled and the optional inverse released. Numerical
and non-allocation CUDA failures propagate; a second allocation failure does too.
`resident_jk_discarded_attempt` flags this case and `resident_jk_retry_seconds`
records its elapsed time separately. Other frame work and phase diagnostics
describe the successful attempt; complete endpoint time and the conservative
numeric-capacity bound cover both attempts, and the benchmark marks incomplete
work receipts unavailable.
Geometry, basis, device and stream lifetime belong to the Direct owner, not to a
dimension-only cache key.
The existing recurrence dispatcher prepares finite-audited values once; repeated
signed J/K actions reuse the same compiler-owned orbit scatter without evaluating
new integrals. Storage remains quartic in Cartesian source dimension, so this
is a bounded optional lease, not a new asymptotically memory-bounded algorithm.
The independent final scalar-CUDA residual and any positive fixed-mask action
explicitly recompute the unscreened/requested source rather than trusting the
lease. Preparation time, retained values, replay reads and uncached integral
evaluations are distinct diagnostics. The 64-AO gate is a conservative large-frame
selection, not a claim of a universally optimal crossover. Changing the default
policy requires renewed complete-endpoint qualification.
Experimental packed partial residency shares the
[RHF bucket-mask policy](scf_module_boundaries.md#optional-phase-local-exact-rhf-values).
Cached and uncached angular buckets form a disjoint complete source partition;
positive-screening and independent final scalar audits still recompute their
whole requested source. The additional packed host descriptors are charged
before admitting optional values.
For this eager exact frame owner, automatic admission now fits whole GPU
buckets to the available device allowance and uses bounded pageable FP64 RAM
for omitted buckets. GPU values retain priority. This applies to relaxed,
unscreened responses admitted by the existing size/capacity policy; it does
not alter the separate RHF-SCF graph or public prepared CPKS owners.
`GENERATIVEQC_RHF_RESPONSE_HOST_MAX_BYTES=0` disables the RAM tier. Unset uses
`min(32 GiB, one quarter of machine and enclosing cgroup headroom)`;
unavailable headroom information disables automatic RAM admission.
`GENERATIVEQC_RHF_RESPONSE_HOST_BUCKET_MASK` restricts the optional buckets.
`GENERATIVEQC_RHF_RESPONSE_HOST_TILE_BYTES` bounds each of two staging slots
(default 256 MiB, positive FP64-aligned, at most 512 MiB). Admission shrinks
tiles to preserve room for values, checks free VRAM after reserving mandatory
response storage, and charges values, two pinned tiles and two device tiles
to the complete numeric budget. No RAM tier is created without an admitted
GPU source; a completely resident GPU source allocates no host staging.

Host allocation/capacity refusal preserves the admitted GPU values. A later
required allocation failure discards both optional tiers before the existing
single exact retry. Non-allocation execution faults still propagate. The
independent final residual and nuclear derivative sources remain uncached.
`host_jk_bytes`, `host_jk_capacity_bytes`, `host_jk_actions`,
`host_jk_h2d_bytes` and `host_jk_setup_seconds` distinguish retained values,
complete storage, actual replay, PCIe uploads and construction. Temporary disk
is not enabled for this response owner: the separate long-range KS disk
qualification does not establish a response crossover.

The [resident-source decision](../../.agents/notes/implemented/performance/2026-10-05-exact-response-resident-source.md)
retains the rationale, measured evidence and conditions for revisiting it.

With `profile_jk`, synchronized wall and CUDA-event device times cover the same
J/K composition; resident timings are subsets of total J/K time. Profiling does
not change the ordinary dispatch. A complete canonical census is reported only
when all selected channels provide it. Cache preparation is charged to setup,
not hidden in or added twice to the J/K subset. The basis-only diagnostic
`benchmarks/rhf_resident_jk.py` records paired action times, source admission and
loaded binary hashes; its wall times include test transfers and do not replace
complete-force endpoint timing. In `benchmarks/df_ccsdt_force_endpoint.cpp`, omit
the final resident ceiling to measure the automatic policy, append `0` for
recomputation, or append a positive byte ceiling for explicit reuse.

Canonical exact-RHF frame actions use compensated FP64 J/K scatter, including
signed source densities, Krylov actions and the independently recomputed final
audit. The Direct provider retains the compiler-owned spin/permutation
contraction; two caller-owned canonical correction planes recover the rounding
residual of each atomic addition and are folded before public-AO projection and
finite audit. Complete admission charges the Cartesian source dimension, which
can exceed the spherical public dimension. This is an accuracy improvement,
not a bitwise-reproducibility guarantee or a tolerance relaxation. Unavailable
canonical storage retains the existing bounded value route. Resident-source
refusal still recomputes the same compensated canonical action, and independent
audits never borrow the resident source. The ordinary device J/K API, mixed
precision paths and ordinary energy/force SCF dispatch remain unchanged.

Compensated fixed-mask actions reuse the compiler-owned whole-shell Hermite and
Coulomb recurrence through f shells when prepared shell rows are available.
This avoids repeating primitive preparation for each Cartesian AO quartet;
it does not retain a quartic global tensor or change the Schwarz predicate.
Separate positive J and K destinations retain independent correction planes.
Resident buckets still take precedence; missing prepared storage retains
component-wise replay. A threshold tighter than the prepared rows is rejected
before any action writer, as in the original fixed-mask API.
`RHFFrameResponseOptions::materialized_jk=false` provides an internal ablation;
the independent final scalar residual always disables both materialization and
residency. Successful-attempt phase times and submitted materialized launch
counts are available in the optional DF progress journal without extra fences.

Restricted CUDA physical-reference exports also admit one canonical Fock
correction plane. Bounded quartet references use the existing generic exact
quartet consumer with the compensated sink throughout SCF, including final
physical-Fock builds; correction folds precede spherical projection and
diagonalization. Generated page ABIs currently lack that sink, so exported
references do not combine their ordinary atomics with corrected accumulation.
The required plane is charged to cold admission, retained device inventory and
reuse capacity and is released on the bucket's stream. Small s/p resident-ERI
references and their fixed-output matrix fallback retain their existing route.

Native `RHFFrameResponseOptions::df_preconditioning` optionally prepares a
same-frame DF numerical inverse from `D_ia=gap_ia-(ii|aa)-(ia|ia)` and
`U_Qia=2 B_Qia`. The compiler owns these expressions. A bounded host
Woodbury/Cholesky helper applies `(D+U^T U)^-1`; it supplies no physical response
values. Nonpositive/unsafe diagonals, failed Cholesky, short budgets or failed
accelerated solves retain the exact diagonal solver. Preparation consumes the
existing correlation source before its release, and all setup/workspace time
and capacities belong to the complete force endpoint. This option remains
opt-in until a representative complete-endpoint benefit is qualified.

An optional caller-owned `RHFFrameResponseRecycle` retains one solved direction
and its independent scalar-CUDA exact image, plus projection scratch. It binds
the exact geometry/basis, bitwise reference arrays/energy, occupation, device,
operator hash and provider policy. A new RHS may project onto that subspace;
the fresh exact residual still controls acceptance. Nearby geometries and
approximately equal canonical frames are rejected. There is no global cache
or promise of reuse across freshly recomputed RHF references. Retained cache
payload is charged during earlier RHF/CC phases as well. A resource-refused
primal attempt releases the optional cache before one cold retry; its elapsed
time is included and its unavailable work counters are explicitly flagged.

`ResponseProblem` binds all scientific state before an operator or subspace is
created:

- immutable converged reference arrays and reference identity;
- explicit occupied and virtual spaces, restricted spin block, and the
  occupied-major/virtual-minor nonredundant rotation layout;
- overlap metric, canonical gauge, RHS layout, operator identity, and
  Hamiltonian/functional/grid model hash.

Changing a reference, even with the same dimensions, changes
`reference.identity` and therefore `ResponseProblem.identity`. A retained
Krylov space checks this compatibility identity before every use. Reusing
vectors across a geometry/basis/model change requires an explicit
`KrylovRecycleSpace.transport(..., transform)` call and re-evaluation; equal
vector length is never treated as compatibility.

`RotationLayout` stores a response vector `x[i,a]` with
`x[i,a] = x_(occ_i, virt_a)`. Its density response is
`Delta P = 2 sym_ov(x)`, and its skew orbital generator is
`K[virt,occ] = -x` and `K[occ,virt] = x`. Occupied-occupied and
virtual-virtual rotations are not independent unknowns.

`ResponseProblem.diagnostics` gates on `minimum_ov_gap`, the smallest absolute
occupied-virtual orbital-energy denominator of the active rotations. A
same-occupancy (occupied-occupied or virtual-virtual) degeneracy is a redundant
direction and does not trip `require_stable`; a symmetry-degenerate occupied or
virtual subspace with a finite occupied-virtual gap remains solvable.

## RHF operator

`RHFResponseOperator` applies the closed-shell RHF Jacobian without assembling
an AO N^4 tensor:

```text
Delta P_mo = 2 sym_ov(x)
Delta P_ao = C Delta P_mo C^T
Delta F_ao = J[Delta P_ao] - 1/2 K[Delta P_ao]
A x       = (eps_virt - eps_occ) x + C^T Delta F_ao C
```

`NativeJKBackend` streams the existing native shell-tile ERI source and forms
only the O(N^2) Coulomb and exchange responses. `DenseAOResponseBackend` is a
tiny independent oracle used only by tests. The CUDA DF backend validates its
metric against the source auxiliary basis, geometry, and execution threshold;
a caller-supplied Hamiltonian label must match that metric. Supplying the
exporter's `metric` avoids rebuilding it during backend preparation.
`RHFResponseOperator.apply` is the JVP, `apply_transpose` is the VJP entry point, and `dot_identity` checks the
Euclidean transpose identity.

`explicit_rhf_response_matrix` independently assembles the tiny MO matrix

```text
A[(i,a),(j,b)] = (eps_a - eps_i) delta_ij delta_ab
                + 4(ai|bj) - (ab|ij) - (aj|ib)
```

and `finite_rotation_jvp` checks the same action against an explicit
`C exp(-t K)` rotation and an independent Fock build.

## Solver and multi-RHS strategies

### Native prepared-problem binding

The C++ `response::prepare_response` returns a `ResponsePlan` bound to the
logical `LinearResponseProblem` that prepared it. `solve_response` and
`solve_response_batch` reject a different binding before solver allocation or
operator execution, even when dimensions, symmetry and numeric options match.
Batch validation also precedes empty-batch completion and workspace refusal.
Problem and plan copies retain their binding; moving a problem transfers it.
Reconstructing or replacing a problem requires a newly prepared plan. A raw
`GmresPlan` cannot substitute for a bound `ResponsePlan` in these entry points;
numeric resource estimators may still consume the inherited GMRES plan.

The binding token retains neither method callbacks nor borrowed scientific
storage. Method adapters must still validate terminal-state generations,
provider/approximation identity and borrowed-state leases when invoking their
actions. Plan binding is not a convergence certificate or a substitute for
those method-owned checks. GMRES equations, residual gates, numeric-payload
budgets and backend selection are unchanged.

Sequential native batch admission includes prior retained solution vectors and,
when enabled, the reserved history/basis diagnostic capacities. The result
reports planned solution and diagnostic retention separately. An empty batch reserves
neither; a one-byte-short numeric budget is rejected before callback work.

Direct native JVP/VJP calls enforce the same FP64 input/output dimensions as
solver actions, before invoking method callbacks. Explicit transpose actions
and certified symmetric fallback actions use this same shape gate. The RHF
frame adapter binds screened and exact actions to separate plans, while CC Z
certifies symmetry only after its independent asymmetry/curvature checks.
Callback right preconditioners pass through the shared solve request to existing
GMRES; they cannot be combined with a diagonal preconditioner. Their descriptor
storage is charged by batch admission and method-owned scratch is separate.

### Installed solver policies

Positive-dimensional CPU host-vector calls to `solve` use the shared C++
`response::solve_gmres` controller when the installed library exposes the public
response protocol. This also covers the individual solves in sequential and
recycled `solve_many`. The C++ owner performs Arnoldi orthogonalization,
right-preconditioned Givens solves, true-residual acceptance and bounded storage
admission; method-specific physical actions and live-reference checks retain
their existing owners. See the [native API](../reference/native_api.md#response-controller)
for standalone C/C++ use without Python or an execution context.

The admitted Python path accepts no preconditioner or `DiagonalPreconditioner`.
Explicit vector engines, device providers, solution consumers, general
preconditioners, zero-dimensional problems and blocked multi-RHS solves retain
their existing controllers. An absent library/additive protocol also preserves
compatibility. A selected library's load error or native solve failure is not
permission to retry Python arithmetic. Eligible CPU `sequential` multi-RHS
scheduling is also native: the controller streams an admitted prefix through
the existing bound response owner, then publishes workspace refusals for the
remaining tail without applying its operators. It validates live leases even
for zero RHS and refused requests. Eligible CPU recycle-space projection and
replacement use the native owner described below. Recycled/block scheduling
and the value-only RHS-rank SVD remain Python-owned; this is not a complete
native molecular Hessian/HVP endpoint.

The sequence bridge borrows validated FP64 RHS storage read-only, supporting
positive element-aligned strides and repeated columns. A noncontiguous column
uses one bounded, tracked staging vector rather than a second whole-matrix
copy; admission precedes its allocation. Foreign dtype, negative/unaligned
strides and existing non-host domains retain the compatibility schedule.
Native publication is synchronous: only detached solution/basis copies remain
charged between requests; history copies and publisher object bookkeeping
have separate budgets. Current solver/publication-copy overlap is covered by
the GMRES workspace bound. The reported peak also includes required bytes for
refused requests; it is not a full physical endpoint allocation measurement.

Native and compatibility controllers solve the same scientific linear problem
and require the true, unpreconditioned residual to meet the requested tolerance.
Givens versus least-squares arithmetic and checkpoint scheduling can produce
different iteration/action histories; diagnostics report actual work, not
fabricated bit-identical counts. Native timings use a native monotonic clock.

`benchmarks/cpu_cpks_endpoint.py` records complete CPU LDA/PBE RKS/UKS endpoint
samples, including construction, SCF, response preparation, multi-RHS solves,
true-residual checks and cleanup. Run with `PYTHONPATH=python:.`, an explicit
`GENERATIVEQC_LIBRARY`, `--source-sha` and an ignored `--output` destination.
Parent comparisons require the parent's Python tree **and** native library.
Keep numerical residual/solution gates and actual semantic work alongside
timings; per-case first-use samples share a process/cache and are not independent
compiler-cold measurements. See [performance qualification](../maintainer/performance_engineering.md).

`solve` implements restarted GMRES with a true residual at every configured
checkpoint. It reports the actual residual, iteration count, operator actions,
orthogonalization/operator/recycling timings, workspace bytes, and a non-success reason.
It does not silently regularize a singular denominator or claim success after
a workspace or stagnation failure.

The native host GMRES workspace bounds simultaneously live numeric payload:
the Arnoldi and preconditioned bases, iteration vectors, Hessenberg/Givens
arrays and returned solution. Optional history and published-basis capacity is
charged only when requested. The bound includes one extra vector of publication
headroom; `measured_workspace_peak_bytes` separately reports actual tracked
allocations. Input spans, callback-owned storage and allocator metadata are
outside this controller's bound and belong to the caller's complete budget.
Resource planners should query this bound instead of reserving unused Krylov
bases; a budget one byte below the queried requirement refuses before allocation
or any operator action.

The operator, preconditioner and retained-subspace roles are separate.
`DiagonalPreconditioner` is opt-in and rejects zero/near-zero diagonal entries;
it is never selected implicitly.
Its inputs and host operator/preconditioner callback outputs must have the
declared real numeric vector shape; complex/string outputs are rejected before
FP64 conversion rather than silently narrowed or broadcast during Krylov work.

`solve_many` provides:

- `sequential`: one bounded solve per RHS;
- `blocked`: a shared orthonormal block basis with projected least squares;
- `recycled`: reference-bound retained vectors used as initial guesses and
  updated after each solve.

All three strategies accept an empty real RHS of shape ``(dimension, 0)`` and
return that same solution shape, no per-RHS results, zero operator actions and
zero solver workspace. They still validate the live generation/provider lease,
recycle identity/owner, options and consumer/preconditioner contracts. No vector
engine workspace, SVD, transfer or scientific action runs for an empty request.
Perturbation labels describe the exact column set: partial requests require
their own matching labels; a labeled nonempty problem cannot silently be used
for an empty/subset request. Ragged input is rejected before operator execution.
The shared stationary nuclear adapter follows these same empty-request rules.
Its real-FP64 metric/RHS algebra rejects complex inputs before conversion,
including complex arrays with zero imaginary parts; it never discards response
components to fit the real scientific domain.

Rank-deficient RHS blocks are diagnosed. The workspace accounting includes the
retained Arnoldi/block basis and solver vectors. A small `max_workspace_bytes`
returns `workspace_limit` before applying the operator.

`SolveResult.relative_residual` is `||r|| / ||b||`; a zero RHS is defined as
`0` for an exactly zero residual and `inf` otherwise, never as an absolute
residual just because `||b|| < 1`. In the recycled strategy the budget sums the
shared immutable RHS copy, previously retained result arrays, independent
recycle-space vectors, projection and bounded replacement temporaries, and the
next solve workspace. The single-RHS API applies the same recycle reservation
before projection or operator application. A successful peak is a conservative
bound within the requested budget; a workspace rejection reports the required
bound. Operator/preconditioner storage and Python interpreter bookkeeping are
outside this solver numeric-buffer budget and require separate accounting.

### Native CPU recycle vectors

Eligible positive-dimensional FP64 `KrylovRecycleSpace` instances own their
retained directions in C++. Normal CPU solve projection/replacement does not
publish those directions back into Python. Native projection uses the retained
orthonormal directions; replacement preserves old-first, two-pass MGS and the
existing `1e-12 * max(1, original_norm)` acceptance cutoff. It uses the existing
scaled native norm and shared CPU linear algebra, not a different response
operator or an operator-image projection.

The additive [native recycle API](../reference/native_api.md#cpu-response-recycle-vectors)
provides shape-only capacities, an opaque owner, generation checks, projection,
atomic replacement and explicit diagnostic copies. Both signed and repeated
element strides are borrowed without making an entire Arnoldi panel copy.
Changing retained limits is an atomic native update; old vectors remain charged
until replacement succeeds. Additional native capacity bounds reserve the
replacement and bounded scratch, separately from existing retained bytes.

Whole-solve admission precedes native handle acquisition and numerical work.
Missing additive protocols, explicit host seeds/transport, foreign data and
resident/device domains preserve explicit compatibility. If an existing native
space enters a compatibility update, its old directions stream through one
reusable column buffer; failure preserves the original native state. A smaller
destination capacity never forces a copy of the complete old space. Explicit
identity/transport diagnostics detach immutable copies. Selected native/load
errors never retry Python arithmetic.

Native telemetry measures owned numeric arrays, including failed attempts;
caller copies, Python/allocator bookkeeping and physical provider storage have
separate lifetimes and budgets. This is not complete physical endpoint resource
authority. Eligible CPU host GMRES uses the shared native controller; outer
RHS/recycled/block scheduling, value-only rank SVD, method-specific equations and
full stationary molecular HVP assembly retain their documented Python/compatibility
owners. CUDA/resident engines are not
admitted by this CPU cutover.

### Actual-path evidence and residency boundaries

The [production-path inventory](../../manifests/maintenance/electronic_structure_production_paths.json)
records response/force/HVP owners, bindings and evidence stages separately.
Its coverage remains a pilot, not a complete capability table.
Default CC Lambda, RHF frame Z and generic/multi-RHS CPKS compatibility adapters
still have host-vector Krylov/orbital seams. The installed native RKS object's
single-RHS `solve`/`solve_resident` endpoint now executes orbital transforms and
the shared GMRES controller natively with resident vectors, as specified in
[response backends](response_backends.md#native-cuda-cpks). This does not establish
resident reconstruction or a complete molecular derivative endpoint.
Generated CUDA actions or resident J/K storage alone do
not establish a resident solver endpoint. The resident CPHF tooling is a
qualification binding, not a public molecular-derivative cutover. Complete CUDA
semilocal RKS HVP remains unsupported until its missing sources and endpoint
gates are delivered; mixed integral stages and local ``fxc`` are insufficient.
These exceptions retain their owners and explicit retirement conditions in the
inventory. No new default algorithm or scientific capability is inferred from
shared orchestration.

### Native CUDA Lambda residency

`LambdaOptions::cuda_resident_krylov` enables an internal, non-default binding
of the existing conventional or DF RCCSD Lambda action to `solve_response_resident`.
The corresponding native RCCSD/CCSD(T) force overloads accept the trailing
`resident_lambda_krylov` execution option. Installed public calculators retain
their existing schedule; this option does not add a scientific capability.
The native DF complete force owner separately accepts a trailing
`lambda_resident_krylov` option, also default-off.

The same nonsymmetric logical response plan serves host and resident schedules.
The method adapter owns weighted pair projection, denominators and both physical
residual gates. Vectors, pair projection and optional right preconditioning stay
on the action owner's existing stream; only Arnoldi/Givens scalars, error flags
and the final solution return to the host. Unsafe denominators still select the
unpreconditioned physical solve, not a different equation.

Admission includes the generated action arena, resident vector/table arena,
borrowed reference allowance and the existing tensor context's bounded provider
allowance. That allowance includes its explicit BLAS workspace; do not add it
again to the complete `resident_device_admission_bytes` diagnostic. Numeric owned
bytes, observed provider preparation growth, versions and Krylov-phase transfer
and fence counters are reported separately. Optional budget/provider-allocation
refusal retains host-vector execution; stream, kernel and numerical failures do
not trigger a scientific retry. All queued work drains before owner teardown.

DF borrows its existing input/error view and stream. Both host and resident
schedules call the same physical transpose dispatcher: expanded Q, scalar
staged, or bounded matrix-tail actions retain their original equations and
independent expanded audit. The internal device entry point returns borrowed
outputs, valid until the next action; consumers enqueue on that same stream and
check the sticky error before publication. Its opaque stream view does not add
a CUDA header dependency to CPU/source-only clients.

DF admission also charges the additional resident/provider allowance against
its complete host-plus-device numeric budget and original available/device
ceiling. The original action provider and vector provider are separate live
owners; neither allowance can be omitted. Refusal must not disable the original
right preconditioner or select a different physical/audit equation.

`tests/native/test_cc_resident_endpoint.cpp` exercises same-backend complete
RCCSD E+F pairs against pinned independent molecular references, process-cold,
process-warm and changed geometry cases, and two valid endpoint budgets. These
fresh-state calls do not claim prepared-plan or borrowed-lease warm reuse.
`tests/python/test_df_lambda_resident_cuda.py` covers physical factor response,
independent determinant finite differences, tail schedules and refusal/recovery.
`tests/python/test_df_resident_force_cuda.py` pairs complete DF RCCSD/CCSD(T)
E+F endpoints with independent molecular energy/force-direction references.
Both residency bindings remain opt-in pending broader performance/resource and
prepared-lifetime acceptance. These fixtures do not establish prepared reuse.

## CPKS boundary

`generativeqc.response_xc.FixedDensityXCDerivativeKernel` evaluates the
audited semilocal feature Hessian and contracts it with the exact
first-order density-feature response.
`generativeqc.response_operator.CPKSResponseOperator` adds that kernel to the
J/K response action. The tools modules re-export the same objects; they do not
retain a second response equation. The kernel/reference basis, grid, functional,
and density identities must match exactly.

The [common contraction generator](xc_contractions.md) owns the scalar-Hessian
chain rule and AO assembly. `apply_spin()` preserves functional-spin channels
and cross-spin terms; `apply()` retains the restricted mean. An optional
matching `PreparedXCContractions` response owner selects bounded native CPU
execution through `prepared=...`, with its shared numeric resource plan.

`NativeRKSResponse.from_native(batch, basis, grid=None, index=0)` connects the
actual successful native CPU or CUDA LDA/PBE RKS state to this same operator and solver.
The optional explicit grid must exactly match the native points, weights and
owners. The optional `functional` must match the canonical SCF composition.
The adapter exports the native canonical orbitals, physical Fock, overlap,
occupations, energy and residual without rerunning SCF or recanonicalizing.
Its reference identity binds the native owner, model/provider, SCF domain and
solve/density/orbital generations. Direct unscreened CPU J/K uses the matching
native integral source. The batch and `NativeAO` must remain open; the adapter
owns its integral source and revocable snapshot lease.

```python
from tools.generativeqc_response import NativeRKSResponse, solve_many

# batch.execute(strict=True) has already converged; basis describes its exact AO source.
with NativeRKSResponse.from_native(batch, basis) as response:
    result = solve_many(response, rhs, strategy="recycled", raise_on_failure=True)
```

Native SCF quadrature includes low-density tails outside the legacy
`interior-v1` kernel domain. The native adapter differentiates the existing
scaled SCF point expression analytically in a restricted total-density
direction and feeds Cartesian potential coefficients into the same compact AO
assembly. It does not clip densities, omit tail points, finite-difference the
production potential, or substitute the interior-domain model. Exact vacuum is
accepted only with zero density/gradient direction. Undefined or unrepresentable
point derivatives fail the action. See [the SCF point domain](xc_scf_domain.md)
and [the binding decision](../../.agents/notes/implemented/numerics/2026-09-20-native-rks-cpks.md).

Every action and solve validates the live lease, including zero RHS and blocked
zero-RHS solves that otherwise skip all actions. Replay (even of identical
geometry), failed replay, batch closure and response closure revoke old solves
and recycle spaces. Changed functional, grid, basis, provider or state are
rejected before publication.

These handoffs qualify all-electron CPU/CUDA LDA/PBE RKS and UKS. DF CPKS,
ECP, exact/range-separated exchange, and meta-GGA response remain unsupported.
AO/MO transforms and Krylov orchestration are host-side; CPU XC uses host tiles
and CUDA XC executes AO evaluation through response assembly on device. Existing
solver workspace accounting is not a complete endpoint memory/performance
claim; the native kernel does not yet qualify implicit-response resource binding.
`tests/python/test_response_native_rks.py` checks independent libcint/Libxc
actions, finite orbital rotations, reconverged one-electron perturbations,
transpose/true residuals, multi-RHS/recycling, tails and lifecycle negatives.
The native `generativeqc_rks_response_tests` target also exercises the actual private
point-response ABI against 30 independent high-precision directions, its batch
layout and invalid-input boundaries, and energy snapshot leases from real
LDA/PBE H2 solves. See [point acceptance](xc_scf_domain.md#executable-evidence)
for the fixture generator and cancellation-aware numerical gate.

`NativeUKSResponse.from_native` uses the same arguments and lifetime contract
for the actual native CPU/CUDA LDA/PBE UKS state. It preserves both canonical spin
frames and occupations. The existing spin reference/layout contract carries
an explicit `algorithm="UKS"` tag and functional/grid identities; the UHF
operator rejects this reference. `UKSResponseOperator` changes only the shared
spin operator's AO Fock-response seam: both outputs contain total Coulomb plus
their own XC response, including cross-spin correlation. The common XC tile
assembler and GMRES/multi-RHS/recycling code are reused without spin averaging.

An empty spin retains its zero-dimensional orbital-rotation block. Its point
density, gradient and response direction must be exactly zero; nonzero normal
directions are rejected because the exchange Hessian is singular there. The
other spin still has a nonzero response, including the correlation potential
in both output channels. This is a tangent-direction qualification, not a
finite full Hessian at the empty-spin boundary.

`tests/python/test_response_native_uks.py` uses real LiH+ and H2+ LDA/PBE states,
independent libcint/Libxc spin actions, finite orbital rotations, reconverged
spin densities, true residuals, transpose identities, multi-RHS/recycling and
lease/domain negatives. `generativeqc_uks_response_tests` checks 48 independent
high-precision point directions, spin permutations and the private batch ABI.
See [the spin binding decision](../../.agents/notes/implemented/numerics/2026-09-20-native-uks-cpks.md).

### Native CUDA CPKS

The native device J/XC response and its exact budget/lease checks are
documented under [CUDA CPKS](response_backends.md#native-cuda-cpks).
The installed GMRES and spin-resolved scientific equations remain shared.

## Downstream consumers

The tools endpoint `BoundCCSDGradient` builds the CC-specific orbital RHS
and weights, binds the converged RHF reference to `RHFResponseOperator`, and
calls `checked_transpose_solve` through `ResponseGMRES`. The callback delegates
to this package's GMRES. A separately generated physical orbital matrix checks
the final Z-vector residual and the complete gradient's stationarity before
publication. `test_cc_complete_gradient.py` qualifies the shared action against
an independent MO matrix, native complete gradients against pinned references,
and explicit Z-vector nonconvergence. The CC-specific weight/source ownership
remains in the correlated-gradient consumer; no SCF/DIIS iteration tape is part
of this contract.

The `solve_rhf_nuclear_perturbations` consumer prepares ordered nuclear and
metric RHS columns, calls `solve_many` with final basis publication disabled,
and reconstructs the occupied-orbital/density responses. `rhf_hvp_many` and
`rhf_hessian` use that same boundary for bounded blocks. Their opt-in exact-RHF
resident execution and independent complete-HVP gates are described in
[hessian.md](hessian.md).

## Backend boundary

Native direct and density-fitted J/K adapters, CUDA-resident RHF/UHF
execution, host boundaries and their tests are in
[response backends](response_backends.md#backend-boundary).

## UHF CPHF boundary

The alpha/beta layout, spin-summed J, spin-specific K, direct/DF
CUDA adapters and numerical qualification are documented in
[the unrestricted response contract](response_backends.md#uhf-cphf-boundary).

## Resident response failure and validation scope

See the [resident lifecycle and failure contract](response_backends.md#resident-response-failure-and-validation-scope)
for teardown, independent device tests and failure publication.

### Combining screened response with optional accelerators

See the [screened-response accelerator contract](response_backends.md#combining-screened-response-with-optional-accelerators)
for provisional solve, exact residual and bounded retry rules.

```{toctree}
:hidden:
:maxdepth: 1

response_backends
```

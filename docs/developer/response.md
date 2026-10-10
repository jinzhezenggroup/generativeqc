# Shared orbital response and bounded Krylov solves

The installed runtime now owns the shared closed-shell response stack:
`generativeqc.response_problem` owns problem/layout compatibility,
`generativeqc.response_operator` owns RHF/CPKS matrix-free equations,
`generativeqc.response_xc` owns the fixed-density semilocal XC response kernel,
and `generativeqc.response_solver` owns bounded true-residual GMRES plus
blocked/recycled multi-RHS execution. `generativeqc.rks_response` binds a live
native LDA/PBE RKS state to those owners using an exact J-only `FockPlan`.
`tools/generativeqc_response` retains compatibility exports, UKS/spin-specific
adapters and response backends not yet migrated. The Krylov controller remains
Python/host-controlled; selected operator backends execute native J/K. This split
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

`solve` implements restarted GMRES with a true residual at every configured
checkpoint. It reports the actual residual, iteration count, operator actions,
orthogonalization/operator/recycling timings, workspace bytes, and a non-success reason.
It does not silently regularize a singular denominator or claim success after
a workspace or stagnation failure.

The operator, preconditioner and retained-subspace roles are separate.
`DiagonalPreconditioner` is opt-in and rejects zero/near-zero diagonal entries;
it is never selected implicitly.

`solve_many` provides:

- `sequential`: one bounded solve per RHS;
- `blocked`: a shared orthonormal block basis with projected least squares;
- `recycled`: reference-bound retained vectors used as initial guesses and
  updated after each solve.

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

## CPKS boundary

`generativeqc.response_xc.FixedDensityXCDerivativeKernel` evaluates the
audited semilocal feature Hessian from #161 and contracts it with the exact
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

The #153 tools endpoint `BoundCCSDGradient` builds the CC-specific orbital RHS
and weights, binds the converged RHF reference to `RHFResponseOperator`, and
calls `checked_transpose_solve` through `ResponseGMRES`. The callback delegates
to this package's GMRES. A separately generated physical orbital matrix checks
the final Z-vector residual and the complete gradient's stationarity before
publication. `test_cc_complete_gradient.py` qualifies the shared action against
an independent MO matrix, native complete gradients against pinned references,
and explicit Z-vector nonconvergence. The CC-specific weight/source ownership
remains in the correlated-gradient consumer; no SCF/DIIS iteration tape is part
of this contract.

The #180 `solve_rhf_nuclear_perturbations` consumer prepares ordered nuclear and
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

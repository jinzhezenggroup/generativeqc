# Native and CUDA response backends

The [shared orbital response contract](response.md) owns the
scientific operator, reference identity, exact residual and
GMRES/multi-RHS policy. This page owns backend-specific plans:
native CPU versus CUDA J/K/XC actions, spin-resolved response,
device-resident vector lifetimes, physical transport and resource
admission. Backend selection is not proof of an all-device solver.

## Native CUDA CPKS

The same `from_native` entry points select CUDA J/XC actions when the borrowed
batch is a CUDA KS owner. A live native proof must establish that both ECP terms
and atom core counts are absent. Legacy libraries without this proof remain
unsupported for CUDA CPKS. Method, exact packed basis/quadrature, canonical spin
frames, physical residual and revocable native token are bound as on CPU.

The RKS adapter requests the exact full-range J/K terms authenticated by the
native model; a semilocal graph requests J only, without an unused K contraction.
Admission uses generated point-response and component-scaling capabilities,
not functional names. Tau, range-separated exchange, nonlocal correlation,
ECP and density-fitted response remain unsupported by this RKS endpoint.
The legacy unrestricted adapter retains its separate admission contract.
XC extends the existing `CudaXcPlan` with
a directional feature panel, reusing its AO evaluator, density contractions,
point policy and potential assembler. The CPU and GPU point differentials use
the same `xc_point_response.hpp` formulas. Preparation copies the native state's
actual packed basis and immutable quadrature sources, and borrows the current
SCF owner's resident D/grid under its revocable token. It does not regenerate
the quadrature, re-export D/F/C/W, or rerun SCF. Compatibility actions upload a signed AO density direction and
download the completed AO response; AO values and point features stay on device.
Invalid directions/domains reject publication and the next action resets the arena.

`NativeRKSResponse.solve` and `solve_resident` execute single-RHS GMRES in the
shared native `solve_gmres_resident` controller. They reuse the existing resident
RHF orbital/vector engine, with the authenticated Fock K coefficient and a
method-owned XC density action. C/orbital energies, AO transforms, response
density/Fock scratch and all dimension-sized Krylov vectors stay on device.
The native controller owns Arnoldi/Givens/true-residual acceptance; Python only
validates and transports inputs/results. No host action, NumPy orbital transform,
Python Krylov loop, SCF rerun or fallback occurs during this solve.

The initial orbitals/energies and RHS/optional guess are uploaded during setup.
Iteration returns only scalar reductions and numerical-status/XC scalar records;
the final solution is downloaded once. The XC direction is consumed directly
from the orbital engine's device density, not its host-upload compatibility panel.
Snapshot/provider identity is checked even for zero RHS and workspace refusals,
and state freshness is rechecked before result publication. The caller retains
the batch/basis/snapshot lifetime; concurrent destruction is not supported.

`device_budget_bytes` (default 128 MiB) bounds retained response XC, J/K and the
native orbital/Krylov arena together. No reservation grows to make a solve fit.
`GMRESOptions.max_workspace_bytes` bounds the native host scalar/result payload;
transport-owned input/output copies and provider setup are accounted separately
from this solver payload. Impossible device capacity raises explicitly; host
workspace refusal reports `workspace_limit` without uploads/operator actions.
`response.diagnostics` identifies the default scalar solver and last execution,
retains `last_resident_solve` with the combined device peak and native physical
transfers, and exposes native XC setup/action
transfer and synchronization counters. For each host-upload XC compatibility
action with `s` spin channels and `n` AOs, H2D is `8*s*n*n` bytes and D2H is
`8*s*n*n + 28` bytes (matrix plus three scalars and a status), with two explicit
fences. Native resident XC actions transfer only the scalar/status record, not
the direction or potential matrices. Initial
snapshot export is reported separately; XC preparation borrows rather than
performing a second final-state export. Coulomb compatibility statistics report host payload, not measured PCIe
traffic. Borrowed SCF/eigensolver storage, preparation temporaries, host arrays
and solver workspace, CUDA context and library-private memory are outside the
retained response device budget. The generic `solve(response, ...)` compatibility
adapter and blocked/recycled `solve_many` still own host orbital/Krylov vectors;
their existence does not change the native scalar endpoint's transfer contract.
This increment does not qualify resident CPKS density/W reconstruction, nuclear
RHS preparation, molecular HVP/Hessian, or complete endpoint performance.
Native results do not export an Arnoldi basis/history or infer separately timed
action/orthogonalization components; their zero component times are unmeasured,
not a speed claim. Time the complete call separately.

With `GENERATIVEQC_RESPONSE_CUDA_TEST=1` under an explicit Slurm GPU allocation,
`tests/python/test_response_native_cuda.py` reuses the independent CPU-tier
libcint/Libxc, finite-rotation and reconverged-perturbation assertions on real
CUDA LDA/PBE water RKS and LiH+ UKS states. Tests forbid CPU AO/XC/J fallbacks and
SCF reruns during actions. They also cover empty-spin tangent directions,
resource rejection, preparation/export counters, legacy-proof/method/ECP gates
and lifetime revocation. `generativeqc_xc_response_cuda_tests` runs all 30 restricted
and 48 unrestricted independent high-precision point directions on device with
the unchanged CPU-tier numerical gates. `tests/python/test_resident_cpks_cuda.py`
qualifies complete native solves against independent libcint/Libxc Jacobians,
renamed graphs, zero/initial-guess RHS, resource refusals/replay and revoked states;
it also verifies that a converged nonlocal meta-GGA SCF does not imply response
support. `tests/python/test_scaled_xc_point_response.py` independently checks
fixed X/C weights, including missing components, against Libxc potential
differences for both spin conventions. See
[the CUDA CPKS decision](../../.agents/notes/implemented/numerics/2026-09-20-native-cuda-cpks.md).
The [native resident scalar decision](../../.agents/notes/implemented/architecture/2026-10-12-native-resident-cpks.md)
records the shared-engine ownership and retained compatibility boundaries.

## Backend boundary

`NativeJKBackend` streams the CPU native shell-tile source. `CudaDFJKBackend`
owns a prepared streamed CUDA density-fitting J/K plan and applies the same
matrix-free RHF action on the RTX 5090 under Slurm. It fails closed when the
source has no auxiliary basis or the binary lacks CUDA support. The real-device
test compares this action with the independent explicit MO matrix from
`DFProvider`.

`CudaDirectJKBackend` adapts the existing method-neutral `FockPlan` to the
shared RHF response operator. It requests exact full-Coulomb J/K, FP64 and
**zero screening**, preserving the `conventional-unscreened` Hamiltonian. It
copies the source's actual shell records rather than resolving a basis name.
The prepared CUDA provider is reused across signed, symmetric density actions;
raw J and K are returned without core-Hamiltonian contamination or additional
RHF factors. There is no CPU integral fallback or substitution of a DF operator.

The default `CudaDirectJKBackend` path remains host-orchestrated around CUDA
J/K contractions. B2 adds an opt-in `CudaResidentRHFResponse` owner for exact
conventional RHF. It borrows the same prepared direct-J/K source and CUDA
stream, retains C/orbital energies, Krylov vectors, AO response density/Fock
scratch and AO/MO transform scratch on device, and uses the existing #179 GMRES
control flow rather than a second solver. Vector copy/AXPY/dot/norm and
orthogonalization execute through cuBLAS; the direct J/K action is the existing
device-to-device provider seam.

The host still owns nuclear/metric RHS preparation, final occupied/density/
energy-weighted-density reconstruction, the small Hessenberg least-squares
problem, and convergence decisions. Thus diagnostics call this
`cuda-resident-host-controlled`, not an all-device CPHF. During a resident
operator action no density/J/K matrix crosses the PCIe boundary: only the
4-byte native numerical-status flag returns; dot/norm reductions return
scalars, and final solution publication is explicit. Scalar and block Hessian
consumers suppress final Arnoldi-basis publication. Host preconditioners and
non-RHF resident operators are not qualified and fail closed.

`solve_many(..., collect_basis=False)` suppresses final basis publication for
all three strategies; each solution is still returned on the host. Block Arnoldi
uses the same vector-engine operations as scalar GMRES: twice-reorthogonalized
thin QR followed by an SVD of its small factor determines the new range. Neither
a long host block nor Gram normal equations are constructed. The initial
sequential/recycled RHS-rank diagnostic still runs a value-only host SVD on
already-host API inputs; small projected block factors also stay on the host.

An automatic recycled space follows the selected vector engine and releases its
retained leases on every exit. For reuse across calls, pass
`KrylovRecycleSpace(problem, vector_engine=resident)` explicitly and close it
before its borrowed resident owner. The same reference/operator key and exact
owner must match, including on zero RHS. Replacement is atomic: an unsuccessful
update preserves the previous space and generation. Explicit diagnostic
`initial_guess`, `update` and `transport` calls may transfer vectors; the solver's
bound resident projection/update path does not.

`resident_vector_slots(dimension, options, rhs_count=..., strategy=...)` plans
conservative lease capacity. Counts above 4096 must be rejected by consumers;
they must not be clamped. A smaller supplied arena reports `vector_slot_limit`
before uploading RHS or applying the operator. The arena's physical byte budget
and the solver's logical numeric-buffer bound are separate reservations.

`MultiRHSResult` exposes aggregate action, orthogonalization and recycling times.
Blocked per-column records describe one shared solve, so the aggregate counts it
once. Action time covers engine application only, orthogonalization covers basis
construction/projection/range factorization, and recycling covers retained-space
projection/replacement. These components exclude residual vector arithmetic,
small least squares, validation and publication; use an outer wall-clock timer
for the complete endpoint. `tools/response_resident_benchmark.py` compares the
same exact CUDA Hamiltonian with host/resident vector storage and checks every
sample against an independent committed-integral matrix. See the
[resident multi-RHS decision](../../.agents/notes/implemented/numerics/2026-09-20-resident-multirhs-response.md)
for numerical, lifecycle and consumer evidence. The
[matched exact-CUDA endpoint record](../../benchmarks/results/response-179-resident/README.md)
includes every measured sample, native binary/source identity, transfers,
synchronizations, resource bounds and complete HVP costs.

The response device budget combines retained direct-J/K storage with the
resident owner allocation. It excludes provider preparation temporaries,
compiler/runtime metadata and CUDA-context/library-private memory. No speedup
is asserted from residency alone; complete solve/action/transfer timings remain
the relevant performance evidence.

The caller owns the `NativeSource` lifetime. Closed sources/backends, unrelated
geometry/basis/reference/Hamiltonian identities, nonsymmetric or nonfinite
inputs, unavailable CUDA and impossible device allocations fail explicitly.
Invalid results never increment successful action counts. The RHF owner is
qualified for closed-shell RHF only. A separate `CudaResidentUHFResponse`
owner now covers exact, unscreened unrestricted HF: alpha and beta
occupied-virtual blocks share one device slot arena, the coupled spin J/K
action stays on the prepared CUDA stream, and the same blocked/recycled GMRES
controller consumes the owner. Density-fitted UHF and KS/CPKS resident owners
remain unsupported until their device action and independent numerical evidence
are qualified.

```python
import numpy as np
from tools.generativeqc_posthf.sources import NativeSource
from tools.generativeqc_posthf.export import export_rhf
from tools.generativeqc_response import CudaDirectJKBackend, RHFResponseOperator

with NativeSource([(1, (0, 0, 0)), (1, (0, 0, 1.4))]) as source:
    reference, _ = export_rhf(source, backend="cpu", tolerance=1e-12)
    with CudaDirectJKBackend(source, device_budget_bytes=64 << 20) as backend:
        problem = RHFResponseOperator.build_problem(reference, backend)
        operator = RHFResponseOperator(problem, backend)
        action = operator.apply(np.ones(problem.dimension))
```

Run `tests/python/test_response_direct.py` for CPU-only ownership/identity and
failure contracts. In an explicitly allocated Slurm GPU job, set
`GENERATIVEQC_RESPONSE_CUDA_TEST=1` and run
`tests/python/test_response_direct_cuda.py`. Device tests compare raw signed J/K
with committed independent AO-integral fixtures (including an f-shell case),
CPHF actions with explicit MO matrices and finite orbital rotations, and all
three shared multi-RHS strategies with independently evaluated true residuals.
One test binds an actual native RHF SCF snapshot. These checks are not complete
molecular-Hessian or all-device-solver acceptance.

See the [direct CUDA response decision](../../.agents/notes/implemented/numerics/2026-09-19-direct-cuda-rhf-response.md).

## UHF CPHF boundary

`UHFReferenceSnapshot`, `UHFSpinRotationLayout`, and `UHFResponseOperator`
provide the shared alpha/beta CPHF contract.  The packed vector stores all
alpha occupied-virtual rotations followed by beta rotations.  The matrix-free
action uses the native UHF Fock convention: Coulomb is evaluated from the
spin-summed density response, while each exchange action uses its own spin
density.  This keeps both spin channels coupled without storing an AO N^4
response tensor.

The UHF snapshot binds both canonical spin references, occupations,
Hamiltonian and SCF generation.  Consequently a recycle space cannot be
reused merely because alpha/beta dimensions happen to match.  The generic
GMRES, blocked multi-RHS and recycling APIs operate on this response problem
unchanged.

The UHF operator remains HF-specific. Native UKS CPKS reuses its spin
layout and orbital-action implementation through the distinct UKS adapter
described above; the actual native KS state supplies its functional identity.

The direct CPU bridge can export a converged open-shell UHF solution through
`export_uhf`.  It canonicalizes the independently returned alpha and beta AO
densities, rechecks both physical commutators and density/Fock reconstruction,
and binds the result to the shared UHF response contract.  The bridge is
intentionally limited to the small direct CPU Hamiltonian. The older
`CudaDFJKBackend` and `CudaDirectJKBackend` remain RHF-specific and are rejected
by UHF, as pinned by `tests/python/test_response_uhf.py`.

`CudaSpinJKBackend` explicitly prepares an unrestricted CUDA `FockPlan` for
either exact or density-fitted J/K with zero screening. One evaluation produces
`J[Delta Pa+Delta Pb]`, `K[Delta Pa]`, and `K[Delta Pb]`; the existing UHF operator
then applies the same orbital action and shared Krylov controller. It uses raw
J/K rather than subtracting hcore from a total Fock, preserving tiny signed
directions. CUDA contracts the integrals; the default response path keeps
AO/MO transforms, returned matrices and Krylov vectors on the host. For exact
conventional UHF, `backend.resident_response(problem)` opts into the resident
owner described above; it shares the direct provider stream and keeps both
spin blocks and response scratch on device while returning only scalar
reductions and the final solution. DF remains host-orchestrated.

The backend borrows a `NativeSource` and owns its copied prepared Fock plan.
Reference validation binds geometry, actual orbital basis/representation,
Hamiltonian and both spin occupations. DF requires explicit auxiliary shells;
its identity binds the prepared plan's mathematical identity, metric cutoff and
retained rank. Its `prepared-spin-df:` identity is deliberately distinct from
the older standalone `MetricFactor` identity. A matching native UHF snapshot is
available through `backend.export_reference()`: this explicitly invokes the
existing native SCF, canonicalizes its returned densities, and checks physical
commutators plus density/Fock reconstruction with the same CUDA plan. Export
uses CPU overlap/hcore preparation and NumPy canonicalization. Response actions
never invoke SCF or CPU integral tiles.

```python
from tools.generativeqc_posthf.sources import NativeSource
from tools.generativeqc_response import CudaSpinJKBackend, UHFResponseOperator, solve_many

with NativeSource(atoms, basis, auxiliary_basis=auxiliary, charge=1,
                  multiplicity=2) as source:
    with CudaSpinJKBackend(source, approximation="density_fitted",
                           device_budget_bytes=64 << 20) as backend:
        reference, report = backend.export_reference()
        problem = UHFResponseOperator.build_problem(reference, backend)
        operator = UHFResponseOperator(problem, backend)
        result = solve_many(operator, rhs, strategy="recycled")
```

The device budget bounds the provider's retained J/K allocations, excluding
preparation temporaries, reference-export SCF/eigensolver caches, host
matrices/solver workspace and CUDA context/library storage. Statistics
distinguish setup and successful action timing; host API
payload counts are not measured PCIe transfer counts. Closing the backend or
borrowed source invalidates actions and zero-RHS solves. Invalid directions,
failed SCF and impossible budgets cannot publish a successful action.

With `GENERATIVEQC_RESPONSE_CUDA_TEST=1` in a Slurm allocation,
`tests/python/test_response_spin_cuda.py` checks independent signed raw J/K,
native open-shell snapshots, explicit coupled MO matrices, true residuals for
sequential/blocked/recycled solves, empty spin, identity and failure replay.
See the [spin CUDA response decision](../../.agents/notes/implemented/numerics/2026-09-20-spin-cuda-response.md).


## Resident response failure and validation scope

Exceptional context exit destroys the resident native owner even when a solver
traceback still retains vector leases. The original solver error is preserved;
subsequent use of a retained vector rejects its closed owner. Ordinary explicit
`close()` still rejects live vector leases. Hardware-independent lifecycle
regressions cover memory, validation and runtime errors and idempotent teardown.

The resident CUDA numerical comparison in `test_cuda_runtime.py` requires an
explicit NVIDIA device allocation (`GENERATIVEQC_RESOURCE_CUDA_TEST=1`) and skips under
`CUMETAL_ROOT`. The CuMetal workflow reports that skip; its green status is not
resident-response numerical qualification. NVIDIA compilation, host GMRES tests,
and ownership tests are distinct from executing the resident operator/solver
against the independent host-orchestrated CUDA reference.

### Combining screened response with optional accelerators

A recycled initial guess and optional DF inverse can seed the screened provisional
solve. Acceptance still requires the fresh scalar, zero-screening physical
residual. A refused provisional solve or failed physical audit receives one exact
diagonal correction; a failed optional accelerator in the unscreened path also
receives one exact diagonal retry. All attempted operator, iteration, and
preconditioner counts are retained. Only the final independently audited exact
operator image is eligible for recycling after the nuclear derivative gates.

The private force benchmark retains DIIS at argument 8, CCSD Q batch at 9, orbital
screening/profile/nuclear controls at 10–12, and the derived-denominator selector
at 13. Residual interval, DF preconditioning, and repeated recycling append at
14–16, followed by packed DIIS at 17 (off by default). After an abandoned
resource-limited force attempt, incomplete phase work
and timing fields are null; complete elapsed endpoint time remains available.

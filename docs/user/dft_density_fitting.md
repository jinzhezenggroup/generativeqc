# Density-fitted DFT energy and force interface

`Calculator(method="pbe-rks", device="cuda", density_fitting="auto",
auxiliary_basis="def2-svp")` selects the shared native DF Coulomb provider for
KS calculations. `auto` follows the calculation backend; explicit
`cpu`/`cuda` DF selections must match `device`. Omitting the auxiliary basis
uses the orbital basis as the auxiliary basis, as in the existing HF interface;
choose an appropriate fitting basis for scientific production calculations.

CUDA HF/KS accepts orbital shells through f and full-range SCF fitting
auxiliaries through h (`l=5`) for energies and first nuclear derivatives,
including the complete Se/Br `def2-universal-JKFIT` spaces. Auxiliary-center
and metric response use the native bounded derivative owner. The h domain
is specific to SCF auxiliary inputs: it does not extend orbital-g,
correlation auxiliary, long-range auxiliary-h, or higher-derivative support.
The generated shell derivative consumer includes auxiliary g, reusing its
primitive geometry across shell components. Generic and bounded panel paths
remain available; auxiliary h uses the bounded FP64 panel consumer rather
than a shell specialization. Mixed shell arithmetic remains limited to
auxiliary s/p/d/f.
For explicitly fitted LR, `GENERATIVEQC_DF_WEIGHTED_EXECUTION=shell` selects
the strict FP64 shell response candidate. It shares the generated omega-aware
radial algebra and disables Coulomb-only screening. Automatic LR response
continues to use the generic derivative; this selector changes execution rather
than the fitted Hamiltonian.

Restricted FP64 all-electron H-Ar DF calculations use native Modified Hückel
under `initial_guess="auto"`, on CPU and CUDA. Exact atomic preparation changes
only the initial density: the molecular fitted operators, auxiliary basis,
metric threshold and force response stay unchanged. Warm/imported densities
take precedence, and `initial_guess=None` restores Hcore. See
[initial guesses](initial_guesses.md) for bounded fallback and compatibility.

CPU and CUDA support local/semilocal RKS/UKS density-fitted energies and
analytic forces. Full-range global hybrids use matching DF-JK on both backends;
the CUDA path reuses the prepared fitted J/K provider and its occupied-RI-K
trajectory optimization. For the qualified molecular range-exchange/VV10
composition (including WB97M-V), the default automatically admits **complete
RI-J/K**: the full-range J/K owner and the long-range K owner are both fitted
when the native LR metric has a resolved fixed-cutoff response and the initial
owner fits its bounded resources. An unresolved cutoff or optional
resource/capability refusal retains full-range DF J/K plus exact Direct LR K,
as described below. Setting
`GENERATIVEQC_DF_RANGE_EXCHANGE=df` explicitly requires complete RI-J/K.
The fitted LR owner uses
its own `erf(omega*r)/r` three-center integrals and auxiliary metric, with the
same auxiliary basis and relative rank cutoff as the full-range owner. It does
not reuse Coulomb-whitened factors for a different radial operator. CUDA orders
the independent provider streams with device events, without a successful-path
host fence during SCF. A cross-provider occupied-factor force response orders
the producer before entering its synchronous derivative consumer.

With complete RI-J/K selected, FP64 CPU and CUDA analytic-force implementations
differentiate both fitted operators, including auxiliary-center and metric
response, self-consistent VV10, Pulay and moving-grid response. Admission follows
the resolved primitive graph and
generated provider capabilities, not a functional-name whitelist. Other
range-separated/nonlocal inventories remain rejected, as do ECP DF and
automatic mixed precision. Full-range DF forces
continue to differentiate the same auxiliary basis and Coulomb metric used by
the energy, including auxiliary-center and metric response. This interface does not change
the selected functional or grid.

`GENERATIVEQC_DF_RANGE_EXCHANGE=direct` explicitly selects the mixed
scheme: full-range DF J/K plus exact Direct LR. `auto`, `df` and `direct` are
accepted when the variable is set; unset is `auto`. Complete RI-J/K is a
scientific approximation, not exact LR with a different execution schedule.
Energy-only and energy-plus-force requests use the same preparation-time selection, including
warm replay and geometry rebinding; requesting forces never changes the
Hamiltonian. For an explicit `df` request, missing LR fitting capability or
insufficient memory is an error, not permission to silently switch Hamiltonians.
Native snapshots authenticate the actual LR approximation, so derivatives never
infer it from the current environment.

Default or explicit `auto` asks the native LR metric owner to admit its
fixed-cutoff response before the first energy, independently of the requested
properties. An unresolved cutoff, optional allocation refusal or unsupported
native LR provider selects exact LR at this initial boundary only. Invalid
controls, eigensolver failures and other CUDA errors still propagate. The
resolved approximation is frozen per prepared batch item, including geometry
rebuilding and retries: an initially fitted item fails if a later geometry has
an unresolved cutoff, rather than silently changing its Hamiltonian. No trial
force or reference eigensystem participates in this decision. Use explicit `df`
or `direct` when reproducibility requires the same approximation across separate
preparations with different initial geometries or resource availability.

This automatic LR admission qualifies energies and analytic first derivatives,
not fitted-LR second-order response. WB97M-V Hessians and HVPs still require
`GENERATIVEQC_DF_RANGE_EXCHANGE=direct` with a full-rank primary fitting metric.
A fitted LR owner is rejected by those consumers, never differentiated with the
exact-LR response equations. See [DF Hessian contracts](../developer/dft_hessian.md).

For automatic CUDA packed storage, the fitted LR owner may rebalance bounded
value staging to retain immutable Hermite geometry for its raw three-center
integrals. This preserves the admitted fitted tensor and occupied capacity,
the response allowance and all explicit memory caps. Unsupported mappings,
insufficient capacity or optional allocation refusal keep the bounded generic
source; they do not select exact LR or change the fitted metric.

Range-fitted orbital shells support s/p/d/f and auxiliary shells support
s/p/d/f/g. The generated g adapter uses the same omega-dependent value and
center-response algebra; unsupported higher angular momenta fail at admission.
For complete RI-J/K on CUDA, an explicit `density_fitting_memory_budget_bytes`
is split between the two prepared radial owners; zero retains each owner's
automatic admission.

Both radial owners can consume the SCF's proved occupied coefficients for RI-K.
Each reserves its own integral and projection storage against the live memory
envelope. When complete occupied projection storage does not fit, a resident
packed integral tensor can still feed bounded projection panels; if neither
representation fits, the source remains streamed. Explicit memory caps and
force-response capacity remain binding. Optional prepared Hermite transforms
accelerate packed Coulomb and erf integrals with the same radial recurrence.
An exact four-center residency request never replaces a fitted LR operator.
For a restricted final state with an authenticated occupied factor, a
rank-truncated LR metric uses raw occupied-space projections for force response.
The complete inverse-root derivative retains motion between kept and discarded
metric directions. The response does not reconstruct raw integrals from fitted
values or relax the configured cutoff. Its projected storage and bounded AO
output panels remain charged to the LR response allowance.

**VV10 itself is not an RI-J/K integral.** Its nonlocal correlation remains the
existing self-consistent grid-pair evaluation; complete DF-J/K does not replace
that kernel with a low-rank approximation or remove its quadratic grid work.

The canonical r2SCAN-3c composite has a separate DF force-composition
admission: its versioned def2-mTZVPP basis is fixed, the fitted r2SCAN
electronic response retains auxiliary/metric and moving-grid derivatives, and
its D4+gCP force/energy corrections remain separate and counted once.
Other external-correction DF MethodIRs stay energy-only until their complete
derivative compositions are qualified. This does not admit ECP or unsupported
auxiliary/basis domains, nor does admission alone establish a successful
source-matched independent CUDA qualification.

For an initially exact fallback or explicit mixed mode, the CUDA force planner
dry-queries the native LR derivative capacity under the live final-state token before AO/Becke tile
admission. It charges that independent owner alongside the complete nonlocal/grid
inventory, with the same conservative
bound on cold creation and warm reuse; a smaller AO tile cannot hide its storage.
This derivative-only allowance uses the actual Cartesian dimension. The primary
SCF provider keeps its separate allowance for optional Coulomb acceleration,
including when it also owns first-order resources.
The derivative bridge then enforces that additional allowance. Its reported host/device
bounds do not include the DF response owner's separate resource envelope and
must not be interpreted as a whole-force peak-memory bound. Both the final-state
lease and orbital/auxiliary geometry bindings are checked on replay.

CPU uses the same resolved-provider validation, fitted J/K response, and native
five-source composer as CUDA. Complete RI-K obtains the LR response from its own
fitted provider and traverses no Direct LR quartets. Explicit mixed mode instead
uses the compiler-generated **exact LR** derivative, weighted by the difference
between LR and SR exchange coefficients; it never substitutes exact SR/full-range
exchange for fitted K. Semilocal, VV10, Pulay, nuclear and moving-grid sources reuse the
existing complete CPU stationary consumer. Production execution does not call
PySCF or a reference oracle.

CPU scalar exchange can retain an optional auxiliary-major copy of each fitted
three-center tensor, capped at 16 MiB per owner. It changes layout, not the fitted
Hamiltonian, precision or auxiliary rank. The copy is included in retained numeric
capacity; this per-owner cap is not a whole-process memory bound. Tensors outside
the scalar cache envelope, or optional allocation failures, keep the uncached
contraction. The external dense-linear-algebra provider retains its existing
layout policy independently of this scalar cap.
The cached scalar schedule can additionally use one AO-matrix transpose and one
AO row of temporary scratch; if that optional allocation fails, it retains the
cached dot-product contraction instead. All auxiliary columns and the original
per-element accumulation order remain unchanged.

The CPU RSH-DF force domain covers all-electron s/p/d/f orbitals, at most
8 atoms, 64 AOs and 128 primitives, with native unpruned grids and distinct
nuclei. Its 256 MiB additional numeric inventory excludes DF response internals,
Python/compiler objects and opaque BLAS/runtime storage, so it is not a complete
force or process-memory bound. The fitted consumer also enforces 32,000,000 primitive
records and 100,000,000 grid-pair visits; quadratic VV10 work and, in mixed mode,
the exact LR quartet traversal are included. These semantic limits describe generated
consumer work; native DF response work is separately owned and unreported.
In explicit mixed mode, the exact LR schedule visits each canonical eightfold quartet once and sums
its distinct compiler-generated density weights before differentiation. Packed
public-AO coefficients retain Cartesian normalization and real-spherical
pullbacks. CPU and CUDA share the same bounded Hermite/forward-AD recurrence
and radial moments, while their host versus resident-device scheduling stays
backend-specific. CPU retains one generic operator-frozen provider, not one
compiled artifact per Cartesian component. Public mixed-DF force requests use
512-quartet weight tiles; the additional inventory charges those tiles and the
bounded scalar recurrence workspace. Primitive visits and actual atom-seeded
recurrence evaluations are reported separately. Native DF response work is
still outside these counters.

The resident CUDA domain is unchanged. Independent reconverged mixed-DF energy
finite differences cover RKS/UKS, Cartesian/spherical bases through f, with
warm replay and orbital/auxiliary geometry rebuilds. Primitive dddd/ffff tests
independently check both SR and LR derivatives against libcint, including
normalization, spherical expansions and repeated-center pullbacks.

```python
from generativeqc import Calculator

calc = Calculator(method="pbe-rks", basis="sto-3g", device="cuda",
                  density_fitting="auto", auxiliary_basis="def2-svp")
result = calc.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
                          properties=("energy", "forces"))
```

The prepared owner copies auxiliary shells before the public descriptor is
released. Batch geometry changes rebind both orbital and auxiliary centers.
CUDA iterations enqueue DF J and XC on the DF owner's stream, retaining density
and Fock matrices on the device; no CPU integral/reference retry is selected.
The ordinary host-controlled KS convergence loop is used for DF. The opt-in
multi-iteration direct-J solver region remains restricted to direct J.

`density_fitting_relative_threshold` controls metric rank selection.
`density_fitting_memory_budget_bytes` bounds the native CUDA DF provider's
explicit source/value storage through its existing bounded tile planner; it
is not a cap on the full KS calculation. Infeasible budgets fail explicitly.
Prepared CUDA batches expose the provider's metric diagnostics. Whole-KS
`estimate_resources`/resource-plan admission is rejected for DF until its
combined inventory is qualified; conventional inventories must not describe DF.

Large ordinary fitted CUDA force workloads may use the native active-AO bitmask
consumer under the unchanged stationary force budgets. It examines all requested
AO value/derivative jets at the existing `1e-16` force cutoff, including second
derivatives where needed; SCF masks are not reused for a different derivative
order. No grid points, weights or Becke response terms are pruned. Missing map
capabilities, insufficient optional storage or excessive AO occupancy retain
the dense bounded consumer. This admission does not include the separate DF
response owner's memory or establish a whole-force peak-memory bound. See
[stationary CUDA scheduling](../developer/stationary_cuda_scheduling.md).

For a CUDA restricted fitted hybrid with method-owned integer occupations,
automatic value storage keeps a fully resident dense owner when it fits. If
dense storage would stream, the same planner may instead retain one symmetric
packed fitted tensor within the resolved DF envelope. For live automatic
budgets only, the singleton restricted owner may rebalance the value/response
split to admit it without increasing the total cap or consuming device
headroom. Response retains at least 20% of that envelope and a conservative
bounded occupied-response workspace; if both owners cannot fit, the original
split and streamed fallback remain. Explicit positive budgets and failed
memory probes retain their original split. The same bounded admission may
retain the method-owned complete occupied projection for the future force
endpoint, rather than dropping it while unused response capacity remains.
The completed projection still requires a valid final-state lease. Optional complete
occupied projections remain separately charged and may be dropped. When the
packed owner cannot fit, the bounded dense/source fallback remains available.
Packed-single preparation batches metric transforms across lower AO-pair rows
within the existing charged scratch buffers; it does not allocate a retained
raw tensor or change the symmetric whitening convention.
Explicit `GENERATIVEQC_DF_VALUE_STORAGE` selections stay authoritative;
unrestricted and arbitrary-density prepared callers do not acquire this policy
from their dimensions alone. This changes source reuse, not the fitting metric,
SCF thresholds or numerical precision.

For a matched prepared-owner cold comparison, use
[`benchmarks/pbe0_df_cold.py`](../../benchmarks/pbe0_df_cold.py) through Slurm.
It measures preparation through the first synchronized public result with a
fresh process, owner and density, retaining persistent disk caches. Imports,
CUDA context initialization, native Calculator construction and post-result
owner/process teardown are outside that endpoint timer. Energy-only and
energy-plus-force runs are separate cold calculations.

Qualified force calculations use token-checked derivative snapshots and the
prepared DF response provider. The CPU diagnostic requires `execution="native"`
for a fitted state; the Direct-only reference derivative path is rejected rather
than differentiating a different Hamiltonian.

The CPU bridge contracts retained host H'/S' derivatives for the one-electron
and Pulay sources. CUDA fitted energy preparation defers those coordinate-major
matrices when retaining future force capability. CUDA first borrows the
token-checked final stationary D/W
already retained by the KS owner and runs the bounded paired one-electron
consumer without uploading those AO matrices again. If that optional device
consumer cannot be admitted under the caller's budget, the prepared owner
materializes H'/S' with the same CUDA exporter on demand, retains them for
replay, and uses the exact host contraction as its bounded fallback. Explicit
derivative Fock preparations still export their requested matrices eagerly.

For restricted CUDA DF response, the Coulomb J' component also borrows the exact
final resident density under the same live KS token. The detached host density
remains the finite/symmetric scientific witness, but the response bridge reads
the device matrix directly and therefore reports zero density H2D bytes for
that J-only call. Exchange K' deliberately keeps its existing
density/projection path, and unrestricted/multi-term response retains the
ordinary upload path. This is therefore not a zero-upload resident whole-force
path.

Full-rank J-only response contracts the resident whitened factor once with the
folded density, applies its symmetric metric root to that charge, and emits
bounded three-center and metric cotangents. It does not repeatedly refit AO
panels or construct exchange-style AO Gram matrices. When only the raw source
is available, it instead reads bounded pair-major panels into existing response
scratch, contracts their density charges, then applies the metric inverse in
eigenvector order. This batches source reads without extra storage or a tensor
transpose. Rank-deficient metrics, mixed J/K terms and diagnostic algebra retain
their general response routes. `GENERATIVEQC_DF_COULOMB_RESPONSE=panels` selects
the original fitted-panel route for qualification; `slices` retains individual
auxiliary reads for source-only J. The default is `auto`. All preserve the same
metric gauge, auxiliary-center response and derivative consumer. The selection
is recorded in resource and restart identities.

Sufficiently large full-rank source-only pure-J response automatically uses the
generated shell derivative consumer, including auxiliary g functions. This
selection does not require a retained raw/fitted tensor and leaves pair layout,
screening and precision unchanged. Small, unprobed, diagnostic and unsupported
owners keep their existing routes. `GENERATIVEQC_DF_WEIGHTED_EXECUTION=generic`
retains the bounded generic consumer for qualification or rollback.

Rank-truncated response differentiates the whitened factors before forming
quadratic products and includes retained/discarded eigenspace motion. CUDA
charges one complete whitened tensor in addition to its bounded raw/weight
panels; insufficient response budget fails explicitly rather than reverting
to an unstable raw-Gram adjoint or to Direct exchange. Full-rank occupied and
resident paths retain their existing admission contracts.

The known native DF snapshot provider separately bounds its additional paired
one-electron/publication storage; DF response scratch stays in its independent
resource contract. Spare geometry budget admits point-parallel Becke phases.
Automatic large fitted grids prefer 256-point tiles to leave room for that
cache; explicit tiles, unknown providers and insufficient budgets retain their
bounded fallback. Reported one-electron device usage must fit the reserved
envelope, rather than relying on unaccounted memory.

Resource metadata distinguishes the resident one-electron path from its host
fallback; `density_fitted_response_resources_included=0` still explicitly
excludes unmeasured DF-provider scratch and transfers. These partial diagnostics
cannot establish a whole-force memory or transport bound. Full DF resource-plan
admission remains unqualified.

`tests/python/test_dft_df_public.py` compares independently converged PySCF
energies and analytic gradients with copied orbital/auxiliary primitives and
identical moving quadrature (absolute energy gate `1e-8 Eh`, force gate
`3e-7 Eh/bohr`, physical residual below `1e-9`). GPU acceptance
requires `GENERATIVEQC_DFT_CUDA_TEST=1` and a scheduler-allocated CUDA device.

`tests/python/test_range_separated_df.py` independently reconstructs PySCF's
omega-dependent fitting metric for RKS/UKS energies and displaced-energy force
checks (energy `2e-8 Eh`, directional force `3e-7 Eh/bohr`). A deliberately small
auxiliary basis distinguishes complete RI-K from the legacy mixed Hamiltonian.
The same gates cover the retained cc-pVDZ-JKFIT auxiliary with SVP/TZVP
orbitals, open-shell response, and fixed-geometry force replay. CPU materialized
and weighted derivatives use the same inverse-square-root response.
Raw contracted Cartesian/spherical s/p/d/f values and center derivatives are
compared with libcint over multiple omega values in `test_df_derivatives.py`.
CUDA implementations require separate real-device qualification; CPU evidence
and successful compilation are not substitutes for that execution.
`test_df_long_range_policy.py` covers auxiliary g values and center derivatives
against libcint. `test_df_md_raw_cuda.py` independently qualifies Coulomb and
erf packed values, mixed basis representations, dirty graph replay and
transactional resource refusal.
`test_df_auxiliary_g.py` and `test_df_auxiliary_g_derivatives.py` also compare
full-range h values and center derivatives with libcint in Cartesian and
spherical representations. `test_se_br_ks_cuda.py` retains the original six
Se/Br inputs and complete fitting spaces. Its opt-in Slurm qualification
(`GENERATIVEQC_SE_BR_KS_CUDA_TEST=1`) requires native cold/warm E/F,
independent matched-root PySCF/D3 E/F and two-step native finite differences;
primitive coverage alone does not establish complete-endpoint acceptance.
The [Se/Br fitting-space decision](../../.agents/notes/implemented/numerics/2026-11-05-se-br-auxiliary-h-and-cold-uks.md)
records the role-specific admission and bounded h response policy.

See the [ownership note](../../.agents/notes/implemented/architecture/2026-09-22-public-df-ks-provider.md)
for the retained boundaries.
The [range-separated RI-K decision](../../.agents/notes/implemented/numerics/2026-10-15-range-separated-density-fitting.md)
records radial ownership, response consistency and numerical qualification conditions.

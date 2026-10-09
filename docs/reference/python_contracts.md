# Public Python contract coverage

The [Python API](api.md) provides signatures, source links, field annotations and
source-owned behavior. [Calculation contracts](python_execution_contracts.md)
cover the main endpoints first. The family contracts below supply shared
preconditions, shape/unit conventions, failure and lifetime boundaries without
copying a backend support matrix.

(python-contract-policy)=
## Audited inventory and applicability

`manifests/python_api_contracts.json` is the exact, source-bound inventory of
the supported `generativeqc` facade's public export occurrences and their
authoritative declarations, including
explicitly defined public/magic class members. Compiler/research modules outside
that facade are included only when their declarations are re-exported through
`generativeqc`, including the curated `generativeqc.extensions` interfaces.
Re-exports and assignment aliases
bind to one source owner. Literal compiler lazy maps are followed statically;
module exports, literal constants, type unions and NumPy dtype tokens are
classified explicitly. Unknown export forms and mutation/aliasing of a literal `__all__` fail with an
actionable diagnostic.
No optional dependency, native library or device is imported to run the gate.
Annotation-only `__all__` statements preserve a preceding runtime inventory.
Public class callable aliases and conditional member definitions are rejected
until expressed as explicit public definitions; ordinary typed data fields and
literal class constants remain supported.

Every inventory record declares a reviewed applicability profile. Execution
requires behavior, values (including applicable shapes/units), errors/failure
publication, ownership/concurrency and backend constraints. Transformations
require values, errors and ownership; records require values and ownership;
validators require values and errors. Lifecycle operations require errors and
ownership without pretending that close has tensor shapes or physical units. Accessors, exception types, modules and
constant/type tokens only require their actual behavioral contract. An identity
hash or integer version does not acquire meaningless allocation or unit sections.

Each applicable facet is bound to its exact source docstring or an explicit
section below. Non-behavior facets require labeled reference sections. Removal
of a facet, label or section body fails; placeholders fail even when a docstring
is nonempty. Added exports or public members, changed alias targets, retired
records and category drift require an inventory update. This is structural
coverage, not a proof that prose is scientifically correct: changes still need
source review and the existing numerical/domain tests. The debt list is empty;
there are no anonymous, wildcard or grandfathered exemptions.

Run `python tools/check_public_api_docstrings.py --ruff /path/to/ruff` and
`python -m unittest discover -s tests/python -p test_public_api_contracts.py`.
Sphinx remains warning-as-error. No word-count threshold determines correctness.

(python-constants-behavior)=
## Version, dtype and exact-scalar tokens

Each public `API_VERSION` describes that facade's construction protocol, not the
native ABI, scientific method version or package release. `FRONTEND_VERSION`
versions the experimental array frontend; `DLPACK_INTEROP_VERSION` versions its
interop contract. `SUPPORTED_FUNCTIONS` is the immutable declared frontend
subset, interpreted together with `capabilities()` and the restrictions in the
[array guide](../user/experimental_array_api.md). Membership is not certification
of every Array API dtype, device or dynamic shape.

Tensor `SCHEMA_VERSION` versions serialized programs and `PRIMITIVE_VERSION`
versions primitive semantics. Preserve these when reading/writing artifacts;
do not use a package version as an interchangeable identity. The `float32` and
`float64` exports are NumPy dtype objects, not conversion functions or device
selectors. `ExactScalar` and XC `Coefficient` are type aliases for `int | str |
Fraction`. Their consuming constructors validate rational spelling and reject
unsupported float/bool coefficients; the alias itself performs no validation.

(python-basis-values)=
## Basis and electron-state values

`BasisSet`, `ElementBasis`, `BasisShell` and `BasisProvenance` describe canonical
local basis data. Shell coefficients are contraction-by-primitive matrices;
combined SP import groups preserve provenance while expanding angular momenta.
Primitive exponents must be finite and positive; exact zero coefficients are
retained. `load_basis` reads canonical JSON and `import_bse` explicitly converts
a supplied BSE JSON file with caller-declared provenance/license. Neither selects
a license by guessing or fetches a network basis service. Serialization and
identity include the normalized content and representation, not just a filename.

`Atom` uses atomic numbers 1–118 and three finite Bohr coordinates.
`electron_state` distinguishes nuclear charge, ECP core removal, ionic charge,
active electrons and spin occupations. Charge and multiplicity are exact integer
identities. A basis can preserve unsupported shell/ECP data without making it
executable. `basis_capability` checks every actual shell/operator/derivative and
returns reasons; eligibility does not prove a device exists or SCF converges.
`load_r2scan3c_basis` supplies the canonical composite basis rather than accepting
an interchangeable named basis.

(python-basis-errors)=
## Basis validation failures

Malformed types, nonfinite/invalid primitives, ambiguous BSE combined/general
contractions, inconsistent nuclear/ECP identities and electron/spin parity are
rejected with validation exceptions. File reads/writes retain filesystem and
JSON errors. Unsupported execution chemistry is separate from parsing valid
basis data; do not turn an ineligible capability record into permission to call
an all-electron ABI with altered ionic charge. No successful basis object is
returned from a failed load/import.

(python-basis-ownership)=
## Basis snapshots and ownership

Loaded basis records own normalized local snapshots. Changing a source file
requires loading it again; it does not mutate an existing calculator's model.
Tuple-based shell/element records and immutable identities can be shared for
read-only use. Returned serialization dictionaries are data, not live native
systems. File writing is explicit through the relevant save operation; readers
must not race a write to the same file. No basis-loading operation owns a GPU
context or starts an electronic solve.

(python-accuracy-values)=
## Accuracy and adaptive numerical values

`ObservableTarget`, `TargetAccuracy` and `TargetErrorBudget` keep independent
energy and force requirements. Energy differences are Hartree; force differences
are `(natoms, 3)` Cartesian components in Hartree/Bohr. Maximum-component and
RMS-over-3N norms are distinct; relative tolerances are dimensionless and use the
reference norm. `compare_observables` compares already computed values; it never
solves a reference or establishes its electronic root.

`ErrorEvidence`, `ResolvedModel`, `NumericalTargetModel`, contribution ledgers
and calibration samples bind the target, evaluated/reference model, scope,
geometry/domain and provenance. A componentwise absolute envelope does not turn
empirical inputs into a theorem. Paired-difference and SCF-force estimators are
calibrated limited-domain estimates. `AdaptiveNumericsPolicy` and
`ForceAwareScfPolicy` propose among explicitly supplied execution levels and
retain hysteresis state; they do not execute a proposed calculation.
`HFConvergence` and `ArithmeticPolicy` describe iteration and final arithmetic
requirements separately from observed error. Final verification checks the
requested target and evidence rather than treating loose-stage convergence as
acceptance.

(python-accuracy-errors)=
## Accuracy validation and inconclusive evidence

Invalid finite/nonnegative tolerances, force shapes, model identities and
calibration metadata are rejected by the relevant constructor/validator.
Unsupported evidence claims, including unimplemented proven-bound claims, fail
closed. Missing, out-of-domain or mismatched evidence yields the documented
unverified/invalid outcome or conservative policy decision; it is not silently
promoted to a passed accuracy target. A policy requiring tighter work is a
proposal for the caller, not automatic scientific fallback or a completed solve.

(python-accuracy-ownership)=
## Accuracy and policy state

Evidence/decision records are Python data. Preserve their model/provenance
identities when serializing or comparing them. Policies and estimators that
collect samples/history are stateful: use a separate instance per optimization
sequence or serialize updates. Do not mutate retained observations during
assessment. Array-bearing records follow their source constructor's snapshot
contract; a frozen dataclass alone does not recursively freeze arbitrary data.
No policy decision reserves native memory or takes ownership of a calculator.

(python-resources-values)=
## Resource plans and capacities

Resource byte counts are nonnegative exact integers within the portable signed
64-bit range; `None` means unlimited and zero permits no allocation.
Host limits include pageable plus pinned memory, with an additional optional
pinned cap. Device-total and per-device caps both apply. Reserve/headroom is
withheld before provider selection. Estimates bind owner, memory space, kind,
scientific identity and inclusive logical phase lifetime. Shared buffers count
once under their owner; overlap of phases models concurrent residency.

`estimate_hf_resources`, `estimate_ks_resources`, `Calculator.estimate_resources`
and `plan_resources` construct metadata plans, not scientific tensors or solves.
The HF/KS estimators accept the same nonempty ragged atom systems, per-system
charges/multiplicities and basis/representation choices as the calculation
boundary; omitted charges/multiplicities are neutral singlets. Their keyword
options select the explicit method/provider, SCF controls and admitted numerical
policy. CUDA inventory queries may load the native library through its non-GPU-
probing CPU loader to obtain scalar allocation layouts; they do not create a
scientific execution context. Missing inventory ABI support is an unsupported
estimate, not a guessed capacity. Plans choose only enumerated provider alternatives in deterministic cost order.
They distinguish feasible, infeasible and unsupported, preserve scope exclusions,
and include retained/output lifetimes. `ResourceSession` maps request names to
caller-supplied factories and advances integer phases; providers expose `close()`.
See [resource planning](../maintainer/resource_planning.md) for the complete
allocation scope, serialized diagnostic schema and provider protocol.

(python-resources-errors)=
## Resource failure and retries

Malformed byte counts/identities/phase intervals and inconsistent restored plan
accounting fail validation. `require_feasible()` raises `NotImplementedError`
for unsupported scope and `MemoryError` for infeasible capacity. Exhausting the
bounded candidate search reports unsupported, not a false infeasibility proof.
A provider factory may raise `ResourceAllocationError` only from an actual
classified allocation failure, after releasing partial resources. The session
can then try a finite lower-memory candidate; numerical/driver failures are not
reclassified by matching strings. Failed groups close in reverse order before
retry. Scientific execution is never automatically retried by the session.

(python-resources-ownership)=
## Resource ownership and concurrency

Requests/plans are serializable metadata without native pointers. Their
capacity accounting does not include unreported allocator, Python-object or
driver overhead; an estimate is not an observation of free system memory.
A session owns the providers it successfully creates, closes expired owners
before allocating new ones and belongs to one synchronous orchestrator. Do not
advance/close/use its provider concurrently. Already live providers keep their
accepted selection during retries. Providers with internally phased arenas
must manage those arenas themselves or expose separate request owners.

(python-resources-backends)=
## Resource provider boundary

Planning itself neither probes nor initializes a backend. A plan's backend and
scientific identity constrain reuse but do not prove that a device, compiler or
allocator is available. Session factories are the explicit activation boundary;
the provider retains its own backend admission and synchronous execution
contract. An unsupported estimator scope cannot be filled with a guessed zero
capacity or an unapproved scientific approximation.

(python-options-values)=
## Grid, KS and initial-guess options

`GridSpec` describes quadrature with lengths/radii in Bohr, explicit element
radius overrides and versioned topology; `GridProfile`/`GridPolicy` resolve that
topology before element radii are attached. Grid changes change the numerical
model. `KsOptions` snapshots the method, spin-resolved XC composition, grid and
bounded tile schedule. Initial-guess options govern cold-start preparation and
its budgets without changing the target method, basis, grid, precision or final
tolerances. Existing explicit/imported/retained densities take precedence.

(python-options-errors)=
## Option validation and fallback

Constructors reject unsupported versions, malformed positive counts, nonfinite
controls and inconsistent spin/composition. Execution admission is still checked
in the calculator. Preliminary HF/LDA/MINAO guess policies use only their
explicitly admitted domains. Failed preliminary solves or exhausted preparation
budget retain a core guess; allocation failures propagate. A failed seeded
target has the documented bounded core retry. These fallbacks do not establish
profitability, the ground state or observable accuracy.

(python-options-ownership)=
## Option snapshots

Options are declarative records, not native owners. Keep the scientific and
numerical configuration fixed while a prepared calculation uses it. A different
resolved model requires a new prepared owner; equal array dimensions do not make
changed grid/functional/precision identities interchangeable. Inspection and
serialization expose data without running a solve or acquiring device storage.

(python-method-values)=
## Method and XC construction

`extensions.xc` and `extensions.method` construct/inspect canonical scientific
specifications. Coefficients are exact integers, rational strings or `Fraction`;
float and bool coefficient spellings are rejected. Duplicate XC components
combine exactly and cancellation is canonicalized. Spin is explicit:
`unpolarized` or `polarized`. Range parameters, when applicable, carry inverse
Bohr and remain separate from exchange coefficients. D3 cutoffs/switch widths
use Bohr and its variant/table digests enter scientific identity.

`MethodIR.identity` hashes mathematical composition/provenance without the
manifest name; `manifest_identity` also retains the name. `TypedMethodIR.identity`
adds dtype, derivative order and logical feature bindings without replacing
mathematical identity. `BackendCapability` declares immutable finite domains.
Feature shapes are per-point component shapes: unpolarized rho/sigma/tau have
one component; polarized rho/tau have two and sigma three. A runtime grid-point
batch dimension is separate. `verify` checks these bindings and a declared
backend capability; it does not execute or probe that backend.

(python-method-errors)=
## Method representability and failures

Wrong record types raise `TypeError`; malformed exact values raise validation
errors. `UnsupportedXC` and `UnsupportedMethod` reject unrepresented component
IDs, versions, invalid composition, spin conflicts and unsupported primitive
combinations. `MethodTypeError` rejects execution-type mismatches, including
implicit casts, missing/extra features and derivative order outside the admitted
set. Choosing a familiar identifier cannot promote a custom composition to a
qualified production endpoint. Failed construction/type-checking returns no
partially typed object. See the [extension guide](../developer/extensions.md).

(python-method-ownership)=
## Method identities and data lifetime

Canonical specs/IR use immutable tuples and exact coefficients. Payload and
inspection methods publish data mappings rather than executable native owners.
Equivalent mathematical aliases may reuse a semantic identity while retaining
different manifest provenance. `resolve` can return the already canonical IR
unchanged. Treat shared canonical objects as immutable; construct a new object
for changed mathematics. Lazy facade imports do not themselves start a native
execution context, compilation or device probe.

(python-tensor-values)=
## TensorIR shapes, layout and derivatives

Tensor builders create typed SSA nodes with explicit `TensorSpec` axes, real
dtype, symmetry and differentiation metadata. `IndexSpace` names distinguish
scientific populations even at equal extents. Builders preserve logical
row-major order; broadcasting, reshaping, slicing and gathering require their
explicit axis maps/ranges. Tensor values carry no implicit chemistry-unit
conversion: the caller must give operands compatible physical meaning.
`DenseLayout` is an explicit physical-axis permutation/alignment contract.
`PackedLayout` maps dense coordinates to independent signed symmetry orbits;
packed values have shape `(layout.size,)`, while dense values use `spec.shape`.
Pack/unpack transposes include the documented orbit metric and are not ordinary
pack/unpack. Packing is a bounded reference enumeration, not a large-system
storage planner.

`Program` binds named outputs and only output-reachable nodes execute.
`linearize` and `transpose_program` produce derivative graphs with `d_` tangent
and `bar_` cotangent inputs/outputs. `jvp`, `vjp` and `dot_test` evaluate the
requested derivative slice with matching primal shapes/dtypes; omitted tangent
or cotangent entries mean zero for their supported paths. Outputs are identified
by program output names, never inferred from dictionary iteration order.

(python-tensor-errors)=
## TensorIR errors and publication

Builders reject incompatible scientific axes, shape/dtype conflicts, invalid
symmetries, nonexact scalar metadata and unsupported primitive forms. Runtime
feeds must match named inputs' exact shape/dtype and declared symmetries.
Interpreters check finite values/results and domains: division by zero,
nonpositive log/power bases and negative square roots fail; square root at zero
is legal for values but singular for AD. Budget limits include the documented
logical arrays, not an assertion about NumPy internal scratch. Native JIT has
separate work/node/byte limits and a finite compile timeout. Compilation/loading
errors propagate; no compiled handle or execution output is returned as success
when that stage fails. Packing rejects lossy symmetry projection and invalid
finite tolerances.

(python-tensor-ownership)=
## TensorIR ownership and concurrency

SSA inputs are read-only and every primitive defines a new logical value.
Optimization/derivative construction returns new graphs rather than overwriting
the original artifact. `execute` returns detached output/debug snapshots that
alias neither one another nor inputs, including for noncontiguous/read-only
feeds. Compiled handles retain their private native artifact owner; published
artifact/resource mappings are detached. Calls are synchronous. Do not mutate
feeds during execution or share mutable compiler/handle state concurrently
without external serialization. Packed output arrays are produced by the named
mapping; preserve the declared metric when consuming adjoints.

(python-tensor-backends)=
## Tensor execution versus production support

`extensions.tensor.execute` defaults to independent NumPy CPU reference
execution. Optional namespace validation is limited to explicitly portable
primitives; it is not a production execution plan. `compile` is an explicit
CPU-only `mode="jit"` action with lazy toolchain activation and bounded resource
admission. A compiler-cache launcher is required on a cache miss; artifact hits
do not require compilation. `compile_capabilities` lowers/inspects without
probing a compiler or creating cache artifacts. Its `compilable` result proves
lowerability within requested bounds, not installed toolchains or validated
science. Tensor/AD `capabilities()` describes the implemented slice; it never
promotes an arbitrary graph into a built-in method/property domain.

(python-array-values)=
## Experimental array shapes and dtypes

The [experimental array guide](../user/experimental_array_api.md) defines the
versioned subset. Generic arrays use static shape/broadcasting and float32/float64
promotion. Scientific `VibeArray` values retain full TensorSpec index identities
and require explicit compatible operations. `trace`/`input_array` build symbolic
graphs; `compile` captures a callable for `backend="reference"` execution. Static creation supports finite uniform values; `_like`
creation does not erase scientific metadata. Static reshape permits its single
inferred dimension where documented; indexing/gather maps are bounded. `T`
reverses all axes; `mT` swaps only the last two and requires rank at least two.
Both properties build new symbolic transpose nodes and preserve scientific
index metadata, rather than returning a stored scalar attribute.

Eager mathematical operations use NumPy host arrays. There is no chemistry-unit
conversion; the mathematical caller supplies consistent units. `DLPackDevice`
records `(device_type, device_id)` and `DLPackImport` records the verified array,
source/target device and copy-control contract. `float32`/`float64` are dtype
tokens; dtype helpers do not add integer, complex or device domains.

(python-array-errors)=
## Experimental array failures

Invalid shape/axes, unsafe dtype requests, unsupported devices or symbolic
Python control flow fail closed with the operation's TypeError/ValueError or
backend error. Symbolic truth/comparison cannot be used as a dynamic Python
branch. Eager operations retain NumPy's numerical behavior; capture/interpreter
execution additionally enforces its finite/domain checks. A failed capture or
execution does not publish a successful compiled result. Foreign device arrays
are rejected before hidden host conversion, including nested host containers.
DLPack protocol/device/copy failures raise `DLPackInteropError`; a failed handoff
is never retried with weaker copy requirements.

(python-array-ownership)=
## Experimental arrays and DLPack ownership

NumPy eager conversion/reshape/broadcast operations can return views under their
normal `copy`/layout rules. Do not assume eager outputs are detached.
Symbolic values are graph nodes; compiled reference execution publishes its
checked snapshots. Compiled-call state/cache may change, so serialize use of a
shared callable and do not mutate feeds during a call.
`import_dlpack` requests a same-device zero-copy handoff through the destination
namespace's `from_dlpack`, including its stream/capsule lifetime protocol. It
never manually extracts/reuses a capsule. A signature-verified legacy one-argument
consumer is supported; otherwise `copy=False` is required. Imported storage can
be shared with the producer: keep the protocol owners alive and do not race
writes across consumers. Device relocation needs a separate explicit copy.

(python-array-backends)=
## Experimental array backend boundary

Eager and reference compiled-call paths admit CPU/NumPy float32/float64 with
`device=None`. `xp.asarray` does not silently stage a foreign CUDA/DLPack array
to host. The experimental `compile` accepts only `backend="reference"`; native
compilation is a separate `extensions.tensor.compile` operation. Unsupported
backends fail rather than falling back. Capability
reports separate frontend representation, execution and production qualification.
DLPack interop verifies same-device transfer, but accepting a DLPack producer
does not add that device to the eager namespace or native chemistry support.

(python-fock-values)=
## Fock and fixed-density values

`FockBuildSpec` declares operator, approximation, spin, derivative order and
signed coefficients; present-with-zero-coefficient still requests the raw
matrix. `FockPlan.evaluate` accepts a finite real density `(nao, nao)` for
restricted total density or `(2, nao, nao)` for unrestricted spin. Returned
Fock/K follow spin layout and J is `(nao, nao)`, with host float64 storage.
Energy components use Hartree. Optional two-electron energy gradients are flat
`(3*natoms,)` in Hartree/Bohr; they are positive `dE2/dR`, not complete forces.
`solve` returns total declared J/K energy and optional complete `(natoms, 3)`
forces, with no XC contribution. Seeds use the same density layout and must
satisfy electron/spin trace and overlap-metric occupation.

`FixedDensityMeanField` combines compatible Fock/XC/nonlocal providers for a
fixed-density energy and AO potential; it does not perform SCF or provide a
complete geometric XC force. `assemble_fixed_density_exchange` consumes
already-built raw K matrices keyed by exact operator/range parameter. It uses
`Vx=-a*K/2` for restricted total density and `Vx_s=-a*K_s` for unrestricted spin,
with `Ex=Tr(D Vx)/2`. Range omega has inverse-Bohr units. Returned identities
bind density, model and providers, not just matrix size.

(python-fock-errors)=
## Fock failures

Type/shape/spin/finiteness mismatches, invalid term controls and inconsistent
orbital/auxiliary geometry are rejected before publication. Unsupported native
operator/approximation/derivative schedules fail admission. Generated CUDA DF
derivatives require symmetric density even where raw value evaluation accepts
nonsymmetric inputs. An invalid seed is rejected rather than silently normalized.
Native numerical/nonconvergence/allocation errors raise, returning no partial
`FockEvaluation` or `FockScfResult`. A failed solve leaves the plan reusable.
Fixed-density assembly rejects an incomplete/extra raw exchange operator set;
nonfinite assembled energy raises `ArithmeticError`.

(python-fock-ownership)=
## Fock ownership and concurrency

A Fock plan owns prepared native J/K sources and retains its immutable orbital
and optional auxiliary basis snapshots. Density input is copied before native
evaluation/provenance hashing; do not race writes while that snapshot is being
created. Results use immutable published arrays and detached diagnostics.
Plan evaluation, solve, native diagnostics and close serialize through its lock.
Repeated close is safe; closed-plan execution raises. Use context-managed
lifetime. Fixed-density composition retains its supplied providers; keep them
open and serialize integration with any operation that closes those providers.

(python-fock-backends)=
## Fock backend restrictions

Plans explicitly select CPU or CUDA and admitted FP64 operator/approximation
schedules. CUDA integral work does not imply an entirely device-resident SCF:
plan-owned host DIIS/eigensolver policy is stated by `solve`. `FixedDensityMeanField`
is a method-neutral energy/Fock consumer, not a registered DFT SCF/force
endpoint. A declared operator or representable MethodIR is not a support promise;
use prepared diagnostics and native admission for the actual basis and options.

(python-correction-values)=
## Geometry correction values

D3 batches take `(atomic_numbers, coordinates)` systems. D4/composite systems
use `(atomic_numbers, coordinates, total_charge)`, where total charge is a finite
real scalar in electron-charge units (the correction-only EEQ input need not be
an integer). Atom numbers are exact
integers and coordinates have shape `(natoms, 3)` in Bohr. Prepared counts remain
fixed; optional replacement geometry entries retain their stored coordinates
when `None`. Additive correction energies use Hartree. Requested `gradient`
arrays have shape `(natoms, 3)` in Hartree/Bohr and are **positive `dE/dR`**;
subtract them when composing electronic forces. D4 atomic charges have shape
`(natoms,)` in electron-charge units. Correction-only results do not include
electronic SCF energy. Runtime diagnostics report named byte capacities, atom
counts and method/table/provider identities.

(python-correction-errors)=
## Correction status and failure publication

Malformed systems, wrong replacement layout and invalid model records raise
validation errors. Native whole-call failures raise. Batch execution returns
input-ordered item records; test `ok`/status before consuming energy. Requested
gradient/charge arrays are omitted (`None`) for failed items. One-shot
`evaluate_*_correction` and gCP helpers raise on an item failure rather than
returning it as success. D3 replacement nonfinite admission can become a native
item failure; D4 replacement geometry checks finiteness before replay. A valid
parameter record with a mismatched compiled table/variant identity is rejected,
not silently substituted with a different correction.

(python-correction-ownership)=
## Correction plan lifetime

Prepared correction batches retain their own native context/model/topology
resources; close explicitly or use a context manager. Returned gradients and
charges are copied from execution buffers and survive close. Calls are
synchronous. These mutable prepared owners do not provide a concurrent replay
or close guarantee: use separate plans or external serialization. Diagnostic
queries require an open owner. One-shot helpers close their temporary batch.

(python-correction-backends)=
## Correction production domains

D3 damping/ATM variants and D4 EEQ/ATM domains have separate explicit admission,
versions and table hashes. Do not infer executability from the presence of a
parameter entry. Native provider availability and memory limits are checked
when preparing the selected CPU/CUDA owner. Canonical r2SCAN-3c corrections
require that exact composite MethodIR and combine D4 with the native CPU gCP
kernel; choosing CUDA for a component does not imply every composite component
is device resident. The correction API does not add electronic method support.

(python-response-values)=
## Orbital and nuclear response layouts

`RotationLayout` uses occupied-major/virtual-minor vectors of length
`nocc*nvirt`; `as_ia`/`pack` convert `(nocc, nvirt)`. Density/generator matrices
are `(nmo, nmo)` with the explicitly documented symmetric-density or skew
orbital-generator conventions. `ResponseProblem` binds reference, ordering,
gauge, model, operator and perturbation identities. Equal dimensions are
insufficient for compatibility. RHS vectors use that rotation layout; `validate_rhs` returns `(dimension, k)`
columns (a single vector becomes one column). Nuclear multi-perturbation inputs
are `(nrhs, nmo, nmo)` and share a positive `nrhs`.
`DenseMatrixResponseOperator` expects `(dimension, dimension)`.

Nuclear RHS builders accept square MO Fock/overlap derivative matrices and
`(nmo,)` orbital energies, with closed-shell `0 < nocc < nmo`, and return
`(nocc, nvirt)` components. Orbital energies/Fock use Hartree; nuclear Fock
derivatives use Hartree/Bohr and overlap derivatives inverse Bohr. Response
coefficients inherit the supplied perturbation's units. Fixed-density XC
response consumes AO density/direction matrices: restricted total-density
adapters split equally by spin, and `apply_spin` retains the functional's
one- or two-channel response. `density_feature_response` consumes finite real AO jets
`(njet, npoint, nao)` with 4, 10 or 20 derivative rows and symmetric total
`(nao, nao)` or spin `(2, nao, nao)` densities/directions. It returns rho/tau
responses `(2, npoint)`, density gradients `(2, npoint, 3)` and sigma responses
`(3, npoint)` in alpha-alpha, alpha-beta, beta-beta order; sigma's cross term
has no extra factor of two. Units follow the AO spatial derivatives and density
convention, without conversion. Nuclear response records retain source identities
and solver diagnostics; RHF-named records are aliases of the stationary owners.

(python-response-errors)=
## Response admission and failure

Invalid finite shapes, layout/reference mismatches and unstable/gap-inadmissible
references fail validation. `ResponseUnsupported` distinguishes unsupported
physics and `ResponseCompatibilityError` incompatible reuse.
`ResponseSolveError` carries a failed solver result where the solver contract
provides one; callers must not interpret it as converged response. Strict solve
paths raise on failure. Dense comparison actions retain NumPy shape errors and
are control tools, not the production matrix-free operator. Native live-state
changes invalidate the response owner; do not reuse a response snapshot after
its batch/reference changes. Wrong XC spin/reference density or grid/model
identity is rejected before claiming a compatible kernel.

(python-response-ownership)=
## Response lifetime and concurrency

Live response adapters borrow source state/basis and validate current identity;
keep those owners open and unmodified while applying/solving. `NativeRKSResponse`
closes its installed response providers, not a license to invalidate borrowed
source owners. Context-managed close is preferred. Detached result/reference
arrays follow their documented immutable publication; dense control matrices
may share an `asarray` input and must not be mutated during use. Operator action
statistics and solver/recycling state are mutable: serialize operations on a
shared operator and do not race source replay or close.

(python-response-backends)=
## Response backend domains

RHF and CPKS operators delegate method-specific Fock/XC actions to explicitly
bound providers. `NativeRKSResponse` admits the checked live all-electron,
unscaled LDA/PBE RKS state and installs the matching CPU/CUDA providers;
this is not blanket support for arbitrary functionals/spin/ECP response.
`DenseMatrixResponseOperator` is explicit dense comparison evidence, not a
matrix-free production implementation. Nuclear algebra helpers do not expand
the injected operator/solver's physical domain. Low-level response support does
not imply a public molecular Hessian endpoint; see
[contextual capabilities](capabilities.md).

(python-integrals-values)=
## Generated RKS integral derivative values

`RKSIntegralTopology` binds a live direct all-electron Cartesian AO basis,
ordered atoms/shells and `nbf`. `checked_direction` requires finite real
`(natoms, 3)` input and returns a detached float64 array. AO weights have shape
`(nbf, nbf)` with the symmetric finite admission of the consuming contraction.
Directional first-order providers return the named overlap/Hcore/Coulomb
response components in their source docstrings. Weighted first gradients and
second-order HVPs return atom/Cartesian components. Coordinates use Bohr,
electronic gradient contributions Hartree/Bohr, and Hessian components
Hartree/Bohr²; overlap-only derivatives inherit inverse-length units instead.
Memory budgets are bytes for the declared generated integral work, separate
from full molecular response and output storage.

(python-integrals-errors)=
## Generated derivative failures

Closed/mismatched borrowed AO owners, unsupported representation/shell orders,
nonfinite directions/weights and invalid backend/budget controls fail before
claiming a derivative result. Compilation and native execution failures
propagate; no partial contraction is published as a completed HVP. Byte/work
admission bounds the generated provider, not every surrounding response
allocation. `checked_second_hvp_options` rejects unsupported second-order
backend/compiler options rather than routing silently to another provider.

(python-integrals-ownership)=
## Generated derivative lifetime

Topology records borrow the NativeAO basis and require it to remain open.
Generated programs/artifacts can be cached, but cache identity never makes a
closed scientific owner reusable. Inputs are read during the synchronous
contraction; do not mutate arrays or replay/close their live source concurrently.
Returned derivative arrays are detached publication. Serialize operations that
share live owners or mutable compilation/cache state.

(python-integrals-backends)=
## Generated derivative backend restrictions

CPU and explicitly named CUDA first-order providers are separate entry points.
The topology currently admits Cartesian shells through f for this bounded
RKS derivative slice; broader value-integral support is irrelevant here.
Second-order options and the complete public molecular HVP remain independently
admitted, including the CPU-only second-order restriction. Generated-provider
availability is not end-to-end method/derivative qualification. Compiler cache
and finite timeout requirements apply when a requested artifact must be built.

(python-projection-values)=
## Projection and overlap shapes

`cross_overlap(target, source, ...)` returns host float64
`(target_nao, source_nao)` overlaps with target rows/source columns. Geometry
uses Bohr and can differ explicitly for the source. Overlap is dimensionless.
`project_occupied` takes square source/target overlaps, that rectangular cross
overlap and `(source_nao, noccupied)` coefficients with `C.T S C = I`.
Occupation 2 produces restricted density; separate occupation-1 calls represent
alpha/beta subspaces. `project_density` validates its density/metric contract
before extracting the occupied space. Output coefficients/densities are target-AO
quantities; they are warm proposals, not converged target solutions.

(python-projection-errors)=
## Projection rejection and publication

Nonfinite/wrong shapes, insufficient metric rank, nonorthonormal source
coefficients and failed target occupation tests reject the entire proposal
with `ProjectionRejected` or validation errors; missing orbitals are not
invented. Raw overlap requires both calculators to use the same native library
and admitted CPU overlap shells. Exceeding `maximum_bytes` raises `MemoryError`.
Native failures propagate without returning a partial overlap matrix.
Progressive callers may catch a rejected proposal and use their explicitly
documented cold start; the algebra helper itself does not run that fallback.

(python-projection-ownership)=
## Projection ownership

Projection is synchronous host linear algebra and publishes immutable coefficient
and density snapshots. Raw overlap owns temporary native context/systems and
releases them before returning its independent array. It borrows calculator
basis data for setup, not the calculators' execution state. Do not mutate
inputs/configuration while projecting or computing overlap. No projected result
owns a prepared target calculation or transfers a source native handle.

(python-projection-backends)=
## Projection backend boundary

Raw cross overlap is an explicit native CPU setup operation even for accelerator
calculators; it allocates no ERIs or nuclear derivatives. Projection algebra is
host NumPy over supplied matrices. These helpers do not promise a faster total
endpoint, change the target model, or expand its execution support.

(python-progressive-values)=
## Progressive HF execution

`make_deterministic_hf_plan` binds source/target model, convergence, arithmetic,
resource budget and transfer policy. `run_progressive_hf` and
`projected_singlepoint` use the ordinary Bohr-coordinate atom/charge/multiplicity
contract and finish with the target calculator. Returned energies/forces use
Hartree and Hartree/Bohr. Stage/transfer diagnostics include complete setup,
source, projection and target timing in seconds; a faster target stage alone
is not an endpoint speedup. Final verification and any supplied accuracy
assessment remain distinct from iteration convergence.

(python-progressive-errors)=
## Progressive fallback and final acceptance

Changed calculator state after planning is rejected. Failed/rejected source
seeds may use the explicitly documented cold target guess; target native
warm-start safeguards retain their bounded cold retry. Neither API substitutes
a successful coarse source for the requested final target model/arithmetic.
`projected_singlepoint` executes the target with `strict=True`: a failed target
item raises rather than returning a result. `run_progressive_hf` executes the
target with `strict=False`: an item-level failure is returned in `target`, with
unsuccessful final verification and `target_density=None`. Its verification
status can be `unmet` or `budget_exhausted` as indicated by the recorded reasons.
Whole-call preparation,
allocation or execution exceptions still propagate in both APIs; a returned
failed item is distinct from those exceptions. Numerical final-arithmetic
requirements do not imply strict exception mode.

An unverified physical-residual/accuracy audit stays unverified or failed
according to the final-verification record; it is not manufactured from timing
or convergence. Inspect target status and verification fields before calling a
progressive result accepted.

(python-progressive-ownership)=
## Progressive lifetime

These synchronous orchestrators use separate context-managed source and target
prepared batches, close them before return and detach any reported target
density. A failed `run_progressive_hf` target has no retained target density. They borrow the supplied calculators and do not close or reconfigure
them. Keep calculators and policy inputs unchanged throughout an invocation;
do not share prepared/estimator state with concurrent workflows.

(python-progressive-backends)=
## Progressive support boundary

The deterministic controller targets its explicitly checked HF model domains
and pre-existing provider alternatives. It does not turn correlated/DFT model
identity support into progressive execution support, invent an approximation,
or enable an unavailable backend. Both source and target must independently
pass ordinary calculator and projection admission. Resource budgets and final
strict arithmetic are retained across the orchestration.

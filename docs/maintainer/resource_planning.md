# Resource planning and dry runs

`ResourceBudget` constrains the declared host/device allocation scope of a
complete prepared calculation. Providers offer finite, executable alternatives;
`plan_resources` chooses a deterministic combination that fits every cap.
Estimation builds basis/topology metadata without SCF iteration or production
integral, density, derivative or tensor arrays.

```python
from generativeqc import Calculator, ResourceBudget, estimate_hf_resources

h2 = [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]  # coordinates in bohr
budget = ResourceBudget(host_bytes=128 << 20, headroom_fraction=0.1)
plan = estimate_hf_resources([h2], budget=budget)  # no native CPU library needed
print(plan.status, plan.peak_bytes, plan.diagnostic)

calculator = Calculator(resource_budget=budget)
with calculator.prepare_batch([h2]) as batch:
    result = batch.execute(strict=True)
    print(batch.resource_plan.to_dict())
    print(batch.resource_diagnostics)
```

`Calculator.estimate_resources(systems, charges=..., multiplicities=...)`
resolves that calculator's actual basis, auxiliary basis, spin and solver
controls, including automatic Hückel/MINAO preparation when admitted. The standalone
`estimate_hf_resources` estimator describes the Hcore target inventory; use
`initial_guess=None` when executing that exact inventory, or use the calculator
estimator to include its active preparation policy. `singlepoint` also accepts
the calculator's budget and attaches diagnostics to its result. Unsupported or
infeasible plans fail before native
context creation in these execution endpoints. Constructing a CUDA calculator
itself can select a profile and initialize the CUDA runtime; use the standalone
estimator or CLI for a dry run without those side effects.

```bash
generativeqc resources molecule.xyz --basis def2-svp --method rhf \
  --backend cpu --host-bytes 1073741824 --headroom-fraction 0.1
```

The Python estimator APIs return the composable `ResourcePlan` schema described
here. The Python-free [native CLI allocation inventory](../user/native_cli.md)
has a separate explicit scope and is not a `ResourcePlan` checkpoint. Native
estimation uses the linked library's scalar layout queries, not
`GENERATIVEQC_LIBRARY` or profile selection.

XYZ units default to angstrom; `--units bohr` is available. The native CLI emits
JSON and exits 2 for unsupported or infeasible plans. The Python CUDA estimator
uses scalar layout queries from the library selected by `GENERATIVEQC_LIBRARY`,
loaded without profile selection or a device query. Neither estimator chooses a
GPU by free memory.

CUDA resource identities retain the binary digest of an already-loaded native
image across frontend wrappers. A bounded process-local cache holds the actual
library owners, preventing loader-handle reuse from aliasing another image;
per-wrapper profile identity stays separate. New images and metadata-only
adapters without loader ownership are hashed independently. Replacing a library
path does not change the identity of an image already loaded from that path;
in-place modification of loaded library bytes is unsupported.

## Native composition authority

Hückel preparation planning queries the selected native library's
`generativeqc_resource_huckel_numeric_capacity_v1`. Compact atom and contracted
shell metadata uses the same occupied atomic-channel counts and checked numeric
capacity as execution admission, including Cartesian/spherical target dimensions
and empty occupied subspaces. The query uses one metadata counter per atom and
linear work in atoms plus shells; it creates no scientific arrays or GPU state.
The planner retains all cold seeds and the largest serialized preparation
workspace alongside the target inventory. This is a conservative numeric bound,
not a measurement of provider allocations or whole-process memory. A missing
symbol keeps old-library portable sizing; a discovered query's rejection is an
error and never triggers a portable retry.

With an already-loaded library exposing `generativeqc_resource_compose_v1`,
runtime `plan_resources(..., library=library)`, calculator estimation and
budgeted preparation use the shared C++ resource composition engine. The native
CLI uses that same owner for effective caps, peak accounting and supported
alternative selection; its `composition_inventory` metadata identifies the
engine separately from the CLI's endpoint-inventory scope.

The engine sweeps sparse inclusive-lifetime endpoints, accounts pageable plus
pinned host storage and aggregate plus individual device caps simultaneously,
and orders supported combinations by budget excess, total relative cost and
lexical selection. Headroom uses the exact IEEE-754 rational value with upward
rounding before fixed reserves. Cost and excess sums use wider integer
arithmetic than individual byte capacities, so ranking cannot wrap int64 or
uint64. Expired buffers retire before newly live buffers at an adjacent phase.

The finite search limit remains explicit: exceeding it returns `unsupported`,
not a false allocation failure. Every enumerated combination is checked,
including overflow in a later alternative after a feasible prefix. For `P`
combinations, at most `B` selected buffers and `S` memory spaces, native work is
`O(P * (B log B + S))`, plus descriptor validation. Metadata scratch is linear
in input buffers and spaces, not phase IDs or the Cartesian product. No
scientific buffers, calculation contexts, backend probes or GPU allocations
are created by the composition query.

Supplied plans are checked by the native owner before scientific provider
preparation, with their explicit selections held fixed. Validation does not
silently replace a user-selected alternative with a cheaper one. Native
authority also rechecks changed optional CPU force workspace before a
force replay, preserving all live owners and selections. The revised plan is
published only after successful execution; rejection leaves the energy plan
and warm state usable. Discovered native errors propagate without retrying
portable selection. Missing symbols
or libraries retain provider-free portable planning; this adapter never loads a
library itself. The compiler's separate portable planner remains available for
offline generation without importing the runtime.

`ResourcePlan` retains its portable data-only schema, independent derived-byte
validation and serialization. A plan is not a native allocation receipt or
execution lease. Method/provider inventories, allocation observation coverage,
live provider reservations and session/lease lifecycle remain separately scoped;
native composition alone does not establish complete endpoint/RSS accounting
or complete the resource-authority cutover.

`benchmarks/native_resource_composition.py` provides opt-in CPU endpoint timing:
calculator construction, budgeted preparation, cold/warm/moved execution and
prepared-owner close. Compare exact baseline/candidate Python trees in separate
processes against the same compiled library, with the same thread settings and
no competing benchmark load. The JSON retains every observation, energies,
forces, convergence residuals, iteration/Fock counts (including unknown counts),
iteration-branch histograms, source identity and the library checksum. A changed
numerical/work signature invalidates a pooled timing comparison. This scoped
procedure does not establish GPU performance or complete physical receipts.

## CUDA DF bucket capacity ownership

The native base DF policy exposes `df_budget_targets` for resource-independent
workload and resident targets. Their maximum bounds automatic base resolution.
`method_owned_df_resident_capacity` separately bounds the extra method-owned
resident promotion using the actual packed planner and its shared response floor.
Complete prepared-owner planning must include both bounds, explicit caps and
source/setup lifetimes; these queries alone do not reserve a fleet or admit
optional XC storage. They perform no CUDA probes or numerical work.

`prepared_fock_device_capacity` composes the native Direct allocator bound with
DF value/response policy and source metadata. A mixed owner's resident-promotion
ceiling includes its already retained Direct provider. Its separate setup bound
includes the initial auxiliary metric, which exists before DF value-plan
construction and is not limited by the later value/response cap. The native KS
fleet accepts this Fock setup phase alongside one-electron/ECP and quadrature
setup, retaining other fleet members and lazy force capacity during serialized
rebuilds. SCF/XC, initial guesses, opaque providers and independent consumers
still require their own bounds before a complete endpoint reservation is valid.

With `generativeqc_resource_hf_df_cuda_fleet_v1`, the Python CUDA HF DF planner
and native CLI share one C++ numeric composition owner. It groups expanded
topologies by orbital/electron/primitive compatibility, checks Cartesian,
auxiliary and atom dimensions, and composes the existing source-upload, DIIS
and tile-planner leaves. Python packs scalar metadata and renders labels;
it does not choose tile capacities or recompute the fleet's numeric bounds.

The owner preserves resident/recomputed alternatives, explicit packed and
packed-single selection, occupied/automatic exchange reservations, requested
or default DF sub-budgets, and RHF/UHF ragged fleets. Packed-single retains
the conservative two-factor shape-query bound rather than claiming the
executable single-factor allocation. All bucket caches remain simultaneously
retained; preparation/setup/force workspace is serialized, and an independent
singleton cold numerical retry receives its own reservation. The existing
explicit CPU numerical-recovery contract is unchanged.

The private ABI validates a complete inventory before either sizing or full
publication. Invalid schemas, topology, byte-domain overflow, overlapping
input/output/error buffers and insufficient sub-budgets leave capacity output
unchanged. It allocates linear topology/plan metadata, not scientific arrays,
and neither probes nor initializes CUDA. Only a missing fleet symbol enables
the isolated old-library Python compatibility composer; native refusal or
malformed output never retries Python capacity arithmetic. These declared
bounds do not establish complete allocator/provider/lease/RSS authority or
qualify a new scientific backend or property.

## Native device ledger binding

An already-loaded library exposing
`generativeqc_resource_ledger_create_from_buffers_v1` derives a prepared owner's
numeric ledger limit and visible device from its selected capacity records.
The same native sparse-lifetime accounting used for global composition sweeps
the full inventory and the subset excluding `runtime_allowance`. Allowances
are withheld independently; they cannot fund tracked numeric allocations.
Subtracting independently measured peaks is not equivalent when lifetimes differ.

The metadata-only query creates a handle compatible with existing ledger
binding, observation, retained-provider reservations, journals and destruction.
It performs no CUDA operation or device selection by availability. Missing or
multiple visible devices are unsupported for this single-device owner; ordinals
outside the native signed-device domain are rejected rather than narrowed.
Invalid/overflowing/aliased records leave both handle and receipt unchanged.

`device_ledger.capacity_admission` identifies this native owner and reports the
full declared device peak separately from `limit_bytes`, the explicit numeric
admission cap. Neither is a physical allocation/RSS measurement. The Python
adapter publishes the handle only after receipt validation, releases unpublished
metadata on validation failure, and never retries native rejection with portable
accounting. Older libraries retain explicit portable capacity derivation; an
allowance-only inventory has a zero numeric limit, not a fabricated allocation.
Retained native buffers/reservations continue to outlive observation scopes and
closed frontend handles until their actual owners release them.

### Native prepared admission lifetime

The public `generativeqc_device_resources_create` accepts the selected inventory
as typed `generativeqc_device_resource_buffer_v1` records with descriptor
`admission = GENERATIVEQC_DEVICE_RESOURCE_SELECTED_INVENTORY` (zero). A raw cap
requires `GENERATIVEQC_DEVICE_RESOURCE_NUMERIC_CAP` and exactly one positive
device row with zero phases and no runtime allowance, as the CLI run-only option
provides. A single selected row remains selected inventory: native methods may
not reinterpret it as spare capacity for optional storage. The private inventory
ABI retains selected-inventory behavior. Neither mode claims that a complete
global plan has been selected or admitted, or grants exclusive use of the handle.
The interface shares the native
lifetime sweep with the older private metadata interface; it does not perform a
second global selection or validate model identity. Host-only metadata creation
and snapshots do not initialize CUDA or certify scientific backend support.

Pass this handle to `generativeqc_calculation_prepare_with_device_resources` or
`generativeqc_batch_prepare_with_device_resources` (or the optional resource
argument of the C++ `Calculation`, `Batch` and `KsComposition::prepare` wrappers).
The native prepared owner binds admission during both construction and each
synchronous execution, including ragged fleets. It retains shared ownership
until its scientific buffers and providers are destroyed. Closing the frontend
metadata handle cannot detach admission or retained charges. The matching
`*_get_device_resources` query returns a separately destroyable metadata handle
to that same owner, not a new budget.

Same-owner nested operations borrow the active binding without resetting work
or allocation counters. A different already-bound owner or concurrent use of
the same owner is rejected, never silently substituted. Exceptions restore the
prior binding. Native operations serialize the bounded host workspace even when
no Python observation exists. When they own a sampling scope they retain its
host/arena samples in the public diagnostic; an outer observer is borrowed and
is not misreported as isolated native sampling. A zero sample count means that
sampled peak is unavailable, not proof of zero memory use. Requested-byte
counter overflow likewise makes that quantity unavailable without relaxing
admission. Numeric allocations and provider reservations remain distinct.
The snapshot echoes the admission mode and reports unused native owner promises
as `capacity_reserved_bytes`. These promises are excluded from allocated-byte
counts, requested bytes and `combined_peak_bytes`; current numeric bytes plus
provider reservations plus unused promises must fit `numeric_limit`.

For an explicit numeric cap, strict FP64 all-electron s/p PBE/PBE0 KS with
homogeneous full-range Direct or DF providers can reserve the native mandatory
fleet before constructing its first item. The bound includes serialized rebuild
excess and deferred consumers; each independently callable prepared-owner port
uses its owner's promise. Optional mapped AO storage, spatial grid copies and
point batches can spend only capacity left after those promises. The dense XC
fallback remains reserved even while optional storage is live. Shared resource
handles retain separate promises for separate prepared owners.

Individual legacy/named D4 corrections add their allocator-owned retained
device capacity to the same fleet promise. Their tables and force workspace
remain protected through warm replay, failed rebuild and later fleet preparation.
Selected external inventories retain the dense/single-tile path. Preliminary
guesses, range/nonlocal corrections, named D3 and composite gCP corrections,
automatic/generated-split programs,
explicit incremental J/K, application precision budgets and experimental PBE
point precision retain incumbent admission. A native bound that does not fit
also retains incumbent admission; this is a bounded fallback under the same
cap. It does not authorize an unbudgeted retry or expand supported properties.
In particular, public CUDA DF KS forces remain unsupported.

The Python adapter uses this public preparation and snapshot path whenever the
matching native preparation symbol exists, without frontend observation/binding
control. Only absence of that symbol selects the explicit old-library path;
partial ABIs, refused creation/preparation, or inconsistent diagnostics never
retry an unbudgeted path. CPU resource-aware preparation is unsupported rather
than silently running without the requested device owner.

This establishes native ownership of the admitted numeric capacity, not complete
provider-internal allocation coverage, global physical-memory authority, RSS
measurement or a new force/HVP capability. The separate global planner and
explicit allowances retain their existing scope.

## CUDA KS to force handoff

CUDA KS keeps its admitted optional ERI value cache and Fast anchor matrices
throughout SCF and after energy-only solves. Before a stationary force consumer
allocates, the live final-state owner compares its additional device envelope
with free memory on its bound device. Under pressure it drains the SCF producer
and retires the idle anchor lease, then rechecks free memory before retiring the
optional ERI lease and its graph borrowers if still needed. Resident density, stationary
weights, XC state, the exact final-state token and the last-good warm seed stay
valid. The force consumer still uses the prepared derivative provider.

Native stationary, Python stationary and composite force owners request this
handoff before allocation. Compatible replays exclude retained force arenas
from the requested headroom. Python force work reports `scf_force_memory` with
requested, free-before, free-after and reclaimed device bytes. Resource
accounting drops retired anchor state capacity and the retired provider's device
and host storage. Repeated handoffs cannot retire an allocation twice. Retired
anchors remain disabled on this owner: later warm/cold solves and error recovery
retain ordinary Fast without reacquiring optional memory ahead of the force owner.

This check is advisory: it does not reserve global free memory, relax an
explicit budget, evict other calculations, or promise that arbitrary mandatory
storage will fit. Allocation failures remain explicit. Energy-only calls do
not release caches for a hypothetical later force request.

## Accounting contract

Host limits cover pageable plus pinned bytes; the pinned cap applies in
addition. The device limit covers the aggregate of devices, and optional
`per_device_bytes=((0, limit), ...)` caps apply in addition. `None` is unlimited;
zero permits no allocation. Fixed reserves are withheld from their aggregate
cap, and fractional headroom is withheld from each specified cap using integer
arithmetic. A reserve without a corresponding aggregate cap cannot constrain
otherwise unlimited memory.

Each `ResourceEstimate` describes one owned capacity over an inclusive
`first_phase`/`last_phase` interval. Persistent state, caches, outputs, library
storage and runtime allowances contribute to resident bytes. Workspaces also
contribute to peak bytes. A shared workspace appears once under its owner.
Concurrent streams must declare overlapping intervals. Caller-retained outputs
beyond the declared interval require their own reservation.

Sizes/products are checked against a portable signed 64-bit limit; native
adapters additionally check `size_t`. Estimates and identities are versioned.
`ResourcePlan.to_dict()` / `from_dict()` serialize data only, verify checksums,
and recompute accounting on load. Execution adapters compare the complete
provider request with the actual scientific inputs and schedule; a checksum
alone does not authorize a foreign estimate.

The planner orders candidates by provider cost and stable names. It enumerates
up to 65,536 combinations by default; a larger search reports unsupported.
Infeasible diagnostics include the closest candidate, violated caps, per-space
minima and dominant allocations. Per-space minima can come from different
candidates and need not be simultaneously attainable.

## Current provider inventory

CPU and CUDA HF/KS plans with an already-loaded `generativeqc_resource_basis_shape_v1`
query obtain public/Cartesian AO, shell, primitive and maximum-angular counts
from native compact expanded-contraction rows. The AO rule is shared with
normalized native systems. Count overflow or native refusal never retries
Python arithmetic; failed queries leave native outputs unchanged. The query
does not read coordinates or coefficients, allocate scientific arrays, load a
provider, or qualify an otherwise unsupported method/basis. Missing-symbol and
library-free planning retain explicit portable shape arithmetic.
Python still owns basis expansion and mathematical provenance in this planner;
these counts are not complete allocation, provider, lease or RSS authority.

CPU RHF/UHF calculator plans with a discovered
`generativeqc_resource_hf_cpu_fleet_v1` query obtain their per-item phase bounds,
summed fleet retained capacity and maximum serialized workspace from the same
allocation-free C++ owners as the native CLI. Direct and density-fitted
inventories preserve Cartesian/public integral coexistence, forward-mode
derivatives, SCF/DIIS matrices and analytic DF response storage. A single fleet
query validates every item before publishing any capacities; invalid later
items, overlapping buffers and aggregate signed-int64 overflow leave output
unchanged. Native rejection propagates without a Python retry.

Schedule identity records phase and fleet sizing owners separately. Libraries
with only `generativeqc_resource_hf_cpu_host_v1` retain native phase sizing and
portable fleet composition; library-free dry runs and older libraries retain
portable metadata sizing without implicitly loading a provider. Declared-capacity
selection and admission use the separately discovered native composition owner
described above; prepared leases remain frontend-owned. Native fleet sizing is
not complete endpoint/RSS allocation accounting.

Direct CUDA RHF/UHF plans with a discovered
`generativeqc_resource_hf_cuda_fleet_v1` obtain host bounds, retained device
arenas and serialized retry composition from the same allocation-free C++ owner
as the native CLI. The topology-aware allocator v2 and ECP queries remain the
leaf authorities; composition never constructs a CUDA provider or context.
Ragged fleets sum retained host/device capacities and preserve the conservative
retry `max(arena) + max(ecp_workspace)`, including maxima from different items.
The entire fleet is validated before publication, and homogeneous CLI batches
apply multiplicity exactly once. Schedule identity names native fleet ownership;
only an absent symbol permits portable compatibility. Native refusal, overflow
or inconsistent publication never restarts Python arithmetic. This does not
change density-fitted HF, scientific admission, prepared lease ownership or the
exclusion of provider/allocator retention and complete simultaneously-live RSS.

CPU KS calculator plans with a discovered
`generativeqc_resource_ks_cpu_fleet_v1` query use the same allocation-free native
owners as the installed CLI for per-item diagnostics, twelve retained-owner
totals and serialized preparation/SCF workspace. The query validates the entire
fleet and the complete signed-int64 retained-plus-workspace peak before
publication. A caller-supplied workspace floor preserves the existing CPU force
consumer allowance without implying native admission of that consumer; its
eligibility and allowance policy remain frontend-owned. Restricted AO caches
remain transient, and nonlocal provider sub-budgets remain explicit.

Native rejection never retries Python sizing. Schedule identity records item
and fleet authority separately. Libraries exposing only
`generativeqc_resource_ks_cpu_host_v1` keep native per-item sizing and portable
fleet composition; old/no-library dry runs retain portable metadata sizing
without implicitly loading a provider. Neither sizing query executes
quadrature, integrals or SCF. Native admission/leases and full simultaneously-live
allocator/RSS receipts remain outside this bridge.

CUDA KS host inventories use the same `scf::KsHostFleetResources` owner through
the additive `generativeqc_resource_ks_cuda_host_fleet_v1` query. An item adds
the XC host-unfused selector to the CPU query's seventeen scalar shape fields.
The query shares the transaction, overflow and no-provider-discovery contract,
and is available in CPU-only libraries. CUDA host bounds include independent
solver, matrix and structured binding reservations; host-unfused XC matrices
are retained, while setup and host XC tiles are serialized workspace. CUDA
shapes do not inherit CPU integral fourth powers or through-g admission.

Python supplies its snapshot-grid and force-consumer retained allowances once
per fleet; they coexist with every preparation and replay. The energy-only
native CLI supplies neither allowance. These are lifetime inputs, not new
force eligibility or admission. Missing symbols retain portable host sizing;
selected native refusal or inconsistent publication cannot retry that path.
Schedule identity records CUDA host fleet authority separately from device
inventory and ECP queries. Device fleet composition and complete
provider/allocation/lease authority are outside this host bridge.

HF/KS ECP workspace sizing uses the already loaded library's
`generativeqc_resource_ecp_workspace_v1` query, shared with the native CLI.
This allocation-free scalar owner uses generated two-grid policy and CUDA
radial scheduling without constructing grids, scientific tensors or a CUDA
context. Public AO counts select the radial batch; Cartesian counts determine
storage. A discovered query's rejection propagates without portable retry.
Old libraries and library-free dry runs retain explicit metadata-only sizing;
all-electron inputs need no ECP query or policy import. ECP sizing provenance is
separate from the enclosing HF/KS inventory in the schedule identity. KS records
host and device ECP owners separately because old-library and portable host
paths need not discover the same ECP query as device layout planning. This changes
sizing ownership,
not scientific qualification, native admission/leases or allocation receipts.

D3(BJ) planning uses the already loaded library's private
`generativeqc_resource_d3_layout_v1` query. This allocation-free shape query and
`D3Plan::prepare()` share the same C++ retained plan/execution, workspace and
actual compiled table-layout capacities. CPU uses the largest serialized
system's workspace; CUDA uses aggregate ragged atom storage. A CUDA selector
does not initialize CUDA and is available in CPU-only libraries. Native shape
validation preserves the uint32 ragged-offset and per-system atom bounds.
The request records native versus portable sizing in its schedule identity;
old libraries and library-free dry runs retain explicit portable metadata
sizing, never used after native rejection. Python presents the estimates and
checks local budget eligibility; global declared-capacity selection uses the
separately discovered native composition owner. The prepared native D3 owner
continues to enforce its local retained-byte cap. These numeric capacities are
not process RSS receipts or new CUDA scientific qualification.

Named D4(BJ)-EEQ global planning requires the native
`generativeqc_ks_d4_resource_query_v1` protocol and an admitted complete KS program.
It sums the actual per-system correction owners, including replicated tables,
and uses the largest individual retained capacity times the fleet size for the
existing equal-share dispersion cap. The standalone D4 worker schedule and this
query share one native sizing authority. CPU plans also reserve the native-sized
standalone correction-gradient owner for an electronic force compatibility path;
CUDA plans require the existing native stationary-capacity context. Selected
native failures propagate. Missing inventory support leaves ordinary D4 available
but rejects global D4 plans. Composite D4/gCP and electronic domains already
excluded by KS planning remain unsupported.

For bounded strict-FP64 all-electron s/p/d Direct PBE/PBE0-D4 CUDA contexts, the additive
`generativeqc_resource_ks_d4_optional_cuda_v1` query supplies an optional XC
headroom candidate. It reserves mapped AO arenas, spatial copies and the native
default point-panel ceiling concurrently. This is a conservative declared
capacity, not a measured allocation peak. The selected complete KS+D4 owner
receives a numeric cap containing only its own estimates, including provider
allowances that native preparation converts into retained leases. It cannot
spend unrelated global-plan capacity. Original selected-inventory candidates
remain available when the extra headroom does not fit or the protocol is absent.

| Provider | Retained storage | Temporary peak and execution schedule |
| --- | --- | --- |
| CPU direct RHF/UHF | All fleet topology, warm densities and outputs | Largest serialized item: forward-mode Jet/recurrence arrays, Cartesian/public integral coexistence, SCF/DIIS/eigensolver matrices and forces |
| CPU density fitting | Fleet state as above | Also includes the current implementation's four-center ERIs/derivatives, raw metric/three-center derivatives, transformed factors and analytic metric response |
| Small CUDA direct RHF/UHF | Sum of every retained bucket arena | Exact production arena layout, plus a single-item arena for existing cold numerical recovery alongside warm caches; conservative host topology/staging bound |
| Small CUDA density fitting | Sum of all bucket J/K caches and solver workspace allowances | Maximum serialized setup/force workspace, source metadata uploads and a cold numerical recovery reservation; native tile planner supplies actual tile dimensions |
| CPU LDA/PBE RKS/UKS | All grids, packed bases, conventional integral providers, warm densities and diagnostic histories | Serialized value-only integral preparation and SCF/XC/DIIS workspace |
| CUDA LDA/PBE RKS/UKS | Sum of every native stream's exact state/XC/direct-J arenas, host grids/bases and histories | Serialized host/one-electron setup; the replaced item's old owner retires before the new owner allocates |
| TensorIR CUDA | Native arena, input/output storage, host staging and library allowance | Existing TensorIR liveness, tiling and recomputation alternatives; full inputs/outputs remain resident |

Budgeted CPU execution explicitly uses one native worker, including multi-item
buckets. Workspace therefore scales with the largest item and retained state
with the fleet. HF CUDA buckets execute serially but retain all their caches.
KS CUDA batches enqueue all active streams before collecting scalar results;
their persistent states and workspaces are all included in the same plan.
Geometry updates preserve resource topology identity when dimensions remain
compatible; scientific warm-state and integral-cache invalidation still apply.

HF CUDA direct inventory v1 supports at most 16 public AOs and DIIS history 64.
CUDA DF additionally limits auxiliary AOs to 128. Larger shapes, missing native
inventory queries and optional profiling/graph-eigensolver overrides report
unsupported. Explicit auxiliary templates must be compatible across a fleet,
matching the existing native batch contract. Response and correlated providers
can implement `ResourceRequest`; these methods do not yet have execution
adapters in this inventory.

### Semilocal KS calculations

`estimate_ks_resources(systems, method="pbe-rks", ...)` resolves the same
LDA/PBE RKS/UKS controls as `Calculator.estimate_resources`. The CLI accepts
these four method names. CPU estimation requires no native library; CUDA
estimation uses allocation-owned shape queries without constructing a context,
quadrature, AO matrix, integral or eigensolver. Native KS inventory v1 supports
the ordinary-stream solver above 16 AOs and up to 64 DIIS entries, subject to
checked native launch dimensions. It declares the current immutable GridSpec-v1
(48 radial, 16 polar, 32 azimuth by default), XC tile, functional/domain policy,
basis/charge/spin, numerical controls, backend and CUDA binary identity.
`ks_options=KsOptions(...)` uses the selected grid/radii/tile in both the
estimator and native preparation; see [KS options](../user/ks_options.md).
Geometry changes preserve the resource shape; model changes require a new plan.

Property coverage differs by entry point, and the difference is deliberate:

- `Calculator.estimate_resources(..., properties=(...))` plans the properties
  that calculator context actually advertises. A
  [direct all-electron CPU semilocal force](../user/dft_density_fitting.md)
  context that requests `"forces"` reserves the transient CPU force workspace
  floor. The retained CUDA force owner is a separate CUDA-backend reservation;
  it is not part of the CPU plan.
- The Python-free `generativeqc resources` dry run still reports unsupported for
  `--forces` and for density-fitted KS, because its scalar v1 layout does not
  enumerate the prepared force or fitted-provider storage. It fails closed
  rather than presenting an energy envelope as a force plan.
- Automatic (Libxc-inventoried) CUDA KS is energy-only in both entry points;
  `include_forces` is rejected before planning.

Incremental Direct-J/K is outside this inventory: its anchor/delta buffers and
closure history have no qualified scalar v1 layout. The native `resources`
command validates both `GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK` and the legacy
`GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK` for these four KS methods. Unset, `0`
and `off` preserve ordinary plans; `1` or `on` in either selector returns
`unsupported` JSON with exit status 2 before basis expansion or layout queries.
Other values are usage errors (exit 2 with no JSON), even when the other selector
is enabled. Tuning suffixes are not consulted, and HF inventory is unaffected.

The host inventory includes 36 bytes per quadrature point, packed AO metadata,
all prepared owners and warm snapshots, iteration records, and conservative
value-only Cartesian/Jet/recurrence, eigensolver, DIIS and XC workspace bounds.
CUDA state/XC and direct-J device capacities come from the allocator's
own checked layouts. One-electron device setup is conservatively bounded and
charged only for its excess over the owner it replaces. KS passes the exact
direct-J source size to that provider rather than inheriting its standalone
default cap.
The public method has one global budget and no separate XC or J user budget.

KS binds its ledger during **preparation and execution**, retaining separate
`preparation` and `observation` evidence. All scientific device allocations
are charged before returning a prepared object. Replays retain charges and
geometry rebuilds release old buffers first. Preparation/execute failures keep
their available evidence. CPU observations remain partial capacity samples;
they include retained fleet/grid/basis/warm buffers but do not measure every
setup, recurrence, XC-tile or eigensolver temporary. They are never reported
as a complete allocator peak. Python objects, allocator bookkeeping, CUDA
driver/modules/stacks and runtime retention are explicitly excluded.

CUDA DF offers resident and source-regeneration choices. A positive explicit
DF sub-budget retains the native source mode; otherwise the planner may choose
either mode. The source candidate regenerates response derivatives on CUDA;
its forward tensor is retained when the native tile allowance fits. Both
candidates use **CUDA SCF with existing CPU numerical recovery**. These
decisions appear in the plan. Allocation failures cannot trigger that host
recovery. The default source sub-budget includes the native one-electron
preparation minimum, including Cartesian/public copies and earlier prepared
items. The request admits both energy and forces: its capacity envelope uses
the larger energy value allowance and also checks the force partition. Bucket
inventories expose `energy_tiles` and `force_tiles`; the historical `tiles`
field retains the force route. The selected one-electron derivative provider
controls host derivative storage. v1 does not enumerate arbitrarily small DF
tiles or expand its supported orbital/auxiliary shape range.

CUDA DF numeric solver workspaces have an explicit conservative 64 MiB
allowance plus shape terms. Opaque library retention has a separate 256 MiB
allowance per bucket and one recovery owner. These are capacity policies, not
measured library usage or minimum physical GPU requirements.

## Composing HF with TensorIR

`tensor_resource_choices` in `generativeqc_compiler.tensor.resources` adapts the existing
TensorIR planner. Its `request` joins an HF request in the same global plan;
`choices.selected(plan)` returns the corresponding exact TensorIR plan.

```python
from generativeqc import ResourceSession, plan_resources
from generativeqc_compiler.tensor.resources import tensor_resource_choices

# program, compiler and cache are ordinary TensorIR objects/paths.
budget = ResourceBudget(host_bytes=1 << 30, device_bytes=1 << 30)
calculator = Calculator(device="cuda", resource_budget=budget)
hf = calculator.estimate_resources([h2]).requests[0]
tensor = tensor_resource_choices(program, compiler.target)
plan = plan_resources([hf, tensor.request], budget).require_feasible()
with ResourceSession(
    plan,
    {
        "hf": lambda selected: calculator.prepare_batch([h2], resource_plan=selected),
        "tensor": tensor.factory(compiler, cache),
    },
) as session:
    session.advance(0)
    hf_result = session.provider("hf").execute(strict=True)
    tensor_result = session.provider("tensor").execute(feeds)
```

Provider input controls must match the request; an explicit
global plan must also match any budget already configured on the calculator.

`ResourceSession.advance(phase)` closes expired owners before preparing newly
live owners. Each session owner must use one uniform interval across all its
buffers/candidates. Richer internal phase schedules remain available in the
planner and are implemented by the provider or separate owners.

## Allocation failures and observations

A factory can raise `ResourceAllocationError` with an allocator-identified
space after releasing partial allocations. The session closes its newly
created group, selects an untried enumerated alternative that reduces active
usage in that space, and records the failure/selection. It preserves already
live owners. Each selection is attempted once per phase, including alternating
host/device failures. Numerical errors are never allocation retries.

TensorIR allocates at this preparation boundary and supports automatic retries.
HF currently allocates native scientific buffers during `execute`, after its
Python prepared object exists. An HF execution failure preserves diagnostics
and returns/raises the failure; it does **not** roll back scientific state and
automatically replay the solve with a new plan. Callers can inspect enumerated
alternatives and explicitly prepare a new calculation. This boundary prevents
replaying successful neighbors or silently changing a partially executed fleet.

The native device ledger charges actual owned CUDA buffer capacities, enforces
the selected HF or KS owner's numeric capacity, retains charges across warm calls,
and releases charges with native cache destruction. A shared global budget
also reserves TensorIR's separate allocations. The ledger excludes CUDA
context/modules, graphs, stacks, pool/page retention and library internals.
Those exclusions and any allowances are explicit in the plan. It is not a
process-wide physical-memory cap.

Native D4 correction owners also charge their retained CUDA buffers to an
active device ledger. Partial construction and host exceptions release those
charges; a surviving correction owner keeps its ledger alive after the submitting
scope ends, including cross-thread destruction. Without an active ledger,
standalone D4 retains its existing `maximum_bytes` worker-schedule bound. This
allocation contract alone does not add D4 to global `ResourcePlan` admission.

CPU diagnostics sample simultaneously owned integral/SCF/DIIS vector capacities
at iteration boundaries. They **exclude transient recurrence/eigensolver
allocations** and are not a complete allocator peak. Host execution is bounded
by the inventory/schedule, rather than a host allocator limiter. The standalone
CPU heap audit measures C++ allocations throughout preparation and execution
to check those broader bounds. CUDA TensorIR reports owned ndarray bytes,
native buffers and measured retained provider storage. Unknown CUDA OOM
placement stays unknown; only actual ledger evidence identifies device OOM.

Reproduction commands and production receipts are in
[`benchmarks/resource-planning`](../../benchmarks/resource-planning/README.md).

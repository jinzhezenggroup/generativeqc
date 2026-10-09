# Python calculation and result contracts

These contracts accompany the signatures and member documentation in the
[Python API](api.md). Each operation's docstring supplies its narrower options;
shared conventions here do not enlarge the admitted scientific domain.
[Capability interpretation](capabilities.md) distinguishes selector discovery,
contextual public support and compiler representation.

(python-calculation-values)=
## Calculation inputs, outputs and units

`Calculator.singlepoint` accepts a nonempty iterable of `Atom` records or
`(element, position)` pairs. Elements are symbols or atomic numbers; positions
have exactly three finite real coordinates in Bohr. Charge is the integer ionic
charge and multiplicity is the positive integer `2S + 1`; electron count and
spin must be consistent with the method. Boolean values are not integer nuclear
identities. A basis is a supported selector, local canonical JSON path,
`BasisSet`, or explicit `Shell` sequence. A shell's atom index is zero-based;
its angular momentum and primitive exponents/coefficients describe the actual
basis. Basis-free methods reject an unrelated Gaussian basis.

Energy is a scalar in Hartree. Requested forces have shape `(natoms, 3)` in
input atom order, in Hartree/Bohr, with `F = -dE/dR`. Energy-only execution
returns `forces=None`. `properties` is an iterable of names containing
`"energy"`, optionally `"forces"`; a bare string is invalid. The defaults are
context-dependent and documented by `singlepoint`; explicitly choose the
properties your application requires. SCF `energy_tolerance` is in Hartree;
`density_tolerance` concerns the density iteration metric, not an energy-error
bound. Correlated energy tolerances and denominator thresholds use Hartree;
residual tolerances concern the named residual equations. Memory options ending
in `_bytes` are integer byte capacities, never atom counts.

`hessian` returns a detached record with a raw Cartesian `(3*natoms, 3*natoms)` matrix in
Hartree/Bohr². `hessian_vector_product` accepts a finite real direction of shape
`(natoms, 3)` and returns a detached record whose `value` has shape
`(natoms, 3)`, containing that matrix applied to the direction; no rotational/translational projection or post-hoc symmetrization is
implied. Integral workspace and full-Hessian publication have separate byte
budgets. See the [energy/force/Hessian conventions](../learn/energy-force-hessian.md).

(python-calculation-errors)=
## Calculation failure and publication

Python validation raises `TypeError` for malformed types and `ValueError` for
invalid geometry, electron state, controls or requested properties. Unsupported
execution combinations raise `NotImplementedError` or the specific validation
exception reported by their admission boundary. Loading a missing or
incompatible native library can raise `OSError` or `RuntimeError`.
Native non-success statuses, including nonconvergence, become `RuntimeError`;
native not-implemented becomes `NotImplementedError`. Resource-aware execution
can raise `MemoryError` with attached resource diagnostics. A failed
`singlepoint` call does not return a partially valid `Result`. In particular,
native diagnostic writes on nonconvergence are not a successful Python result.
Failures while obtaining mandatory diagnostics also abort publication.

A force request can fail after electronic energy converges. The composed
single-system path raises rather than returning an energy-only success for a
failed force request. Hessian/response admission, convergence and byte-budget
failures likewise do not publish a partial Cartesian derivative as success.

(python-calculation-ownership)=
## Calculation lifetime and concurrency

Calls are synchronous at the Python boundary: returned numerical arrays are
ready to consume. Single-point forces are copied out of native output storage;
temporary systems and calculation handles are released before return. Returned
records remain usable after the call. Frozen dataclasses do not imply that all
contained NumPy arrays or dictionaries are recursively immutable; copy them
before changing data that another consumer shares.

A calculator may retain native workspace between calls. `clear_cache()` waits
for its retained single-point transaction owner and releases that workspace;
later calls can rebuild it. Prepared batches have separate owners and are not
closed by this operation. Where a retained single-point context exists, its
lock serializes single-point execution and cache clearing. This does not grant
a general guarantee for concurrent changes to calculator configuration or for
sharing prepared batches. Keep configuration fixed during use and use separate
calculators/plans for independent concurrent operations.

(python-calculation-backends)=
## Calculation admission and precision

The [generated method catalog](../public_methods.md) discovers selectors.
`Calculator.capabilities` is the contextual property boundary after method,
backend, basis, spin, precision, density fitting, ECP and grid choices.
`method_capabilities(name)` is a backend-neutral registry query; it loads the
native registry and is not a probe proving a requested device/context works.
Neither record bypasses per-system or convergence checks. `second_order_capabilities`
is distinct from first-order properties. Public `hessian`/HVP admission is the
CPU, direct, all-electron, Cartesian, strict-FP64, closed-shell LDA/PBE RKS domain
currently checked by those methods; a low-level CUDA response kernel does not
expand it.

A precision option selects an admitted numerical policy. It does not certify
observable accuracy or silently enable an unsupported method/derivative.
`TargetAccuracy` separates a requested observable tolerance from iteration
convergence; without suitable evidence, a completed calculation can remain
`unverified`. A resource estimate does not guarantee that the operating system,
driver or other processes can provide memory at execution time. Refer to the
[resource planning contract](../maintainer/resource_planning.md) for allocation
scope and exclusions.

(python-batch-values)=
## Ragged batches and prepared topology

`prepare_batch` and `batch_singlepoint` take a nonempty sequence of nonempty
systems in the single-point atom format. `charges` and `multiplicities` have one
entry per system; omitted values mean zero and one respectively. A prepared
batch fixes atom ordering/counts, charge, multiplicity and scientific model.
`execute(coordinates)` accepts one real `(natoms_i, 3)` array per prepared item,
or `None` entries to retain stored coordinates. The outer `None` keeps all
stored coordinates. Coordinates use Bohr. Shape/layout errors detected by
Python fail before native replay; native atom-count/nonfinite-coordinate
admission can produce an individual item failure.

Outputs retain input order and each force array keeps its own `(natoms_i, 3)`
shape. No largest-system padding is introduced. `BatchResult.energies` is a
fresh float64 `(nsystems,)` array with NaN in failed slots. An individual failed
record may still contain native scalar diagnostics: inspect `succeeded` before
consuming its energy. Forces are `None` for failed or energy-only items.

(python-batch-errors)=
## Batch failures and strict mode

Malformed outer lengths, property selections, closed handles and changed
prepared model identities raise before a result is published. Whole-call
native/query failures raise just as in the single-point boundary. With
`strict=False`, item failures are returned alongside successful neighbors;
`status`, `status_message`, `failure_indices` and `succeeded` describe the result.
A generated force failure preserves its exception type and detail in the
item's status message. `strict=True` calls `raise_for_failures()` after replay
and raises `RuntimeError` if any item failed, so that call returns no
`BatchResult`. Strictness does not make the replay transactional: successful
neighbors may already have updated warm-start state.

Checkpoint load validates corruption before applying any seed. Exact restart
is the default; `allow_warm` and non-strict compatibility behavior are described
by `load_checkpoint`. Loaded/projected densities are proposals, not a claim of
convergence. `PreparedBatch.initialize_from` only installs compatible projected
seed densities; the caller must subsequently call `execute` to solve the target.
`projected_singlepoint` performs that target solve itself with `strict=True`,
using its explicit cold-start fallback on incompatible source seeds.

(python-batch-ownership)=
## Batch ownership and concurrency

`PreparedBatch` retains its calculator and independent native context, plan and
warm-state owners. Preparation copies native system data; callers do not retain
temporary native system handles. Result arrays/records outlive the plan.
Use `with calculator.prepare_batch(...) as batch` or call `close()` to release
resources promptly. Repeated close is safe; execution and native diagnostic
queries after close raise. Finalization is best-effort cleanup, not a substitute
for deterministic close.

A prepared batch is not concurrently re-entrant. Execution, checkpointing,
projection, warm-start controls, diagnostics and close must not race on one
plan. Use separate prepared plans for concurrent callers. Successful executions
can update warm densities. `set_warm_start_updates(False)` freezes existing
seeds; it does not make the owner thread-safe or disable warm starts.

(python-batch-backends)=
## Batch capabilities and profiling

Preparation requires contextual `supports_batch`. Property selection uses the
calculator's contextual support and rechecks scientific/derivative resource
admission when needed; changing requested outputs does not change topology or
scientific identity. Repeated force calls may rebuild response caches.
A caller-supplied resource plan must match the prepared systems/model/budget.
Shell-class and inactive-eigensolver profiling are opt-in CUDA diagnostics;
requesting an unenabled profile raises. Their collection can copy data and
synchronize, so these queries belong outside normal endpoint timing. Prepared-owner profiles may remain from an earlier successful replay when a
later call is rejected before that owner runs. Transport counters can be
cumulative. Interpret each record's status, scope and evidence/completeness
fields; query availability alone does not establish fresh execution or capability.

(python-results-values)=
## Results, diagnostic records and units

`Result` and `BatchItemResult` distinguish iteration convergence from physical
stationarity and observable accuracy. `energy_change` is in Hartree;
`density_rms` is a density-update metric; `physical_residual_rms`, when present,
is the physical commutator residual. Absence (`None`) means that diagnostic was
not supplied, not zero error. `CorrelationResult` energy components and
minimum absolute denominator are in Hartree; count fields are counts, byte
fields capacities or transfer volumes as named, and `_ms` timing fields are
milliseconds. Execution backend, precision and equation/operator identities
describe the selected/executed path on a successful call. Failed batch items
can instead expose prior owner diagnostics or conservative/default/empty
precision, incremental or work records. For example, an MP2/UMP2 nonconvergence
exception can precede completed result assignment while diagnostic queries still
succeed. KS/correlation records from a completed solve can also survive a later
output-buffer rejection. Item status and record evidence/completeness govern
interpretation: availability alone is not proof of a completed solve, fresh
measurement or numerical acceptance.

KS energy-component records use Hartree, iteration records retain their named
SCF residuals, and transport records describe the bound XC execution path.
Batch shell/eigensolver/DF metric records are profile or conditioning evidence,
not an extra solve. `_bytes` means bytes; `_seconds` means seconds; counts are
integer semantic events. Source field annotations and each record's docstring
are authoritative for optionality and ordering.

(python-results-ownership)=
## Result ownership

Result records expose Python scalars, tuples, NumPy arrays and diagnostic
mappings. No public result is a permission to dereference a native handle after
its owner closes. Arrays copied/published by calculation and batch endpoints
remain valid after close. Dataclass freezing only prevents rebinding fields;
treat nested arrays and mappings as shared read-only data unless an individual
operation explicitly returns an immutable snapshot or a detached copy.
Serializing/inspecting a record is not synchronization for a concurrently
executing owner. Check status before consuming batch or correction data.

(python-torch-values)=
## Torch shapes, units and derivatives

`energy` accepts a real floating-point Torch tensor of shape `(natoms, 3)` in
Bohr and matching exact integer atomic numbers. It returns a scalar tensor in
Hartree with the coordinate tensor's dtype/device. `batched_energy` accepts a
nonempty sequence of ragged `(natoms_i, 3)` tensors with a common dtype/device,
matching atomic-number sequences and per-system charges/multiplicities. It
returns `(nsystems,)` energies. Defaults are neutral singlets and a CPU RHF
STO-3G calculator if none is supplied.

Backward returns `dE/dR = -forces`, in Hartree/Bohr, multiplied by the upstream
energy cotangent. Gradients are only with respect to coordinates. Atomic
numbers, charge, multiplicity, calculator options and prepared topology are
not differentiable inputs. Only first-order backward is supported;
differentiable backward (`create_graph=True`), Hessians and HVPs are rejected.

(python-torch-errors)=
## Torch failure behavior

Invalid tensor types/dimensions, mismatched identities, inconsistent batch
dtype/device or noninteger identity values raise `TypeError`/`ValueError`.
Calculator admission/execution exceptions propagate. Batched forward always
uses strict batch execution, so an item failure raises and no energy tensor is
published. Higher-order differentiation raises `RuntimeError` rather than
silently producing zero force derivatives. A supplied prepared batch must have
exactly the provided atom numbers, charges and multiplicities.

(python-torch-ownership)=
## Torch staging and lifetime

The wrapper explicitly detaches coordinates and stages them on the host;
batched forward converts them to CPU float64 before NumPy conversion, including
Torch dtypes without a NumPy representation. Single forward transfers non-CPU
coordinates to CPU before constructing atom values. This is not a zero-copy
CUDA-coordinate ABI. Energies and saved force tensors are then returned/copied
to the input tensor's dtype/device. Casting to a lower-precision Torch dtype
does not change the native method's numerical policy.

Saved detached force tensors belong to the autograd context and support its
first-order backward after forward returns. The wrapper does not differentiate
through native SCF iterations or force evaluation. Do not mutate coordinate
inputs while forward runs. Caller-owned prepared batches remain open and keep
their warm state; the wrapper does not take ownership of closing them. Their
non-reentrancy also applies to concurrent Torch forwards.

(python-torch-backends)=
## Torch backend boundary

PyTorch is optional and imported only when this module is selected. The tensor's
device determines result placement, not the native calculator backend: a CUDA
input tensor can use a CPU calculator after host staging. A supplied calculator
must admit both energy and analytic forces for the requested context. Single
forward requests both explicitly; batched forward uses the prepared/calculator
default property set, which must therefore already include forces. Energy-only
default contexts are not supported by this batched wrapper, even when an explicit
force request would be qualified. The wrapper does not add force capability. A supplied prepared batch owns execution
and topology admission. Higher-order AD is unsupported regardless of backend.

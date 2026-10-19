# Native C and C++ API

The stable native declarations live in
[`generativeqc.h`](https://github.com/jinzhezenggroup/generativeqc/blob/master/include/generativeqc/generativeqc.h).
The move-only C++ convenience wrappers are in
[`generativeqc.hpp`](https://github.com/jinzhezenggroup/generativeqc/blob/master/include/generativeqc/generativeqc.hpp).
The explicit Kohn-Sham composition builder is in
[`ks.hpp`](https://github.com/jinzhezenggroup/generativeqc/blob/master/include/generativeqc/ks.hpp).
Internal `src/` headers are not public API declarations.

The [complete symbol reference](native_symbols.md) contains stable targets and
source-owned contracts for every exported C function, every explicitly declared
public C++ operation, and all public C++ owner/record types. Record fields
are described with their owning type. Generated method identifiers are covered
by the [capability authorities](capabilities.md), rather than a second copied
support matrix.

```{toctree}
:maxdepth: 1

native_symbols
```

## ABI and capabilities

Call `generativeqc_get_abi_version()` to discover the loaded ABI.
Zero-initialize public versioned descriptors, then set
`struct_size = sizeof(the complete descriptor type)` and
`abi_version = GENERATIVEQC_ABI_VERSION` before assigning options. Initialize
nested versioned option descriptors and every caller-supplied versioned history
row too. Most admission checks accept at least the current complete size; the
ordinary system constructor additionally accepts its historical prefix before
`basis_representation` and defaults that prefix to Cartesian AOs. ECP system
creation requires the complete system descriptor. Unversioned profile arrays
have no size/ABI fields. Consult each symbol rather than assuming universal
exact-size or prefix compatibility.

`generativeqc_method_available` and `generativeqc_method_get_capabilities`
expose registry information, not support for every basis/device/precision and
derivative combination. A prepared execution can apply stricter checks.
See [capability sources](capabilities.md).

## Core C handles and ownership

| Entry points | Contract |
| --- | --- |
| `generativeqc_context_create/destroy` | Create the execution context and release it after dependent owners |
| `generativeqc_system_create/destroy` | Create a system with atom, geometry and basis descriptors |
| `generativeqc_calculation_prepare/destroy` | Prepare and dispose of single-system method state |
| `generativeqc_calculation_execute` | Execute synchronously; callers provide the output descriptor and force buffer when requested |
| `generativeqc_batch_prepare/destroy` | Own and release a persistent ragged-fleet plan |
| `generativeqc_batch_execute` | Execute the fleet with per-input result statuses rather than assuming every member succeeded |
| `generativeqc_context_get_last_detail` | Borrow context detail; copy it before another context operation because D4/nonlocal calls may clear it even on success |

For bundled Gaussian bases, `generativeqc_system_create_bundled` accepts an
atom-only full-size system descriptor, a bundled name and an optional expected
catalog checksum. `generativeqc_bundled_basis_catalog_sha256` identifies the
compiled source pack; supplying it as the expected checksum binds external
metadata to that exact catalog. The C++ `System(context, atoms, basis_name, ...)`
constructor uses the same expansion and existing normalization/validation owner.
These paths read no basis files, import no Python and invoke no compiler at
runtime. They do not broaden any method/backend capability.

For caller-supplied FP64 data, `generativeqc_system_create_basis_template_v1`
accepts the same atom-only system descriptor and an element-local template
descriptor. Shared exponents and row-major general contractions are expanded in
atom, shell and contraction-row order, retaining zero coefficients. Scalar ECP
terms carry their nonnegative source angular channels; the greatest channel of
each element becomes the local potential. The compiled catalog, canonical-file
CLI adapter and Python `BasisSet` adapter share this expansion owner. Existing
descriptor constructors retain normalization and scientific/backend admission.
The C++ `System(context, atoms, template_descriptor, ...)` overload provides the
same copying, synchronous lifetime contract. Array ranges and expanded uint32
capacities are checked before expansion; no system is published on failure.

Template arrays are numerical inputs, not authenticated canonical JSON or a
native model/provenance identity. Format/schema and provenance adapters remain
outside this constructor. The template route preserves the existing bounded
scalar ECP scope (orbital s/p/d/f, local channel through g, radial powers 0–4);
method/backend restrictions still apply. Explicit atom-expanded descriptors
remain supported for existing clients.

Keep the execution context valid for dependent prepared owners, and release each
handle with its matching `*_destroy` function. Systems own copied atom/basis
data. Calculations and batches copy those systems during successful preparation,
so input system handles and descriptor arrays need not survive their consumers.
A borrowed native handle never transfers ownership. Destroy functions accept
NULL; double destruction or use-after-destruction is invalid. Never assume a function
allocates an output buffer merely because it accepts a pointer: consult its
buffer-size and null-pointer conventions. Unless a function explicitly
documents transactional behavior, do not rely on output contents after failure.

The high-level `generativeqc::Context`, `System`, `Calculation`, and
`Batch` C++ wrappers own the corresponding C handles using RAII, are movable
but not copyable; prepared execution owners require their Context to remain alive.
They support move construction, not move assignment; only destruction is
supported on a moved-from execution owner.
`generativeqc::check` and wrapper methods throw `generativeqc::Error` for
failed native status calls. `Error::status()` preserves the status code.
`Calculation::execute` checks supported properties; omitted forces are
represented by `std::optional` rather than an all-zero vector.

## Empirical SCF effort policy

`generativeqc_scf_effort_decide_v1` selects an already-defined strict/hold/relax/
tighten level and detached hysteresis state. It applies the supplied empirical
diagnostic envelope, joint-domain metadata and inclusive energy/force allowances.
Strict reproducibility, missing force and near-stationary cleanup precede
forecasting. Out-of-domain or nonfinite-forecast fallback is an explicit successful
`STRICT_FALLBACK` decision, not a failed native call or permission to run an
unqualified model. Integer counts and density-level storage are checked; native
storage is constant and level work is linear, with no physical SCF/device work.

`generativeqc_scf_effort_verify_v1` independently checks the measured final model,
strict-level, convergence, energy-change, density and force gates. It reports
every unmet gate; a model mismatch takes `unverified` precedence. Equality passes
each numeric gate. This does not replace the caller's actual strict cleanup solve.

Both protocols borrow inputs only for the call and publish detached scalar
records on success; errors preserve output. Complete ABI headers are required.
The caller owns calibration provenance, physical identity comparisons and leases;
neither an empirical estimate nor a caller-supplied identity flag authenticates a
physical calculation. See [accuracy policy](../user/accuracy.md#native-scf-effort-decisions)
and the source-owned [symbol contracts](native_symbols.md).

(response-recycle)=
## CPU response recycle vectors

`generativeqc_response_recycle_*` exposes a context-free CPU retained-vector
owner through the installed C header, also usable from C++. Creation owns only
metadata; shape-only capacity queries and whole-consumer admission precede
dimension-sized allocations. Initialize all complete ABI headers, including
nested strided-column descriptors. Signed strides locate logical values inside
declared storage using `origin`; empty columns need no storage.

Projection borrows one RHS and publishes a detached guess on success only.
Replacement visits old directions, solution, then basis using two-pass MGS and
the existing norm cutoff. Retained limits can change atomically without copying
old directions to the caller. Stale generations, invalid extents and insufficient
workspace do not publish new vectors; failed numerical attempts still contribute
to allocation telemetry. A diagnostic column-range copy remains valid after
replacement or destruction. Serialize operations and destruction per owner.

This numeric owner does not retain a physical operator or certify method/model
compatibility. The caller validates live scientific leases and updates only
accepted solves. Additional workspace excludes existing retained payload and
object/provider overhead; telemetry is not a complete physical endpoint receipt.
See the [response architecture](../developer/response.md#native-cpu-recycle-vectors)
and each function's [source-owned contract](native_symbols.md).

## Response controller

`generativeqc_response_gmres_workspace_v1` and
`generativeqc_response_solve_v1` expose the same CPU FP64 restarted GMRES owner
used by native method consumers. They require no context, Python interpreter,
compiler or device. The descriptor supplies a synchronous physical linear
action and an optional, separate linear right-preconditioner callback. These
callbacks do not define or infer CPHF, CPKS, Lambda or CC equations or symmetry.

The shape-only workspace query validates controls without reading vector or
callback pointers. Insufficient workspace publishes a diagnostic result without
dimension-sized solver allocations or callback work. API success means a result
was published, **not** that the solve converged: inspect `converged`, `reason`
and the true unpreconditioned residual. The host triangular backsolve rejects
zero/nonfinite pivots, nonfinite coefficients and nonfinite solution candidates;
small nonzero pivots alone do not reject a solution. A nonfinite candidate returns
breakdown with the best finite iterate, without invoking the operator on that
candidate. The existing absolute Arnoldi/Givens `breakdown_tolerance` checks remain,
so general operator-scale invariance is not guaranteed. Callback failure propagates
its status and leaves the result slot null. See the [symbol contracts](native_symbols.md)
for exact flags, layouts and failure behavior.

`generativeqc_response_result_view_v1_get` borrows immutable solution/history
arrays and a column-major Arnoldi basis until
`generativeqc_response_result_destroy`. The basis belongs to the reported best
iterate; diagnostic collection is opt-in. The move-only C++
`generativeqc::ResponseResult` wrapper owns the same publication. Inputs and
callbacks need survive only the synchronous solve, not result inspection.
Workspace bounds and measured peaks describe solver-owned numeric storage,
excluding callback/provider allocations and complete-endpoint resource use.
The [response guide](../developer/response.md#installed-solver-policies) defines
Python dispatch and the remaining compatibility domains.

## Native basis projection

`generativeqc_basis_projection_query_v1` validates bounded shape/policy metadata
and returns target coefficient/density counts and a conservative owner-visible
numeric working capacity. It reads no numerical arrays, allocates no numeric
workspace and initializes no provider. The caller may query even when its
workspace budget is insufficient. Input/output query descriptors must be disjoint.

`generativeqc_basis_project_cpu_v1` accepts either occupied source columns or a
pure integer-occupied HF density, rectangular target/source overlaps and a
versioned policy. It shares the existing occupied transport and ordinary CPU
algebra/eigen provider. C++ callers use the installed C header directly; no
additional public C++ owner class is required. Context accelerator selection
does not change this explicit host operation or add a device transfer/probe.

Execution refuses an undersized positive workspace limit before numerical-array
reads; zero is unbounded. Written coefficient/density ranges and the output
record must be disjoint from inputs and one another. ABI, buffer, admission,
scientific or provider failure preserves both output arrays and the output
record. Unused output tails remain unchanged on success. Empty occupied channels
publish zero density without inventing orbitals. This is a proposed seed, never
evidence of target SCF convergence or model identity. See the
[basis projection contract](../user/basis_projection.md).

The capacity covers owner-visible numeric state/results and ordinary eigen
scratch; caller arrays, metadata, allocator and provider-retained overhead remain
outside it. It is not complete physical memory authority or an RSS guarantee.

## Units and failure semantics

Coordinates use **Bohr**, energies **Hartree**, and forces **Hartree/Bohr**
unless the selected function specifies a different convention; nuclear
gradients are the negative of forces. Matrix and vector layouts remain
specific to their C descriptors. See [units](units.md).

`GENERATIVEQC_STATUS_SUCCESS` means the native call succeeded.
Other status codes distinguish invalid arguments, ABI mismatch,
unsupported operations, non-convergence, numerical failure,
CUDA failure, out-of-memory and internal errors. The special
`GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE` means a completed precision
provenance record is not available; it does not, by itself, mean a solve
failed. Use `generativeqc_status_message` for status explanations and
`generativeqc_context_get_last_detail` for contextual detail.

Batch execution may succeed structurally while individual member results fail.
Always inspect each input-indexed member's status and convergence fields before
using its numerical data; a missing force array is not a zero force.

For advanced Python/compiler extensions instead of C handles, consult
the [extension guide](../developer/extensions.md) and
[Python API](api.md).


## Execution, concurrency, and failed writes

Native execution completes synchronously before returning; output buffers are
caller-owned host storage unless a declaration explicitly says otherwise.
Serialize operations using the same owner/context, including multi-call
count/copy queries, execution, moves, and destruction. Internal locking is not
permission to race caller buffers or destruction. Independent immutable registry
queries do not require an execution context.

Failure behavior is entry-point-specific. A normal single-calculation result
publishes scalar diagnostics before returning NOT_CONVERGED, but a backend can
throw that status before producing a result (GFN2 is one example). Force
copy-out requires convergence. A rejected output descriptor can leave some
single-calculation diagnostic records from an earlier run available; batch
replay clears its C-handle precision-work, initial-guess, Fock-count, aggregate
precision/incremental-JK, SCF and KS records before validation. This is not a
blanket invalidation of method-owned diagnostics: HF FleetPlan profiles can
survive an early rejected replay. A batch item's invalid force buffer can be
detected after its native solve, leaving that solve's SCF/precision/KS and
correlation diagnostics available despite the item error.

For MP2/UMP2, a reference NOT_CONVERGED exception before native Result assignment
can still populate batch precision/incremental/work slots with default or empty
records. Query SUCCESS means the cached record was available and copied; it
does not certify that the record describes a completed observed execution.
Inspect item outcomes and work-completeness/counter-validity fields. Do not
infer record freshness or measured scientific work solely from a status code.

Count-query APIs differ: KS/precision-work validate all destinations before
writing, whereas eigensolver/DF/inactive-profile queries publish the required
count before returning a short-buffer error. Missing such profile records
return NOT_IMPLEMENTED, rather than a successful empty result. gCP outputs may
be changed on failure; nonlocal numerical outputs are staged transactionally.
The [individual contracts](native_symbols.md) are authoritative for these
exceptions and for NULL probe conventions.

## Maintaining coverage

Run `python tools/check_native_api_contracts.py` to check the reviewed
native declaration/facet inventory and exact generated-reference freshness.
Run `python -m unittest discover -s tests/python -p test_native_api_contracts.py`
for positive and negative regression fixtures. After reviewing a header contract
change, regenerate with
`python tools/check_native_api_contracts.py --write-reference`.

The checker requires explicit source-owned contract markers and applicable
behavior/input/output/lifetime/error/execution/unit facets. It follows no
optional native or GPU imports. New declarations, changed signatures, changed
C++ public record field types/defaults, and unknown declaration forms fail closed.
C descriptor fields are not inventoried by this declaration gate. Unsupported
friend declarations (including private ADL-visible friends), export aliases,
function-generating macros and direct visibility declarations are rejected
rather than silently omitted.
The inventory is an audited applicability policy, not a debt allowlist:
reviewers must verify semantics against the implementation and capability
sources when adding or changing contracts. Structured coverage cannot prove
that prose is scientifically correct.

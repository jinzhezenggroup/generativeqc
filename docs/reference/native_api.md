# Native C and C++ API

The stable native declarations live in
[`generativeqc.h`](https://github.com/jinzhezenggroup/generativeqc/blob/master/include/generativeqc/generativeqc.h).
The move-only C++ convenience wrappers are in
[`generativeqc.hpp`](https://github.com/jinzhezenggroup/generativeqc/blob/master/include/generativeqc/generativeqc.hpp).
Internal `src/` headers are not public API declarations.

The [complete symbol reference](native_symbols.md) contains stable targets and
source-owned contracts for every exported C function, every explicitly declared
public C++ operation, and all nine public C++ owner/record types. Record fields
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
